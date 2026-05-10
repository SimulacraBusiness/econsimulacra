from __future__ import annotations

from datetime import datetime
from typing import cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import (
    OrderReactionRecord,
    ProposalReactionRecord,
)
from .store import RecordStore


class StoreSalesAnalyzer(AnalyzerBase[dict[str, dict[datetime | int, float]]]):
    """Sales analyzer.

    StoreSalesAnalyzer analyzes sales data for each firm
    based on OrderReactionRecord and ProposalReactionRecord.
    """

    name = "store_sales"

    def analyze(self, store: RecordStore) -> dict[str, dict[datetime | int, float]]:
        """Analyzes sales data for each firm.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            dict[str, dict[datetime | int, float]]: A dictionary where keys are firm IDs and
                values are dictionaries mapping timestamps to total sales amounts.
        """
        agent_id2name: dict[int, str] = self.get_agent_id2name(store)
        sales: dict[str, dict[datetime | int, float]] = {}
        order_reactions: list[OrderReactionRecord] = store.typed(OrderReactionRecord)
        proposal_reactions: list[ProposalReactionRecord] = store.typed(
            ProposalReactionRecord
        )
        time: datetime | int
        firm_id: int
        firm_name: str
        for order_reaction in order_reactions:
            firm_id = order_reaction.counterparty_id
            if firm_id not in agent_id2name:
                raise ValueError(f"Agent ID {firm_id} not found.")
            firm_name = agent_id2name[firm_id]
            time = order_reaction.time
            if firm_name not in sales:
                sales[firm_name] = {}
            if time not in sales[firm_name]:
                sales[firm_name][time] = 0.0
            sales[firm_name][time] += order_reaction.accept_amount * (
                order_reaction.price
            )
        for proposal_reaction in proposal_reactions:
            if not proposal_reaction.accept:
                continue
            firm_id = proposal_reaction.responder_agent_id
            if firm_id not in agent_id2name:
                raise ValueError(f"Agent ID {firm_id} not found.")
            firm_name = agent_id2name[firm_id]
            time = proposal_reaction.time
            if firm_name not in sales:
                continue
            if time not in sales[firm_name]:
                sales[firm_name][time] = 0.0
            sales[firm_name][time] += proposal_reaction.give_item_amount
        return sales

    def draw_figs(
        self,
        result: dict[str, dict[datetime | int, float]],
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        for firm_name, time_sales in result.items():
            fig: Figure = Figure(figsize=(10, 6))
            ax: Axes = fig.add_subplot(1, 1, 1)
            times: list[datetime | int] = sorted(time_sales.keys())
            sales: list[float] = [time_sales[time] for time in times]
            x = np.arange(len(times))
            ax.bar(x, sales)
            ax.set_xticks(x)
            num_ticks: int = min(10, len(times))
            step: int = max(1, len(times) // num_ticks)
            tick_positions = x[::step]
            tick_times = times[::step]
            ax.set_xticks(tick_positions)
            if tick_times and isinstance(tick_times[0], datetime):
                datetime_tick_times = cast(list[datetime], tick_times)
                ax.set_xticklabels(
                    [time.strftime("%Y-%m-%d") for time in datetime_tick_times],
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )
            else:
                ax.set_xticklabels(
                    [str(time) for time in tick_times],
                    rotation=45,
                    ha="right",
                    rotation_mode="anchor",
                )
            ax.set_xlabel("Time")
            ax.set_ylabel("Sales Amount")
            fig_dic[f"store_sales_{firm_name}"] = fig
        return fig_dic

    def build_summary(
        self,
        result: dict[str, dict[datetime | int, float]],
    ) -> RenderableType:
        """Build a rich summary table for total store sales."""

        firm_name2total_sales: dict[str, float] = {
            firm_name: sum(time_sales.values())
            for firm_name, time_sales in result.items()
            if "Household" not in firm_name
        }

        sorted_items: list[tuple[str, float]] = sorted(
            firm_name2total_sales.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        table = Table(
            title="Store Sales Ranking",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Rank", justify="right")
        table.add_column("Store")
        table.add_column("Total Sales", justify="right")

        for rank, (firm_name, total_sales) in enumerate(sorted_items, start=1):
            table.add_row(
                str(rank),
                firm_name,
                f"{total_sales:,.3f}",
            )

        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
