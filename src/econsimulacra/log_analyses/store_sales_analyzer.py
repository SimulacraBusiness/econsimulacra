from __future__ import annotations

from datetime import datetime

import matplotlib.dates as mdates
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

    name = "sales"

    def analyze(self, store: RecordStore) -> dict[str, dict[datetime | int, float]]:
        """Analyzes sales data for each firm.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            A dictionary where keys are firm IDs and
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
        fig: Figure = Figure(figsize=(10, 6))
        ax: Axes = fig.add_subplot(1, 1, 1)
        for firm_name, time_sales in result.items():
            if "Household" in firm_name:
                continue
            times = sorted(list(time_sales.keys()))
            sales = [time_sales[time] for time in times]
            ax.plot(np.array(times), sales, label=firm_name)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=10))  # type: ignore
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))  # type: ignore
        ax.set_ylabel("Sales Amount")
        ax.legend()
        return {"sales": fig}
    
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
