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
from .records import OrderReactionRecord
from .store import RecordStore


class ItemSalesAnalyzer(AnalyzerBase[dict[str, dict[datetime | int, float]]]):
    """Item sales analyzer.

    ItemSalesAnalyzer analyzes sales data for each item
    based on OrderReactionRecord and ProposalReactionRecord.
    """

    name = "item_sales"

    def analyze(self, store: RecordStore) -> dict[str, dict[datetime | int, float]]:
        """Analyzes sales data for each item.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            dict[str, dict[datetime | int, float]]: A dictionary mapping item names to their sales data.
        """
        sales: dict[str, dict[datetime | int, float]] = {}
        order_reactions: list[OrderReactionRecord] = store.typed(OrderReactionRecord)
        time: datetime | int
        item_name: str
        for order_reaction in order_reactions:
            item_name = order_reaction.item_name
            time = order_reaction.time
            if item_name not in sales:
                sales[item_name] = {}
            if time not in sales[item_name]:
                sales[item_name][time] = 0.0
            sales[item_name][time] += order_reaction.accept_amount * (
                order_reaction.price
            )
        return sales

    def draw_figs(
        self,
        result: dict[str, dict[datetime | int, float]],
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        for item_name, time_sales in result.items():
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
            fig.tight_layout()
            fig_dic[f"item_sales_{item_name}"] = fig
        return fig_dic

    def build_summary(
        self,
        result: dict[str, dict[datetime | int, float]],
    ) -> RenderableType:
        """Build a rich summary table for total item sales."""
        item_name2total_sales: dict[str, float] = {
            item_name: sum(time_sales.values())
            for item_name, time_sales in result.items()
        }
        sorted_items: list[tuple[str, float]] = sorted(
            item_name2total_sales.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        table = Table(
            title="Item Sales Ranking",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Rank", justify="right")
        table.add_column("Item")
        table.add_column("Total Sales", justify="right")
        for rank, (item_name, total_sales) in enumerate(sorted_items, start=1):
            table.add_row(
                str(rank),
                item_name,
                f"{total_sales:,.3f}",
            )
        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
