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


class ItemSalesAnalyzer(
    AnalyzerBase[
        tuple[
            dict[str, dict[datetime | int, float]],
            dict[str, dict[datetime | int, float]],
        ]
    ]
):
    """Item sales analyzer.

    ItemSalesAnalyzer analyzes sales data for each item
    based on OrderReactionRecord and ProposalReactionRecord.
    """

    name = "item_sales"

    def analyze(
        self, store: RecordStore
    ) -> tuple[
        dict[str, dict[datetime | int, float]], dict[str, dict[datetime | int, float]]
    ]:
        """Analyzes sales data for each item.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            tuple[dict[str, dict[datetime | int, float]], dict[str, dict[datetime | int, float]]]: A tuple containing dictionaries mapping item names to their sales data and sold amounts.
        """
        sales: dict[str, dict[datetime | int, float]] = {}
        sold_amounts: dict[str, dict[datetime | int, float]] = {}
        order_reactions: list[OrderReactionRecord] = store.typed(OrderReactionRecord)
        time: datetime | int
        item_name: str
        for order_reaction in order_reactions:
            item_name = order_reaction.item_name
            time = order_reaction.time
            if item_name not in sales:
                sales[item_name] = {}
                sold_amounts[item_name] = {}
            if time not in sales[item_name]:
                sales[item_name][time] = 0.0
                sold_amounts[item_name][time] = 0.0
            sales[item_name][time] += order_reaction.accept_amount * (
                order_reaction.price
            )
            sold_amounts[item_name][time] += order_reaction.accept_amount
        return sales, sold_amounts

    def draw_figs(
        self,
        result: tuple[
            dict[str, dict[datetime | int, float]],
            dict[str, dict[datetime | int, float]],
        ],
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        sales_by_item, sold_amounts_by_item = result
        item_names: set[str] = set(sales_by_item.keys()) | set(
            sold_amounts_by_item.keys()
        )
        for item_name in item_names:
            time2sales = sales_by_item.get(item_name, {})
            time2sold_amounts = sold_amounts_by_item.get(item_name, {})
            times: list[datetime | int] = sorted(
                set(time2sales.keys()) | set(time2sold_amounts.keys())
            )
            x = np.arange(len(times))
            sales_values: list[float] = [time2sales.get(time, 0.0) for time in times]
            sold_amount_values: list[float] = [
                time2sold_amounts.get(time, 0.0) for time in times
            ]
            fig_sales: Figure = Figure(figsize=(10, 6))
            ax_sales: Axes = fig_sales.add_subplot(1, 1, 1)
            ax_sales.bar(x, sales_values)
            ax_sales.set_xlabel("Time")
            ax_sales.set_ylabel("Sales Amount")
            fig_sold_amounts: Figure = Figure(figsize=(10, 6))
            ax_sold_amounts: Axes = fig_sold_amounts.add_subplot(1, 1, 1)
            ax_sold_amounts.bar(x, sold_amount_values)
            ax_sold_amounts.set_xlabel("Time")
            ax_sold_amounts.set_ylabel("Sold Amount")
            num_ticks: int = min(10, len(times))
            step: int = max(1, len(times) // num_ticks)
            tick_positions = x[::step]
            tick_times = times[::step]
            for ax in [ax_sales, ax_sold_amounts]:
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
            fig_sales.tight_layout()
            fig_sold_amounts.tight_layout()

            fig_dic[f"item_sales_amount_{item_name}"] = fig_sales
            fig_dic[f"item_sold_amount_{item_name}"] = fig_sold_amounts

        return fig_dic

    def build_summary(
        self,
        result: tuple[
            dict[str, dict[datetime | int, float]],
            dict[str, dict[datetime | int, float]],
        ],
    ) -> RenderableType:
        """Build a rich summary table for total item sales and sold amounts."""
        sales_by_item, sold_amounts_by_item = result

        item_names: set[str] = set(sales_by_item.keys()) | set(
            sold_amounts_by_item.keys()
        )

        item_name2totals: dict[str, tuple[float, float]] = {
            item_name: (
                sum(sales_by_item.get(item_name, {}).values()),
                sum(sold_amounts_by_item.get(item_name, {}).values()),
            )
            for item_name in item_names
        }

        sorted_items: list[tuple[str, tuple[float, float]]] = sorted(
            item_name2totals.items(),
            key=lambda x: x[1][0],
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
        table.add_column("Sold Amount", justify="right")

        for rank, (item_name, (total_sales, total_sold_amount)) in enumerate(
            sorted_items,
            start=1,
        ):
            table.add_row(
                str(rank),
                item_name,
                f"{total_sales:,.3f}",
                f"{total_sold_amount:,.3f}",
            )

        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
