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
from .records import ChangePriceRecord, ItemGenerationRecord
from .store import RecordStore


class PriceAnalyzer(AnalyzerBase[dict[str, dict[datetime | int, float]]]):
    """Price analyzer.

    PriceAnalyzer tracks the prices of items over time.
    """

    name = "price"

    def analyze(self, store: RecordStore) -> dict[str, dict[datetime | int, float]]:
        """Tracks the prices of items over time.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            A dictionary mapping item names to a dictionary of timestamps and prices.
        """
        item_prices: dict[str, dict[datetime | int, float]] = {}
        record: ChangePriceRecord | ItemGenerationRecord

        for record in store.typed(ItemGenerationRecord) + store.typed(
            ChangePriceRecord
        ):
            item_name = record.item_name
            time = record.time
            price: float
            if isinstance(record, ChangePriceRecord):
                price = record.new_price
            else:
                price = record.price
            if item_name not in item_prices:
                item_prices[item_name] = {}
            item_prices[item_name][time] = price

        return item_prices

    def draw_figs(
        self,
        result: dict[str, dict[datetime | int, float]],
    ) -> dict[str, Figure]:
        """Draw price time series figures for each item.

        Args:
            result (dict[str, dict[datetime | int, float]]): A dictionary mapping item names to dictionaries of
                timestamps and prices.

        Returns:
            A dictionary mapping figure names to matplotlib Figure objects.
        """
        fig_dic: dict[str, Figure] = {}

        for item_name, time2price in result.items():
            if not time2price:
                continue

            times: list[datetime | int] = sorted(time2price.keys())
            prices: list[float] = [time2price[time] for time in times]
            x = np.arange(len(times))

            fig: Figure = Figure(figsize=(10, 6))
            ax: Axes = fig.add_subplot(1, 1, 1)

            ax.plot(x, prices, marker="o")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.set_title(f"Price: {item_name}")

            num_ticks: int = min(10, len(times))
            step: int = max(1, len(times) // num_ticks)
            tick_positions = x[::step]
            tick_times = times[::step]

            ax.set_xticks(tick_positions)

            if tick_times and isinstance(tick_times[0], datetime):
                datetime_tick_times = cast(list[datetime], tick_times)
                ax.set_xticklabels(
                    [time.strftime("%Y-%m-%d %H:%M") for time in datetime_tick_times],
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

            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            fig_dic[f"price_{item_name}"] = fig

        return fig_dic

    def build_summary(
        self,
        result: dict[str, dict[datetime | int, float]],
    ) -> RenderableType:
        """Build a rich summary table for item prices."""
        table = Table(
            title="Item Price Summary",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Item")
        table.add_column("Initial Price", justify="right")
        table.add_column("Final Price", justify="right")
        table.add_column("Change", justify="right")
        table.add_column("Change Rate", justify="right")
        table.add_column("Num Updates", justify="right")

        for item_name, time2price in sorted(result.items()):
            if not time2price:
                continue

            times: list[datetime | int] = sorted(time2price.keys())

            initial_price: float = time2price[times[0]]
            final_price: float = time2price[times[-1]]
            price_change: float = final_price - initial_price

            if initial_price == 0:
                change_rate_str = "N/A"
            else:
                change_rate: float = price_change / initial_price
                change_rate_str = f"{change_rate:.3%}"

            table.add_row(
                item_name,
                f"{initial_price:,.3f}",
                f"{final_price:,.3f}",
                f"{price_change:,.3f}",
                change_rate_str,
                str(len(times)),
            )

        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
