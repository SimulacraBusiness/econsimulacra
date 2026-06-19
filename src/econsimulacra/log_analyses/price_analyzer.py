from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import ChangePriceRecord, ItemGenerationRecord
from .store import RecordStore


class PriceAnalyzer(AnalyzerBase[dict[str, dict[int, float]], None]):
    """Track the listed price of each item over simulation time.

    Price changes in EconSimulacra are triggered by two event types:

    * :class:`~econsimulacra.log_analyses.records.ItemGenerationRecord` –
      sets the initial price when an item is introduced.
    * :class:`~econsimulacra.log_analyses.records.ChangePriceRecord` –
      records every subsequent price update by a seller agent.

    :class:`PriceAnalyzer` collects all such events and returns a step
    function :math:`p_{\\text{item}}(t)` for each item, where :math:`t`
    is the simulation step at which the price was last set.

    :meth:`draw_figs` renders each item's price trajectory using a
    step-wise line plot (``where="post"``), faithfully representing that
    prices remain constant between updates.

    :meth:`build_summary` reports for each item: initial price, final price,
    absolute change, relative change :math:`(p_{\\text{final}} - p_{\\text{init}}) / p_{\\text{init}}`,
    and the total number of update events.
    """

    name = "price"

    def analyze(self, store: RecordStore) -> dict[str, dict[int, float]]:
        """Collect the price history for every item in *store*.

        Combines
        :class:`~econsimulacra.log_analyses.records.ItemGenerationRecord`
        (initial prices) and
        :class:`~econsimulacra.log_analyses.records.ChangePriceRecord`
        (subsequent updates). When multiple events occur at the same
        ``time_step`` for the same item, the last one processed overwrites
        earlier ones.

        Args:
            store (RecordStore): Record store containing item-generation and
                price-change records.

        Returns:
            dict[str, dict[int, float]]: Mapping from item name to a
            ``{time_step: price}`` dict representing the step function
            :math:`p_{\\text{item}}(t)`.
        """
        self._prepare_time_axis(store)
        item_prices: dict[str, dict[int, float]] = {}
        record: ChangePriceRecord | ItemGenerationRecord

        for record in store.typed(ItemGenerationRecord) + store.typed(
            ChangePriceRecord
        ):
            item_name = record.item_name
            time_step = record.time_step
            price: float
            if isinstance(record, ChangePriceRecord):
                price = record.new_price
            else:
                price = record.price
            if item_name not in item_prices:
                item_prices[item_name] = {}
            item_prices[item_name][time_step] = price

        return item_prices

    def analyze_stores(self, stores: list[RecordStore]) -> None:
        return None

    def draw_figs(
        self,
        result: dict[str, dict[int, float]],
    ) -> dict[str, Figure]:
        """Draw price time series figures for each item.

        Args:
            result (dict[str, dict[int, float]]): A dictionary mapping item names to dictionaries of
                timestamps and prices.

        Returns:
            A dictionary mapping figure names to matplotlib Figure objects.
        """
        fig_dic: dict[str, Figure] = {}

        for item_name, time2price in result.items():
            if not time2price:
                continue

            fig: Figure
            ax: Axes
            fig, ax = plt.subplots(figsize=(8, 6))

            times: list[int] = sorted(time2price.keys())
            last_time: int = times[-1]

            plot_time2price = dict(time2price)

            if self._time_axis_config is not None:
                max_time = int(self._time_axis_config.x_max)
                if last_time < max_time:
                    plot_time2price[max_time] = plot_time2price[last_time]

            times = sorted(plot_time2price.keys())
            prices: list[float] = [plot_time2price[t] for t in times]
            x_values: list[float] = [float(t) for t in times]

            ax.step(x_values, prices, where="post", marker="o")
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            self._apply_time_axis(ax)

            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            fig_dic[f"price_{item_name}"] = fig

        return fig_dic

    def draw_figs_all(
        self, individual_results: list[dict[str, dict[int, float]]]
    ) -> dict[str, Figure]:
        return {}

    def build_summary(
        self,
        result: dict[str, dict[int, float]],
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

            times: list[int] = sorted(time2price.keys())

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
