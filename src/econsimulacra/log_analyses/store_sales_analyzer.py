from __future__ import annotations

from typing import TypeAlias

import matplotlib.pyplot as plt
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

SalesResult: TypeAlias = dict[str, dict[int, float]]
SalesAmountResult: TypeAlias = dict[str, dict[int, float]]


class StoreSalesAnalyzer(AnalyzerBase[tuple[SalesResult, SalesAmountResult], None]):
    """Analyse revenue and unit sales for each store over simulation time.

    A "store" is any agent that acts as the counterparty in an
    :class:`~econsimulacra.log_analyses.records.OrderReactionRecord` or as
    the responder in an accepted
    :class:`~econsimulacra.log_analyses.records.ProposalReactionRecord`.
    For each such store this analyzer aggregates two quantities per time step:

    **Revenue (sales)**
        The total monetary value of goods sold:

        .. math::

            \\text{sales}_{\\text{store},t} =
            \\sum_{i \\,:\\, t_i = t}
            \\text{accept\\_amount}_i \\times \\text{price}_i

    **Unit sales (sold amount)**
        The total physical quantity of goods sold:

        .. math::

            \\text{sold\\_amount}_{\\text{store},t} =
            \\sum_{i \\,:\\, t_i = t} \\text{accept\\_amount}_i

    Both quantities are plotted as bar charts over time by :meth:`draw_figs`.
    :meth:`build_summary` prints a revenue ranking across all stores,
    excluding household agents.
    """

    name = "store_sales"

    def analyze(
        self,
        store: RecordStore,
    ) -> tuple[SalesResult, SalesAmountResult]:
        """Aggregate per-store revenue and unit sales from transaction records.

        Processes :class:`~econsimulacra.log_analyses.records.OrderReactionRecord`
        and accepted
        :class:`~econsimulacra.log_analyses.records.ProposalReactionRecord`
        entries. For order reactions the store is identified as
        ``counterparty_id``; for accepted proposals it is
        ``responder_agent_id``.

        Revenue per time step is accumulated as:

        .. math::

            \\text{sales}_{t} = \\sum_{i \\,:\\, t_i = t}
            \\text{accept\\_amount}_i \\times \\text{price}_i

        Args:
            store (RecordStore): Record store containing transaction records.

        Returns:
            tuple[SalesResult, SalesAmountResult]: A 2-tuple where

            * ``sales`` maps store names to ``{time_step: total_revenue}``.
            * ``sold_amounts`` maps store names to
              ``{time_step: total_units_sold}``.

        Raises:
            ValueError: If a transaction references an agent ID that is not
                present in :class:`~econsimulacra.log_analyses.records.AgentGenerationRecord`.
        """
        self._prepare_time_axis(store)
        agent_id2name: dict[int, str] = self.get_agent_id2name(store)
        sales: SalesResult = {}
        sold_amounts: SalesAmountResult = {}

        order_reactions: list[OrderReactionRecord] = store.typed(OrderReactionRecord)
        proposal_reactions: list[ProposalReactionRecord] = store.typed(
            ProposalReactionRecord
        )

        for order_reaction in order_reactions:
            firm_id = order_reaction.counterparty_id
            if firm_id not in agent_id2name:
                raise ValueError(f"Agent ID {firm_id} not found.")

            firm_name = agent_id2name[firm_id]
            time_step = order_reaction.time_step

            if firm_name not in sales:
                sales[firm_name] = {}
                sold_amounts[firm_name] = {}

            if time_step not in sales[firm_name]:
                sales[firm_name][time_step] = 0.0
                sold_amounts[firm_name][time_step] = 0.0

            sales[firm_name][time_step] += (
                order_reaction.accept_amount * order_reaction.price
            )
            sold_amounts[firm_name][time_step] += order_reaction.accept_amount

        for proposal_reaction in proposal_reactions:
            if not proposal_reaction.accept:
                continue

            firm_id = proposal_reaction.responder_agent_id
            if firm_id not in agent_id2name:
                raise ValueError(f"Agent ID {firm_id} not found.")

            firm_name = agent_id2name[firm_id]
            time_step = proposal_reaction.time_step

            if firm_name not in sales:
                sales[firm_name] = {}
                sold_amounts[firm_name] = {}

            if time_step not in sales[firm_name]:
                sales[firm_name][time_step] = 0.0
                sold_amounts[firm_name][time_step] = 0.0

            sales[firm_name][time_step] += proposal_reaction.give_item_amount
            sold_amounts[firm_name][time_step] += proposal_reaction.give_item_amount

        return sales, sold_amounts

    def analyze_stores(self, stores: list[RecordStore]) -> None:
        return None

    def draw_figs(
        self,
        result: tuple[SalesResult, SalesAmountResult],
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        sales_by_store, sold_amounts_by_store = result
        store_names: set[str] = set(sales_by_store.keys()) | set(
            sold_amounts_by_store.keys()
        )
        for store_name in store_names:
            time2sales = sales_by_store.get(store_name, {})
            time2sold_amounts = sold_amounts_by_store.get(store_name, {})
            times: list[int] = sorted(
                set(time2sales.keys()) | set(time2sold_amounts.keys())
            )
            sales_values: list[float] = [time2sales.get(time, 0.0) for time in times]
            sold_amount_values: list[float] = [
                time2sold_amounts.get(time, 0.0) for time in times
            ]
            last_time: int = times[-1] if times else 0
            if self._time_axis_config is not None:
                max_time = int(self._time_axis_config.x_max)
                if last_time < max_time:
                    times.append(max_time)
                    sales_values.append(0.0)
                    sold_amount_values.append(0.0)
            fig_sales: Figure
            ax_sales: Axes
            fig_sales, ax_sales = plt.subplots(figsize=(8, 6))
            ax_sales.bar(times, sales_values)
            ax_sales.set_xlabel("Time")
            ax_sales.set_ylabel("Sales Amount")
            ax_sales.set_title(f"Sales Amount: {store_name}")
            fig_sold_amounts: Figure
            ax_sold_amounts: Axes
            fig_sold_amounts, ax_sold_amounts = plt.subplots(figsize=(8, 6))
            ax_sold_amounts.bar(times, sold_amount_values)
            ax_sold_amounts.set_xlabel("Time")
            ax_sold_amounts.set_ylabel("Sold Amount")
            ax_sold_amounts.set_title(f"Sold Amount: {store_name}")
            self._apply_time_axis(ax_sales)
            self._apply_time_axis(ax_sold_amounts)

            fig_dic[f"store_sales_{store_name}"] = fig_sales
            fig_dic[f"store_sold_amount_{store_name}"] = fig_sold_amounts

        return fig_dic

    def draw_figs_all(
        self, individual_results: list[tuple[SalesResult, SalesAmountResult]]
    ) -> dict[str, Figure]:
        return {}

    def build_summary(
        self,
        result: tuple[SalesResult, SalesAmountResult],
    ) -> RenderableType:
        """Build a rich summary table for total store sales and sold amounts."""
        sales_by_store, sold_amounts_by_store = result

        store_names: set[str] = set(sales_by_store.keys()) | set(
            sold_amounts_by_store.keys()
        )

        store_name2totals: dict[str, tuple[float, float]] = {
            store_name: (
                sum(sales_by_store.get(store_name, {}).values()),
                sum(sold_amounts_by_store.get(store_name, {}).values()),
            )
            for store_name in store_names
            if "Household" not in store_name
        }

        sorted_items: list[tuple[str, tuple[float, float]]] = sorted(
            store_name2totals.items(),
            key=lambda x: x[1][0],
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
        table.add_column("Sold Amount", justify="right")

        for rank, (store_name, (total_sales, total_sold_amount)) in enumerate(
            sorted_items,
            start=1,
        ):
            table.add_row(
                str(rank),
                store_name,
                f"{total_sales:,.3f}",
                f"{total_sold_amount:,.3f}",
            )

        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
