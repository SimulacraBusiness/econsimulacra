from __future__ import annotations

from statistics import mean, median, stdev
from typing import TypeAlias

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import (
    BaseRecord,
    MoveRecord,
    OrderReactionRecord,
    TweetRecord,
)
from .store import RecordStore

AgentBehaviorStats: TypeAlias = dict[str, dict[int, float]]


class AgentBehaviorStatsAnalyzer(
    AnalyzerBase[AgentBehaviorStats, list[AgentBehaviorStats]]
):
    """Summarise each agent's economic and social behaviour over a simulation run.

    :class:`AgentBehaviorStatsAnalyzer` computes four scalar statistics for
    every agent across the entire simulation:

    **Total purchase price**
        Total monetary value of all goods bought via order reactions:

        .. math::

            S_{\\text{total}} =
            \\sum_{i} \\text{accept\\_amount}_i \\times \\text{price}_i

    **Average unit purchase price**
        Mean price paid per transaction:

        .. math::

            \\bar{p} = \\frac{1}{n} \\sum_{i=1}^{n} \\text{price}_i

    **Total move distance**
        Cumulative Euclidean displacement over all move events:

        .. math::

            L = \\sum_{j} \\sqrt{\\sum_{k}
            (\\text{new\\_pos}_{j,k} - \\text{old\\_pos}_{j,k})^2}

    **Total word count**
        Sum of word counts (whitespace-split) across all tweets.

    These statistics are useful for identifying behavioural heterogeneity:
    e.g., high-spending consumers, highly mobile agents, or unusually
    prolific tweeters. :meth:`draw_figs` plots a histogram for each
    statistic across all agents.
    """

    name = "agent_behavior_stats"

    def __init__(self, exclude_agent_ids: list[int] = []):
        """Initialization.

        Args:
            exclude_agent_ids (list[int], optional): A list of agent IDs to exclude from the analysis. Defaults to an empty list.
        """
        self.exclude_agent_ids = exclude_agent_ids

    def analyze(self, store: RecordStore) -> AgentBehaviorStats:
        """Compute per-agent behaviour statistics from *store*.

        Iterates over all agents found in
        :class:`~econsimulacra.log_analyses.records.AgentGenerationRecord`
        entries. For each agent, scans the corresponding records and
        accumulates:

        * ``"total_purchase_price"``:
          :math:`\\sum_i \\text{accept\\_amount}_i \\times \\text{price}_i`
          from :class:`~econsimulacra.log_analyses.records.OrderReactionRecord`.
        * ``"avg_unit_purchase_price"``: mean price per order-reaction record.
        * ``"total_move_distance"``: Euclidean distance summed over
          :class:`~econsimulacra.log_analyses.records.MoveRecord` entries.
        * ``"total_word_counts"``: word count summed over
          :class:`~econsimulacra.log_analyses.records.TweetRecord` entries.

        Agents in ``exclude_agent_ids`` are skipped.

        Args:
            store (RecordStore): Record store containing the simulation log.

        Returns:
            AgentBehaviorStats: Outer keys are stat names
            (``"total_purchase_price"``, ``"avg_unit_purchase_price"``,
            ``"total_move_distance"``, ``"total_word_counts"``). Inner keys
            are agent IDs (int) and values are the corresponding scalar
            statistics (float).
        """
        stats: AgentBehaviorStats = {
            "total_purchase_price": {},
            "avg_unit_purchase_price": {},
            "total_move_distance": {},
            "total_word_counts": {},
        }
        agent_id2name: dict[int, str] = self.get_agent_id2name(store)
        for agent_id in agent_id2name.keys():
            if agent_id in self.exclude_agent_ids:
                continue
            records: list[BaseRecord] = store.get_by_agent(agent_id)
            total_purchase_price: float = 0.0
            total_unit_price: float = 0.0
            purchase_count: int = 0
            total_move_distance: float = 0.0
            total_word_counts: int = 0
            for record in records:
                if isinstance(record, OrderReactionRecord):
                    total_purchase_price += record.accept_amount * record.price
                    total_unit_price += record.price
                    purchase_count += 1
                elif isinstance(record, MoveRecord):
                    old_pos: tuple[int, ...] = record.old_pos
                    new_pos: tuple[int, ...] = record.new_pos
                    total_move_distance += (
                        sum((new - old) ** 2 for old, new in zip(old_pos, new_pos))
                        ** 0.5
                    )
                elif isinstance(record, TweetRecord):
                    total_word_counts += len(record.message.split())
            stats["total_purchase_price"][agent_id] = total_purchase_price
            stats["avg_unit_purchase_price"][agent_id] = (
                (total_unit_price / purchase_count) if purchase_count > 0 else 0.0
            )
            stats["total_move_distance"][agent_id] = total_move_distance
            stats["total_word_counts"][agent_id] = total_word_counts
        return stats

    def analyze_stores(self, stores: list[RecordStore]) -> list[AgentBehaviorStats]:
        """Compute per-agent behaviour statistics for multiple stores.

        Calls :meth:`analyze` independently for each store. The resulting
        list preserves the order of *stores*.

        Args:
            stores (list[RecordStore]): One record store per simulation run.

        Returns:
            list[AgentBehaviorStats]: One :class:`AgentBehaviorStats` dict
            per store. Use :meth:`draw_figs_all` to produce a box plot
            comparing stat distributions across runs.
        """
        stats_list: list[AgentBehaviorStats] = []
        for store in stores:
            stats: AgentBehaviorStats = self.analyze(store)
            stats_list.append(stats)
        return stats_list

    def draw_figs(
        self,
        result: AgentBehaviorStats,
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        for stat_name, agent_id2stat in result.items():
            fig: Figure = Figure(figsize=(8, 6))
            ax: Axes = fig.add_subplot(1, 1, 1)
            stat_values: list[float] = list(agent_id2stat.values())
            ax.hist(stat_values, bins=50)
            ax.set_xlabel(stat_name)
            ax.set_ylabel("Count")
            fig_dic[stat_name] = fig
        return fig_dic

    def draw_figs_all(
        self, individual_results: list[AgentBehaviorStats]
    ) -> dict[str, Figure]:
        stat_names: list[str] = list(individual_results[0].keys())
        data_for_all_stats: list[list[float]] = []
        fig: Figure = Figure(figsize=(12, 6))
        ax: Axes = fig.add_subplot(1, 1, 1)
        for stat_name in stat_names:
            data: list[float] = []
            for result in individual_results:
                agent_id2stat: dict[int, float] = result[stat_name]
                stat_values: list[float] = list(agent_id2stat.values())
                data.extend(stat_values)
            data_for_all_stats.append(data)
        ax.boxplot(data_for_all_stats, tick_labels=stat_names)
        ax.set_xticklabels(stat_names, rotation=45, ha="right", rotation_mode="anchor")
        return {
            "all_stats": fig,
        }

    def build_summary(
        self,
        result: AgentBehaviorStats,
    ) -> RenderableType:
        """Build a rich summary table for agent behavior stats."""
        table = Table(
            title="Agent Behavior Stats",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Stat")
        table.add_column("Mean", justify="right")
        table.add_column("Median", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("N", justify="right")
        for stat_name, agent_id2stat in result.items():
            values: list[float] = list(agent_id2stat.values())
            if not values:
                table.add_row(stat_name, "-", "-", "-", "0")
                continue
            avg: float = mean(values)
            med: float = median(values)
            std: float = stdev(values) if len(values) >= 2 else 0.0
            table.add_row(
                stat_name,
                f"{avg:.3f}",
                f"{med:.3f}",
                f"{std:.3f}",
                f"{len(values):,}",
            )
        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
