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
    """Agent behavior stats analyzer.

    AgentBehaviorStatsAnalyzer summarizes the behavior of each agent in the log.
    """

    name = "agent_behavior_stats"

    def __init__(self, exclude_agent_ids: list[int] = []):
        """Initialization.

        Args:
            exclude_agent_ids (list[int], optional): A list of agent IDs to exclude from the analysis. Defaults to an empty list.
        """
        self.exclude_agent_ids = exclude_agent_ids

    def analyze(self, store: RecordStore) -> AgentBehaviorStats:
        """Summarizes the behavior of each agent in the log.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            stats (AgentBehaviorStats): A dictionary mapping stat names to dictionaries
                that map agent IDs to the stat values.

        Note:
            The stat names are:
            "total_purchase_price": Total purchase price for each agent.
            "avg_unit_purchase_price": Average unit purchase price for each agent.
            "total_move_distance": Total move distance for each agent.
            "total_word_counts": Total word counts in tweets for each agent.
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
        """Summarizes the behavior of each agent across multiple stores.

        Args:
            stores (list[RecordStore]): A list of record stores to analyze.

        Returns:
            stats_list (list[AgentBehaviorStats]): A list of AgentBehaviorStats for each store.
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
