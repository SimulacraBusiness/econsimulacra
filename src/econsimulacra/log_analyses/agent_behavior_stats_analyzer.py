from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from econsimulacra.log_analyses.records import (
    MoveRecord,
    OrderReactionRecord,
    TweetRecord,
)

from .base import AnalyzerBase
from .store import RecordStore


class AgentBehaviorStatsAnalyzer(AnalyzerBase[dict[str, dict[int, float]]]):
    """Agent behavior stats analyzer.

    AgentBehaviorStatsAnalyzer summarizes the behavior of each agent in the log.
    """

    name = "agent_behavior_stats"

    def analyze(self, store: RecordStore) -> dict[str, dict[int, float]]:
        """Summarizes the behavior of each agent in the log.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            stats (dict[str, dict[int, float]]]): A dictionary mapping stat names to dictionaries
                that map agent IDs to the stat values.

        Note:
            The stat names are:
            "total_purchase_price": Total purchase price for each agent.
            "avg_unit_purchase_price": Average unit purchase price for each agent.
            "total_move_distance": Total move distance for each agent.
            "total_word_counts": Total word counts in tweets for each agent.
        """
        stats: dict[str, dict[int, float]] = {
            "total_purchase_price": {},
            "avg_unit_purchase_price": {},
            "total_move_distance": {},
            "total_word_counts": {},
        }
        for react_record in store.typed(OrderReactionRecord):
            agent_id = react_record.agent_id
            if agent_id not in stats["total_purchase_price"]:
                stats["total_purchase_price"][agent_id] = 0.0
                stats["avg_unit_purchase_price"][agent_id] = 0.0
            stats["total_purchase_price"][agent_id] += (
                react_record.accept_amount * react_record.price
            )
            stats["avg_unit_purchase_price"][agent_id] += react_record.price
        for move_record in store.typed(MoveRecord):
            agent_id = move_record.agent_id
            if agent_id not in stats["total_move_distance"]:
                stats["total_move_distance"][agent_id] = 0.0
            old_pos: tuple[int, ...] = move_record.old_pos
            new_pos: tuple[int, ...] = move_record.new_pos
            stats["total_move_distance"][agent_id] += (
                sum((new - old) ** 2 for old, new in zip(old_pos, new_pos)) ** 0.5
            )
        for tweet_record in store.typed(TweetRecord):
            agent_id = tweet_record.agent_id
            if agent_id not in stats["total_word_counts"]:
                stats["total_word_counts"][agent_id] = 0.0
            stats["total_word_counts"][agent_id] += len(tweet_record.message.split())
        return stats

    def draw_figs(
        self,
        result: dict[str, dict[int, float]],
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        for stat_name, agent_id2stat in result.items():
            fig: Figure = Figure(figsize=(12, 6))
            ax: Axes = fig.add_subplot(1, 1, 1)
            stat_values: list[float] = list(agent_id2stat.values())
            ax.hist(stat_values, bins=50)
            ax.set_xlabel(stat_name)
            ax.set_ylabel("Count")
            fig_dic[stat_name] = fig
        return fig_dic
