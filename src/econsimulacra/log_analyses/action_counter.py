from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import AnalyzerBase
from .store import RecordStore


class ActionCounter(AnalyzerBase[dict[str, int]]):
    """Action counter analyzer.

    ActionCounter counts the number of each action kinds
    of all records in the log.
    """

    name = "action_count"

    def analyze(self, store: RecordStore) -> dict[str, int]:
        """Counts the number of each action kinds.

        Args:
            store (RecordStore): The record store containing the records to analyze.
        
        Returns:
            counts (dict[str, int]): A dictionary mapping action types to counts.
                Action types are
                "move", "tweet", "follow", "unfollow", "order", "proposal",
                "consumption", "order_reaction", "proposal_reaction",
                "change_price".
        """
        counts: dict[str, int] = {
            "move": 0,
            "tweet": 0,
            "follow": 0,
            "unfollow": 0,
            "order": 0,
            "proposal": 0,
            "consumption": 0,
            "order_reaction": 0,
            "proposal_reaction": 0,
            "change_price": 0,
        }
        for type in counts.keys():
            counts[type] = len(store.get_by_type(type))
        return counts

    def draw_figs(
        self,
        result: dict[str, int],
    ) -> dict[str, Figure]:
        fig: Figure = Figure(figsize=(12, 6))
        ax: Axes = fig.add_subplot(1, 1, 1)
        action_names: list[str] = list(result.keys())
        counts: list[int] = [result[action] for action in action_names]
        ax.bar(action_names, counts)
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(
            action_names,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        return {"action_count": fig}
