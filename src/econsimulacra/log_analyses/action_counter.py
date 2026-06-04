from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .store import RecordStore


class ActionCounter(AnalyzerBase[dict[str, int], dict[str, list[int]]]):
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
                "sleep_end", "move", "tweet", "follow", "unfollow", "inner_thought",
                "order", "proposal", "consumption", "order_reaction", "proposal_reaction",
                "change_price".
        """
        counts: dict[str, int] = {
            "sleep_end": 0,
            "move": 0,
            "tweet": 0,
            "follow": 0,
            "unfollow": 0,
            "inner_thought": 0,
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

    def analyze_stores(self, stores: list[RecordStore]) -> dict[str, list[int]]:
        """Counts the number of each action kinds across multiple stores.

        Args:
            stores (list[RecordStore]): A list of record stores to analyze.

        Returns:
            counts (dict[str, list[int]]): A dictionary mapping action types to lists of counts.
                Action types are
                "sleep_end", "move", "tweet", "follow", "unfollow", "inner_thought",
                "order", "proposal", "consumption", "order_reaction", "proposal_reaction",
                "change_price".
        """
        total_counts: dict[str, list[int]] = {
            "sleep_end": [],
            "move": [],
            "tweet": [],
            "follow": [],
            "unfollow": [],
            "inner_thought": [],
            "order": [],
            "proposal": [],
            "consumption": [],
            "order_reaction": [],
            "proposal_reaction": [],
            "change_price": [],
        }
        for store in stores:
            counts = self.analyze(store)
            for type in total_counts.keys():
                total_counts[type].append(counts[type])
        return total_counts

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

    def draw_figs_all(
        self, individual_results: list[dict[str, int]]
    ) -> dict[str, Figure]:
        fig: Figure = Figure(figsize=(12, 6))
        ax: Axes = fig.add_subplot(1, 1, 1)
        action_names: list[str] = list(individual_results[0].keys())
        data = [
            [result[action] for result in individual_results] for action in action_names
        ]
        ax.boxplot(data, tick_labels=action_names)
        ax.set_ylabel("Count")
        ax.set_xticks(range(len(action_names)))
        ax.set_xticklabels(
            action_names,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        return {"action_count_all": fig}

    def build_summary(
        self,
        result: dict[str, int],
    ) -> RenderableType:
        """Build a rich summary table for action counts."""
        table = Table(
            title="Action Counts",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Action Type", style="green")
        table.add_column("Count", justify="right", style="magenta")
        total_count: int = 0
        for action_name, count in result.items():
            table.add_row(
                action_name,
                f"{count:,}",
            )
            total_count += count
        return Panel(
            table,
            title=f"{self.name} Summary",
            border_style="blue",
        )
