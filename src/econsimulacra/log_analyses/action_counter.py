from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .store import RecordStore


class ActionCounter(AnalyzerBase[dict[str, int], dict[str, list[int]]]):
    """Count the occurrences of each action type across a simulation log.

    EconSimulacra records twelve distinct action types that agents can
    perform. :class:`ActionCounter` scans a :class:`RecordStore` and
    tallies the number of log entries of each type. The result is a flat
    ``dict[str, int]``, where each key is an action-type string and the
    value is the total count across all agents and time steps.

    **Action types counted**

    * ``sleep_start`` – agent begins a sleep period
    * ``move`` – agent changes its spatial position
    * ``tweet`` – agent posts a message on the social network
    * ``follow`` – agent follows another agent
    * ``unfollow`` – agent unfollows another agent
    * ``inner_thought`` – agent records an internal monologue
    * ``order`` – agent places a market order
    * ``proposal`` – agent proposes a barter trade
    * ``consumption`` – agent consumes an item from its inventory
    * ``order_reaction`` – agent accepts or rejects an incoming order
    * ``proposal_reaction`` – agent accepts or rejects an incoming proposal
    * ``change_price`` – agent updates the listed price of an item

    The bar chart produced by :meth:`draw_figs` provides a quick visual
    overview of the activity mix in a simulation run. Use
    :meth:`draw_figs_all` with multiple stores to compare the distribution
    of action counts across runs via a box plot.
    """

    name = "action_count"

    def analyze(self, store: RecordStore) -> dict[str, int]:
        """Count the occurrences of each action type in *store*.

        Iterates over the twelve predefined action types and calls
        :meth:`~RecordStore.get_by_type` for each. Zero counts are always
        preserved so that the output dict has a fixed set of keys regardless
        of which actions actually occurred in this run.

        Args:
            store (RecordStore): The record store to analyse.

        Returns:
            dict[str, int]: Mapping from action-type string to its total
            occurrence count. Keys are always exactly:
            ``"sleep_start"``, ``"move"``, ``"tweet"``, ``"follow"``,
            ``"unfollow"``, ``"inner_thought"``, ``"order"``,
            ``"proposal"``, ``"consumption"``, ``"order_reaction"``,
            ``"proposal_reaction"``, ``"change_price"``.
        """
        counts: dict[str, int] = {
            "sleep_start": 0,
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
        """Count action occurrences across a collection of stores.

        Calls :meth:`analyze` independently for each store and collects the
        per-action counts into lists, enabling cross-run comparison. The box
        plot produced by :meth:`draw_figs_all` shows the distribution of
        counts across runs.

        Args:
            stores (list[RecordStore]): One record store per simulation run.

        Returns:
            dict[str, list[int]]: Mapping from action-type string to a list
            of counts where ``result[action][i]`` is the count in
            ``stores[i]``. Each inner list has the same length as *stores*.
        """
        total_counts: dict[str, list[int]] = {
            "sleep_start": [],
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
