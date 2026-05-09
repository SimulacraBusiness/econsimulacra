from __future__ import annotations

from datetime import datetime
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import FollowRecord, TweetRecord, UnfollowRecord
from .store import RecordStore


class FollowerCounter(AnalyzerBase[dict[str, dict[datetime | int, int]]]):
    """Follower counter analyzer.

    FollowerCounter counts the number of followers for each agent over time.
    """

    name = "follower_count"

    def analyze(self, store: RecordStore) -> dict[str, dict[datetime | int, int]]:
        """Counts the number of followers for each agent over time.

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            A dictionary mapping agent names to
            a dictionary of timestamps and follower counts.
        """
        agent_id2name: dict[int, str] = self.get_agent_id2name(store)
        follower_counts: dict[str, dict[datetime | int, int]] = {}

        for record in (
            store.typed(FollowRecord)
            + store.typed(UnfollowRecord)
            + store.typed(TweetRecord)
        ):
            agent_id = record.agent_id
            time = record.time
            num_followers = record.num_followers
            agent_name = agent_id2name.get(agent_id, f"Agent {agent_id}")
            if agent_name not in follower_counts:
                follower_counts[agent_name] = {}
            follower_counts[agent_name][time] = num_followers

        return follower_counts

    def draw_figs(
        self,
        result: dict[str, dict[datetime | int, int]],
    ) -> dict[str, Figure]:
        fig: Figure = Figure(figsize=(15, 6))
        ax: Axes = fig.add_subplot(1, 1, 1)
        agent_name2max_followers: dict[str, int] = {
            agent_name: max(time_counts.values())
            for agent_name, time_counts in result.items()
        }
        agent_name2max_followers = {
            agent_name: count
            for agent_name, count in agent_name2max_followers.items()
            if count > 0
        }
        sorted_agent_names: list[str] = sorted(
            agent_name2max_followers.keys(),
            key=lambda x: agent_name2max_followers[x],
            reverse=True,
        )
        sorted_max_followers: list[int] = [
            agent_name2max_followers[agent_name] for agent_name in sorted_agent_names
        ]
        ax.bar(sorted_agent_names, sorted_max_followers)
        ax.set_xticks(range(len(sorted_agent_names)))
        ax.set_xticklabels(
            sorted_agent_names,
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Agent Name")
        ax.set_ylabel("Max Number of Followers")
        return {"follower_count": fig}
    
    def build_summary(
        self,
        result: dict[str, dict[datetime | int, int]],
    ) -> RenderableType:
        """Build a rich summary table for follower counts."""
        agent_name2max_followers: dict[str, int] = {
            agent_name: max(time_counts.values())
            for agent_name, time_counts in result.items()
            if time_counts
        }
        sorted_items: list[tuple[str, int]] = sorted(
            agent_name2max_followers.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        top_table = Table(title="Top 5 Agents by Max Followers")
        top_table.add_column("Rank", justify="right")
        top_table.add_column("Agent Name")
        top_table.add_column("Max Followers", justify="right")
        for rank, (agent_name, count) in enumerate(sorted_items[:5], start=1):
            top_table.add_row(str(rank), agent_name, f"{count:,}")
        counts: list[int] = list(agent_name2max_followers.values())
        quantile_table = Table(title="Follower Count Quantiles")
        quantile_table.add_column("Quantile", justify="right")
        quantile_table.add_column("Max Followers", justify="right")
        if counts:
            sorted_counts: list[int] = sorted(counts)
            n: int = len(sorted_counts)
            for q in [
                0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99
            ]:
                index: int = round((n - 1) * q)
                quantile_table.add_row(
                    f"{int(q * 100)}%",
                    f"{sorted_counts[index]:,}",
                )
        return Panel(
            Group(top_table, quantile_table),
            title=f"{self.name} Summary",
            border_style="blue",
        )
