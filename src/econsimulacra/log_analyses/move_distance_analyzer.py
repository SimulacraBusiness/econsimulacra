from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import Literal, Optional, TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import MoveRecord
from .store import RecordStore

DistanceMetric: TypeAlias = Literal["euclidean", "manhattan", "chebyshev"]


@dataclass(frozen=True)
class MoveDistanceWindowStats:
    """Aggregated movement statistics for a single time window.

    Attributes:
        total_distance: Sum of movement distances in the window.
        mean_distance: Average distance per move in the window. This value is
            zero when ``move_count`` is zero.
        move_count: Number of movement records in the window.
        moving_agent_count: Number of distinct agents that moved in the window.
    """

    total_distance: float
    mean_distance: float
    move_count: int
    moving_agent_count: int


MoveDistanceResult: TypeAlias = dict[int, MoveDistanceWindowStats]


@dataclass
class MoveDistanceAnalyzer(AnalyzerBase[MoveDistanceResult, None]):
    """Analyze aggregate movement distance over time windows.

    This analyzer aggregates :class:`MoveRecord` instances into fixed-size
    windows based on ``time_step``. For each window, it computes the total
    movement distance, average movement distance, number of move events, and
    number of distinct moving agents.

    The primary purpose is to diagnose how much spatial movement occurs over
    simulation time. This is useful for evaluating whether spatial constraints,
    local demand, congestion, or location-dependent incentives affect agent
    behavior.

    Distances are computed from ``old_pos`` and ``new_pos`` using one of three
    metrics:

    - ``"euclidean"``: square-root of squared coordinate differences.
    - ``"manhattan"``: sum of absolute coordinate differences.
    - ``"chebyshev"``: maximum absolute coordinate difference.

    Attributes:
        name: Analyzer name used for organizing outputs.
        window_size: Number of simulation steps in one aggregation window.
        total_time_steps: Total number of simulation steps. If ``None``, this is
            inferred from the maximum ``time_step`` in movement records.
        distance_metric: Distance metric used to compute movement length.
        include_zero_distance: Whether to include records where
            ``old_pos == new_pos``.
        top_k_agents: Number of high-mobility agents shown in the summary.
    """

    name: str = "move_distance"

    window_size: int = 10
    total_time_steps: Optional[int] = None
    distance_metric: DistanceMetric = "euclidean"
    include_zero_distance: bool = False
    top_k_agents: int = 10

    agent_distance_: Optional[dict[int, float]] = None
    agent_move_count_: Optional[dict[int, int]] = None

    def analyze(self, store: RecordStore) -> MoveDistanceResult:
        """Aggregate movement distances over fixed-size time windows.

        Args:
            store: Record store containing :class:`MoveRecord` instances.

        Returns:
            Mapping from window start time to movement statistics.

        Raises:
            ValueError: If no valid movement records are available, if
                ``window_size`` is not positive, or if position dimensionalities
                are inconsistent.
        """
        self._prepare_time_axis(store)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        move_records: list[MoveRecord] = list(store.typed(MoveRecord))
        if not move_records:
            raise ValueError("No MoveRecord entries found.")
        filtered_records: list[MoveRecord] = []
        for record in move_records:
            if not self.include_zero_distance and record.old_pos == record.new_pos:
                continue
            filtered_records.append(record)
        if not filtered_records:
            raise ValueError(
                "No valid MoveRecord entries remain after filtering zero-distance moves."
            )
        if self.total_time_steps is None:
            max_time_step: int = max(
                int(record.time_step) for record in filtered_records
            )
            total_time_steps: int = max_time_step + 1
        else:
            total_time_steps = self.total_time_steps
        window_starts: list[int] = list(range(0, total_time_steps, self.window_size))
        window_distance_sum: defaultdict[int, float] = defaultdict(float)
        window_move_count: defaultdict[int, int] = defaultdict(int)
        window_agents: defaultdict[int, set[int]] = defaultdict(set)
        agent_distance: defaultdict[int, float] = defaultdict(float)
        agent_move_count: defaultdict[int, int] = defaultdict(int)
        for record in filtered_records:
            record_time_step: int = int(record.time_step)
            window_start: int = (
                record_time_step // self.window_size
            ) * self.window_size
            distance: float = self._distance(record.old_pos, record.new_pos)
            window_distance_sum[window_start] += distance
            window_move_count[window_start] += 1
            window_agents[window_start].add(int(record.agent_id))
            agent_distance[int(record.agent_id)] += distance
            agent_move_count[int(record.agent_id)] += 1
        self.agent_distance_ = dict(agent_distance)
        self.agent_move_count_ = dict(agent_move_count)

        result: MoveDistanceResult = {}

        for window_start in window_starts:
            total_distance: float = float(window_distance_sum[window_start])
            move_count: int = int(window_move_count[window_start])
            mean_distance: float = (
                total_distance / float(move_count) if move_count > 0 else 0.0
            )
            moving_agent_count: int = len(window_agents[window_start])
            result[window_start] = MoveDistanceWindowStats(
                total_distance=total_distance,
                mean_distance=mean_distance,
                move_count=move_count,
                moving_agent_count=moving_agent_count,
            )

        return result

    def analyze_stores(self, stores: list[RecordStore]) -> None:
        return None

    def draw_figs(self, result: MoveDistanceResult) -> dict[str, Figure]:
        """Draw movement-distance diagnostic figures.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Dictionary mapping figure names to Matplotlib figures.
        """
        ordered_items: list[tuple[int, MoveDistanceWindowStats]] = sorted(
            result.items(),
            key=lambda item: item[0],
        )
        times: list[int] = [key for key, _ in ordered_items]
        total_distances: list[float] = [
            stats.total_distance for _, stats in ordered_items
        ]
        mean_distances: list[float] = [
            stats.mean_distance for _, stats in ordered_items
        ]
        last_time: int = times[-1] if times else 0
        move_counts: list[int] = [stats.move_count for _, stats in ordered_items]
        moving_agent_counts: list[int] = [
            stats.moving_agent_count for _, stats in ordered_items
        ]
        if self._time_axis_config is not None:
            max_time = int(self._time_axis_config.x_max)
            if last_time < max_time:
                times.append(max_time)
                total_distances.append(0.0)
                mean_distances.append(0.0)
                move_counts.append(0)
                moving_agent_counts.append(0)
        figures: dict[str, Figure] = {}
        fig_total: Figure
        ax_total: Axes
        fig_total, ax_total = plt.subplots(figsize=(8, 6))
        ax_total.plot(np.array(times), total_distances, marker="o")
        ax_total.set_title("Total movement distance by time window")
        ax_total.set_xlabel("window start")
        ax_total.set_ylabel("total distance")
        ax_total.tick_params(axis="x", rotation=45)
        self._apply_time_axis(ax_total)
        fig_total.tight_layout()
        figures["window_total_distance"] = fig_total

        fig_mean: Figure
        ax_mean: Axes
        fig_mean, ax_mean = plt.subplots(figsize=(8, 6))
        ax_mean.plot(np.array(times), mean_distances, marker="o")
        ax_mean.set_title("Mean movement distance per move by time window")
        ax_mean.set_xlabel("window start")
        ax_mean.set_ylabel("mean distance")
        ax_mean.tick_params(axis="x", rotation=45)
        self._apply_time_axis(ax_mean)
        fig_mean.tight_layout()
        figures["window_mean_distance"] = fig_mean

        fig_count: Figure
        ax_count: Axes
        fig_count, ax_count = plt.subplots(figsize=(8, 6))
        ax_count.plot(np.array(times), move_counts, marker="o", label="move count")
        ax_count.plot(
            np.array(times),
            moving_agent_counts,
            marker="o",
            label="moving agent count",
        )
        ax_count.set_title("Movement frequency by time window")
        ax_count.set_xlabel("window start")
        ax_count.set_ylabel("count")
        ax_count.legend()
        ax_count.tick_params(axis="x", rotation=45)
        self._apply_time_axis(ax_count)
        fig_count.tight_layout()
        figures["window_movement_counts"] = fig_count

        if self.agent_distance_:
            top_agents: list[tuple[int, float]] = sorted(
                self.agent_distance_.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self.top_k_agents]

            agent_ids: list[str] = [str(agent_id) for agent_id, _ in top_agents]
            agent_distances: list[float] = [distance for _, distance in top_agents]

            fig_agent: Figure
            ax_agent: Axes
            fig_agent, ax_agent = plt.subplots(figsize=(8, 6))
            ax_agent.bar(agent_ids, agent_distances)
            ax_agent.set_title(f"Top {len(top_agents)} agents by movement distance")
            ax_agent.set_xlabel("agent_id")
            ax_agent.set_ylabel("total distance")
            ax_agent.tick_params(axis="x", rotation=45)
            fig_agent.tight_layout()
            figures["top_agent_total_distance"] = fig_agent

        return figures

    def draw_figs_all(
        self, individual_results: list[MoveDistanceResult]
    ) -> dict[str, Figure]:
        return {}

    def build_summary(self, result: MoveDistanceResult) -> Panel:
        """Build a Rich summary panel for movement-distance analysis.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Rich panel summarizing aggregate movement statistics.
        """
        if not result:
            return Panel.fit(
                "No movement-distance result is available.",
                title="Move Distance Summary",
                border_style="yellow",
            )

        total_distance_all: float = sum(
            stats.total_distance for stats in result.values()
        )
        total_move_count: int = sum(stats.move_count for stats in result.values())
        mean_distance_all: float = (
            total_distance_all / float(total_move_count)
            if total_move_count > 0
            else 0.0
        )

        max_window: tuple[int, MoveDistanceWindowStats] = max(
            result.items(),
            key=lambda item: item[1].total_distance,
        )
        min_window: tuple[int, MoveDistanceWindowStats] = min(
            result.items(),
            key=lambda item: item[1].total_distance,
        )

        table: Table = Table(title="Move Distance Summary")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Distance metric", self.distance_metric)
        table.add_row("Window size", str(self.window_size))
        table.add_row("Total distance", f"{total_distance_all:.3f}")
        table.add_row("Total move count", str(total_move_count))
        table.add_row("Mean distance per move", f"{mean_distance_all:.3f}")
        table.add_row(
            "Max-distance window",
            f"{self._format_time_key(max_window[0])} "
            f"({max_window[1].total_distance:.3f})",
        )
        table.add_row(
            "Min-distance window",
            f"{self._format_time_key(min_window[0])} "
            f"({min_window[1].total_distance:.3f})",
        )

        if self.agent_distance_ is not None:
            top_agents: list[tuple[int, float]] = sorted(
                self.agent_distance_.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: self.top_k_agents]

            ranking_text: str = ", ".join(
                f"{agent_id}: {distance:.3f}" for agent_id, distance in top_agents
            )
            table.add_row(
                f"Top {len(top_agents)} agents by distance",
                ranking_text or "-",
            )

        return Panel.fit(table, title="Analysis Summary", border_style="cyan")

    def _distance(self, old_pos: tuple[int, ...], new_pos: tuple[int, ...]) -> float:
        """Compute movement distance between two positions.

        Args:
            old_pos: Source position.
            new_pos: Destination position.

        Returns:
            Distance between ``old_pos`` and ``new_pos``.

        Raises:
            ValueError: If the two positions have different dimensionalities or
                if ``distance_metric`` is unsupported.
        """
        if len(old_pos) != len(new_pos):
            raise ValueError(
                f"Position dimensionality mismatch: old_pos={old_pos}, "
                f"new_pos={new_pos}."
            )

        diffs: list[int] = [
            abs(int(new_coord) - int(old_coord))
            for old_coord, new_coord in zip(old_pos, new_pos, strict=True)
        ]

        if self.distance_metric == "euclidean":
            squared_sum: int = sum(diff * diff for diff in diffs)
            return float(sqrt(float(squared_sum)))

        if self.distance_metric == "manhattan":
            return float(sum(diffs))

        if self.distance_metric == "chebyshev":
            return float(max(diffs, default=0))

        raise ValueError(f"Unsupported distance_metric: {self.distance_metric}")
