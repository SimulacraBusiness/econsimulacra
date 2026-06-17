from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import (
    AgentGenerationRecord,
    ConsumptionRecord,
    FollowRecord,
    MoveRecord,
    OrderRecord,
    SleepStartRecord,
    TimedRecord,
    TweetRecord,
    UnfollowRecord,
)
from .store import RecordStore

# action record types that count as overt agent activity for inter-event timing
DEFAULT_ACTION_TYPES: tuple[type[TimedRecord], ...] = (
    MoveRecord,
    ConsumptionRecord,
    OrderRecord,
    TweetRecord,
    FollowRecord,
    UnfollowRecord,
    SleepStartRecord,
)


@dataclass(frozen=True)
class TemporalDynamicsResult:
    """Temporal-dynamics statistics for one simulation run.

    Attributes:
        mean_burstiness: Mean across agents of the burstiness parameter
            ``B = (sigma - mu) / (sigma + mu)`` of inter-event times. ``B`` is
            ``1`` for maximally bursty activity, ``0`` for Poisson, and ``-1``
            for perfectly regular activity (Goh & Barabási, 2008).
        mean_memory: Mean across agents of the memory coefficient ``M``, the
            lag-1 correlation between consecutive inter-event times.
        inter_event_times: Pooled inter-event times (in steps) across agents.
        activity_by_hour: Count of action events per clock-hour (0-23).
        meals_by_hour: Count of consumption events per clock-hour (0-23).
        sleep_onsets_by_hour: Count of sleep-start events per clock-hour (0-23).
        circadian_autocorr: Autocorrelation of per-step activity at a lag of
            ``period_steps``. ``None`` if the run is shorter than one period.
        n_agents: Number of agents with at least one action event.
        n_events: Total number of action events.
    """

    mean_burstiness: Optional[float]
    mean_memory: Optional[float]
    inter_event_times: list[int]
    activity_by_hour: dict[int, int]
    meals_by_hour: dict[int, int]
    sleep_onsets_by_hour: dict[int, int]
    circadian_autocorr: Optional[float]
    n_agents: int
    n_events: int


@dataclass
class TemporalDynamicsAnalyzer(AnalyzerBase[TemporalDynamicsResult, None]):
    """Analyze the temporal structure of agent activity.

    This analyzer quantifies the human "stylized facts" of activity timing from
    a simulation log: burstiness and memory of inter-event times, the
    distribution of activity / meals / sleep onsets across the 24-hour clock,
    and the autocorrelation of activity at a daily lag (a circadian signature).

    It complements the spatial view of :class:`MoveDistanceAnalyzer`: rather than
    asking how far agents move, it asks *when* they act, and whether that timing
    is bursty and rhythmic rather than uniform.

    Attributes:
        name: Analyzer name used for organizing outputs.
        action_types: Record types treated as overt agent actions.
        period_steps: Number of steps per day, used for the circadian-lag
            autocorrelation.
        agent_type: If set, restrict the analysis to agents of this type (e.g.
            ``"LLMAgent"``); otherwise all agents with action records are used.
    """

    name: str = "temporal_dynamics"

    action_types: tuple[type[TimedRecord], ...] = DEFAULT_ACTION_TYPES
    period_steps: int = 24
    agent_type: Optional[str] = None

    def analyze(self, store: RecordStore) -> TemporalDynamicsResult:
        """Compute temporal-dynamics statistics from a record store.

        Args:
            store: Record store containing the run's records.

        Returns:
            The aggregated :class:`TemporalDynamicsResult`.

        Raises:
            ValueError: If ``period_steps`` is not positive or no action records
                are found.
        """
        if self.period_steps <= 0:
            raise ValueError("period_steps must be positive.")

        allowed_agents: Optional[set[int]] = self._select_agents(store)

        actions: list[tuple[int, TimedRecord]] = []
        for action_type in self.action_types:
            for record in store.typed(action_type):
                agent_id: Optional[int] = getattr(record, "agent_id", None)
                if agent_id is None:
                    continue
                if allowed_agents is None or agent_id in allowed_agents:
                    actions.append((int(agent_id), record))
        if not actions:
            raise ValueError("No action records found.")

        steps_by_agent: defaultdict[int, set[int]] = defaultdict(set)
        for agent_id, record in actions:
            steps_by_agent[agent_id].add(int(record.time_step))

        inter_event_times: list[int] = []
        burstiness_values: list[float] = []
        memory_values: list[float] = []
        for steps in steps_by_agent.values():
            intervals: list[int] = [
                later - earlier
                for earlier, later in zip(sorted(steps), sorted(steps)[1:])
            ]
            inter_event_times.extend(intervals)
            burstiness, memory = self._burstiness_and_memory(intervals)
            if burstiness is not None:
                burstiness_values.append(burstiness)
            if memory is not None:
                memory_values.append(memory)

        activity_by_hour: Counter[int] = Counter()
        for _, record in actions:
            hour: Optional[int] = self._clock_hour(record)
            if hour is not None:
                activity_by_hour[hour] += 1

        meals_by_hour: Counter[int] = self._hour_histogram(
            store.typed(ConsumptionRecord), allowed_agents
        )
        sleep_onsets_by_hour: Counter[int] = self._hour_histogram(
            store.typed(SleepStartRecord), allowed_agents
        )

        return TemporalDynamicsResult(
            mean_burstiness=(
                float(np.mean(burstiness_values)) if burstiness_values else None
            ),
            mean_memory=float(np.mean(memory_values)) if memory_values else None,
            inter_event_times=inter_event_times,
            activity_by_hour=dict(sorted(activity_by_hour.items())),
            meals_by_hour=dict(sorted(meals_by_hour.items())),
            sleep_onsets_by_hour=dict(sorted(sleep_onsets_by_hour.items())),
            circadian_autocorr=self._circadian_autocorr(
                [record for _, record in actions]
            ),
            n_agents=len(steps_by_agent),
            n_events=len(actions),
        )

    def analyze_stores(self, stores: list[RecordStore]) -> None:
        return None

    def draw_figs(self, result: TemporalDynamicsResult) -> dict[str, Figure]:
        """Draw temporal-dynamics figures.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Dictionary mapping figure names to Matplotlib figures.
        """
        figures: dict[str, Figure] = {}

        hours: list[int] = list(range(24))
        activity: list[int] = [result.activity_by_hour.get(hour, 0) for hour in hours]
        fig_activity: Figure
        ax_activity: Axes
        fig_activity, ax_activity = plt.subplots(figsize=(8, 6))
        ax_activity.bar(hours, activity)
        ax_activity.set_title("Activity by clock-hour")
        ax_activity.set_xlabel("hour of day")
        ax_activity.set_ylabel("action count")
        ax_activity.set_xticks(range(0, 24, 2))
        fig_activity.tight_layout()
        figures["activity_by_hour"] = fig_activity

        meals: list[int] = [result.meals_by_hour.get(hour, 0) for hour in hours]
        fig_meals: Figure
        ax_meals: Axes
        fig_meals, ax_meals = plt.subplots(figsize=(8, 6))
        ax_meals.bar(hours, meals)
        ax_meals.set_title("Meals by clock-hour")
        ax_meals.set_xlabel("hour of day")
        ax_meals.set_ylabel("consumption count")
        ax_meals.set_xticks(range(0, 24, 2))
        fig_meals.tight_layout()
        figures["meals_by_hour"] = fig_meals

        if result.inter_event_times:
            fig_iet: Figure
            ax_iet: Axes
            fig_iet, ax_iet = plt.subplots(figsize=(8, 6))
            max_interval: int = max(result.inter_event_times)
            ax_iet.hist(
                result.inter_event_times,
                bins=range(1, max_interval + 2),
                align="left",
            )
            ax_iet.set_title("Inter-event time distribution")
            ax_iet.set_xlabel("steps between actions")
            ax_iet.set_ylabel("frequency")
            fig_iet.tight_layout()
            figures["inter_event_times"] = fig_iet

        return figures

    def draw_figs_all(
        self, individual_results: list[TemporalDynamicsResult]
    ) -> dict[str, Figure]:
        return {}

    def build_summary(self, result: TemporalDynamicsResult) -> Panel:
        """Build a Rich summary panel for temporal-dynamics analysis.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Rich panel summarizing the temporal statistics.
        """
        table: Table = Table(title="Temporal Dynamics Summary")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Agents", str(result.n_agents))
        table.add_row("Action events", str(result.n_events))
        table.add_row(
            "Mean burstiness B",
            "-" if result.mean_burstiness is None else f"{result.mean_burstiness:.3f}",
        )
        table.add_row(
            "Mean memory M",
            "-" if result.mean_memory is None else f"{result.mean_memory:.3f}",
        )
        table.add_row(
            f"Circadian autocorr (lag {self.period_steps})",
            "-"
            if result.circadian_autocorr is None
            else f"{result.circadian_autocorr:.3f}",
        )
        peak_hour: Optional[int] = (
            max(result.activity_by_hour, key=lambda hour: result.activity_by_hour[hour])
            if result.activity_by_hour
            else None
        )
        table.add_row(
            "Peak activity hour",
            "-" if peak_hour is None else f"{peak_hour:02d}:00",
        )

        return Panel.fit(table, title="Analysis Summary", border_style="cyan")

    def _select_agents(self, store: RecordStore) -> Optional[set[int]]:
        """Return the agent ids to include, or ``None`` to include all."""
        if self.agent_type is None:
            return None
        return {
            record.agent_id
            for record in store.typed(AgentGenerationRecord)
            if record.agent_type == self.agent_type
        }

    def _burstiness_and_memory(
        self, intervals: list[int]
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute burstiness B and memory coefficient M for one agent."""
        if len(intervals) < 2:
            return None, None
        values: np.ndarray = np.asarray(intervals, dtype=float)
        mean: float = float(values.mean())
        std: float = float(values.std())
        denom: float = std + mean
        burstiness: float = (std - mean) / denom if denom > 0 else 0.0
        memory: Optional[float] = None
        if len(values) >= 3:
            earlier, later = values[:-1], values[1:]
            if earlier.std() > 0 and later.std() > 0:
                memory = float(np.corrcoef(earlier, later)[0, 1])
        return burstiness, memory

    def _circadian_autocorr(self, actions: list[TimedRecord]) -> Optional[float]:
        """Autocorrelation of per-step activity at a lag of ``period_steps``."""
        max_step: int = max(int(record.time_step) for record in actions)
        if max_step < 2 * self.period_steps:
            return None
        counts: np.ndarray = np.zeros(max_step + 1)
        for record in actions:
            counts[int(record.time_step)] += 1.0
        centered: np.ndarray = counts - counts.mean()
        denom: float = float(np.sum(centered * centered))
        if denom == 0.0:
            return None
        lag: int = self.period_steps
        return float(np.sum(centered[:-lag] * centered[lag:]) / denom)

    def _hour_histogram(
        self, records: list[TimedRecord], allowed_agents: Optional[set[int]]
    ) -> Counter[int]:
        """Count records per clock-hour, optionally filtered by agent."""
        histogram: Counter[int] = Counter()
        for record in records:
            if allowed_agents is not None:
                agent_id: Optional[int] = getattr(record, "agent_id", None)
                if agent_id is None or agent_id not in allowed_agents:
                    continue
            hour: Optional[int] = self._clock_hour(record)
            if hour is not None:
                histogram[hour] += 1
        return histogram

    def _clock_hour(self, record: TimedRecord) -> Optional[int]:
        """Clock-hour of a record, from datetime or step-within-period."""
        if isinstance(record.time, datetime):
            return record.time.hour
        if self.period_steps > 0:
            return int(record.time_step) % self.period_steps
        return None
