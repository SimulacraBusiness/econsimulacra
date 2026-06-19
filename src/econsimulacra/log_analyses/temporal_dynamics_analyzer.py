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

    These statistics characterise the temporal *texture* of agent activity.
    All scalar metrics are averaged across agents unless otherwise noted.

    Attributes:
        mean_burstiness (float, optional): Mean across agents of the
            burstiness parameter :math:`B` (Goh & Barabási, 2008):

            .. math::

                B = \\frac{\\sigma_\\tau - \\mu_\\tau}{\\sigma_\\tau + \\mu_\\tau}

            where :math:`\\mu_\\tau` and :math:`\\sigma_\\tau` are the mean
            and standard deviation of inter-event times for the agent.
            :math:`B = 1` for maximally bursty activity, :math:`B = 0`
            for Poisson-random activity, and :math:`B = -1` for perfectly
            regular (clock-like) activity. ``None`` if no agent had at
            least 2 inter-event intervals.

        mean_memory (float, optional): Mean across agents of the memory
            coefficient :math:`M`, the Pearson correlation between
            consecutive inter-event times:

            .. math::

                M = \\text{corr}(\\tau_i,\\, \\tau_{i+1})

            :math:`M > 0` means long pauses cluster together (positive
            memory); :math:`M < 0` means activity alternates between short
            and long gaps. ``None`` if no agent had at least 3 intervals.

        inter_event_times (list[int]): Pooled inter-event times (in
            simulation steps) across all agents.

        activity_by_hour (dict[int, int]): Count of action events per
            clock-hour (0–23). The "hour" is taken from the ``datetime``
            field of the record when available; otherwise
            ``time_step mod period_steps`` is used.

        meals_by_hour (dict[int, int]): Count of
            :class:`~econsimulacra.log_analyses.records.ConsumptionRecord`
            events per clock-hour (0–23).

        sleep_onsets_by_hour (dict[int, int]): Count of
            :class:`~econsimulacra.log_analyses.records.SleepStartRecord`
            events per clock-hour (0–23).

        circadian_autocorr (float, optional): Autocorrelation of the
            per-step action-count series at lag ``period_steps``:

            .. math::

                R(\\ell) = \\frac{\\sum_t (x_t - \\bar{x})
                                         (x_{t+\\ell} - \\bar{x})}
                                {\\sum_t (x_t - \\bar{x})^2}

            where :math:`\\ell` = ``period_steps``. A value close to 1
            indicates a strong daily activity rhythm. ``None`` if the run
            is shorter than two periods.

        n_agents (int): Number of agents with at least one recorded action.
        n_events (int): Total number of action events across all agents.
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
    """Quantify the temporal structure of agent activity in a simulation run.

    This analyzer measures whether agent activity is bursty or regular,
    whether inactive periods cluster together (memory), and whether there is
    a circadian rhythm. These properties mirror the empirical "stylized facts"
    of human activity timing (Barabási, 2005; Goh & Barabási, 2008).

    **Burstiness** :math:`B`
        Measured per agent as the normalised difference between standard
        deviation and mean of that agent's inter-event times:

        .. math::

            B = \\frac{\\sigma_\\tau - \\mu_\\tau}{\\sigma_\\tau + \\mu_\\tau}
            \\in [-1,\\; 1]

        LLM-driven agents often show :math:`B > 0` due to reasoning and
        API latency introducing bursty idle periods.

    **Memory** :math:`M`
        The lag-1 Pearson correlation of inter-event times:

        .. math::

            M = \\text{corr}(\\tau_i,\\, \\tau_{i+1})

        :math:`M > 0` indicates that short active periods follow other
        short active periods; :math:`M < 0` indicates alternating active
        and quiet phases.

    **Circadian autocorrelation**
        The normalised autocorrelation of the per-step event count at lag
        ``period_steps``:

        .. math::

            R(\\ell) = \\frac{\\sum_t (x_t - \\bar{x})(x_{t+\\ell} - \\bar{x})}
                            {\\sum_t (x_t - \\bar{x})^2}

        A value near 1 means the activity pattern repeats every
        ``period_steps`` steps — a sign of a functioning day–night cycle.

    **Hourly histograms**
        Activity, meal, and sleep-onset counts are binned by clock-hour
        (0–23). If records carry a ``datetime`` timestamp, the hour is
        extracted directly; otherwise ``time_step mod period_steps`` is used.

    This analyzer complements :class:`MoveDistanceAnalyzer`: rather than
    asking *how far* agents move, it asks *when* they act and whether that
    timing is rhythmic.

    Attributes:
        name (str): Analyzer name used for organizing outputs.
        action_types (tuple): Record types treated as overt agent actions.
            Defaults to move, consumption, order, tweet, follow, unfollow,
            and sleep-start events.
        period_steps (int): Number of simulation steps per day, used both
            as the circadian autocorrelation lag and the fallback for
            clock-hour binning when records lack datetime timestamps.
        agent_type (str, optional): If set, restrict the analysis to agents
            of this type (e.g. ``"LLMAgent"``); otherwise all agents with
            action records are used.
    """

    name: str = "temporal_dynamics"

    action_types: tuple[type[TimedRecord], ...] = DEFAULT_ACTION_TYPES
    period_steps: int = 24
    agent_type: Optional[str] = None

    def analyze(self, store: RecordStore) -> TemporalDynamicsResult:
        """Compute temporal-dynamics statistics from a record store.

        **Algorithm**

        1. Optionally filter agents by ``agent_type`` using
           :class:`~econsimulacra.log_analyses.records.AgentGenerationRecord`
           entries.
        2. Collect all action events for qualifying agents.
        3. Per agent, sort event steps and compute inter-event intervals
           :math:`\\tau_i = t_{i+1} - t_i`. Compute :math:`B` and :math:`M`
           from the interval sequence.
        4. Build hourly histograms for action events, consumption events,
           and sleep-onset events.
        5. Compute the circadian autocorrelation at lag ``period_steps``.

        Args:
            store (RecordStore): Record store containing the simulation log.

        Returns:
            TemporalDynamicsResult: Aggregated temporal statistics.

        Raises:
            ValueError: If ``period_steps`` is not positive, or if no action
                records are found.
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
        """Compute burstiness :math:`B` and memory coefficient :math:`M`.

        Given the sequence of inter-event intervals
        :math:`(\\tau_1, \\tau_2, \\ldots, \\tau_n)`:

        .. math::

            B = \\frac{\\sigma_\\tau - \\mu_\\tau}{\\sigma_\\tau + \\mu_\\tau},
            \\qquad
            M = \\text{corr}(\\tau_i,\\, \\tau_{i+1})

        :math:`B` requires :math:`n \\geq 2`. :math:`M` requires
        :math:`n \\geq 3` and non-zero variance in both the earlier and
        later sub-sequences.

        Args:
            intervals (list[int]): Sequence of inter-event times in steps.

        Returns:
            tuple[Optional[float], Optional[float]]: ``(B, M)``. Either
            value is ``None`` when the corresponding requirement is not met.
        """
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
