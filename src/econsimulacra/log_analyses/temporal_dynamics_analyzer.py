from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
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
class ActionTypeStats:
    """Temporal statistics for a single action type.

    Attributes:
        mean_burstiness (float, optional): Mean across agents of the
            burstiness parameter :math:`B` (Goh & Barabási, 2008):

            .. math::

                B = \\frac{\\sigma_\\tau - \\mu_\\tau}{\\sigma_\\tau + \\mu_\\tau}

            ``None`` if no agent had at least 2 inter-event intervals.

        mean_memory (float, optional): Mean across agents of the memory
            coefficient :math:`M`, the Pearson correlation between
            consecutive inter-event times. ``None`` if no agent had at
            least 3 intervals.

        inter_event_times (list[int]): Pooled inter-event times (in
            simulation steps) across all agents for this action type.

        activity_by_hour (dict[int, int]): Count of events per clock-hour
            (0–23) for this action type.

        n_events (int): Total number of events of this action type.
    """

    mean_burstiness: Optional[float]
    mean_memory: Optional[float]
    inter_event_times: list[int]
    activity_by_hour: dict[int, int]
    n_events: int


@dataclass(frozen=True)
class ActionTypeAggregateStats:
    """Aggregated temporal statistics for one action type across multiple runs.

    Attributes:
        mean_burstiness_mean (float, optional): Mean of per-run burstiness values.
        mean_burstiness_std (float, optional): Std of per-run burstiness values.
            ``None`` when fewer than 2 runs contributed a value.
        mean_memory_mean (float, optional): Mean of per-run memory coefficients.
        mean_memory_std (float, optional): Std of per-run memory coefficients.
            ``None`` when fewer than 2 runs contributed a value.
        mean_iet_mean (float, optional): Mean of per-run mean inter-event times.
        mean_iet_std (float, optional): Std of per-run mean inter-event times.
            ``None`` when fewer than 2 runs contributed a value.
        mean_n_events (float): Mean event count across runs.
        std_n_events (float, optional): Std of event counts across runs.
            ``None`` when fewer than 2 runs contributed.
        pooled_inter_event_times (list[int]): All inter-event times pooled
            across every run for this action type.
        mean_activity_by_hour (dict[int, float]): Mean activity count per
            clock-hour averaged across runs.
    """

    mean_burstiness_mean: Optional[float]
    mean_burstiness_std: Optional[float]
    mean_memory_mean: Optional[float]
    mean_memory_std: Optional[float]
    mean_iet_mean: Optional[float]
    mean_iet_std: Optional[float]
    mean_n_events: float
    std_n_events: Optional[float]
    pooled_inter_event_times: list[int]
    mean_activity_by_hour: dict[int, float]


@dataclass(frozen=True)
class TemporalDynamicsAggregateResult:
    """Aggregated temporal-dynamics statistics across multiple simulation runs.

    Attributes:
        by_action_type (dict[str, ActionTypeAggregateStats]): Per-action-type
            aggregate statistics.
        circadian_autocorr_mean (float, optional): Mean circadian autocorrelation
            across runs.
        circadian_autocorr_std (float, optional): Std of circadian autocorrelation.
            ``None`` when fewer than 2 runs produced a value.
        n_runs (int): Number of simulation runs included in the aggregate.
    """

    by_action_type: dict[str, ActionTypeAggregateStats]
    circadian_autocorr_mean: Optional[float]
    circadian_autocorr_std: Optional[float]
    n_runs: int


@dataclass(frozen=True)
class TemporalDynamicsResult:
    """Temporal-dynamics statistics for one simulation run.

    Statistics are computed **per action type** so that, for example,
    the burstiness of movement can be compared against the burstiness of
    consumption independently.

    Attributes:
        by_action_type (dict[str, ActionTypeStats]): Mapping from action-type
            name (e.g. ``"Move"``, ``"Consumption"``) to its temporal
            statistics.  Only action types that produced at least one event
            are included.

        circadian_autocorr (float, optional): Autocorrelation of the
            per-step action-count series (pooled over all action types) at
            lag ``period_steps``:

            .. math::

                R(\\ell) = \\frac{\\sum_t (x_t - \\bar{x})
                                         (x_{t+\\ell} - \\bar{x})}
                                {\\sum_t (x_t - \\bar{x})^2}

            ``None`` if the run is shorter than two periods.

        n_agents (int): Number of agents with at least one recorded action.
        n_events (int): Total number of action events across all agents and
            all action types.
    """

    by_action_type: dict[str, ActionTypeStats]
    circadian_autocorr: Optional[float]
    n_agents: int
    n_events: int


@dataclass
class TemporalDynamicsAnalyzer(
    AnalyzerBase[TemporalDynamicsResult, TemporalDynamicsAggregateResult]
):
    """Quantify the temporal structure of agent activity in a simulation run.

    This analyzer measures whether agent activity is bursty or regular,
    whether inactive periods cluster together (memory), and whether there is
    a circadian rhythm. Statistics are computed **per action type**: burstiness
    of movement, burstiness of consumption, etc. are reported independently.

    **Burstiness** :math:`B`
        Measured per agent as the normalised difference between standard
        deviation and mean of that agent's inter-event times for the given
        action type:

        .. math::

            B = \\frac{\\sigma_\\tau - \\mu_\\tau}{\\sigma_\\tau + \\mu_\\tau}
            \\in [-1,\\; 1]

        :math:`B > 0` indicates bursty activity with many short intervals and
        a few long intervals; :math:`B < 0` indicates regular activity with
        similar-length intervals.

    **Memory** :math:`M`
        The lag-1 Pearson correlation of inter-event times:

        .. math::

            M = \\text{corr}(\\tau_i,\\, \\tau_{i+1})

        :math:`M > 0` indicates that short active periods follow other
        short active periods; :math:`M < 0` indicates alternating active
        and quiet phases.

    **Circadian autocorrelation**
        The normalised autocorrelation of the per-step event count (pooled
        across all action types) at lag ``period_steps``:

        .. math::

            R(\\ell) = \\frac{\\sum_t (x_t - \\bar{x})(x_{t+\\ell} - \\bar{x})}
                            {\\sum_t (x_t - \\bar{x})^2}

        A value near 1 means the activity pattern repeats every
        ``period_steps`` steps — a sign of a functioning day–night cycle.

    **Hourly histograms**
        Activity counts are binned by clock-hour (0–23) for each action type
        separately. If records carry a ``datetime`` timestamp, the hour is
        extracted directly; otherwise ``time_step mod period_steps`` is used.

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
        exclude_agent_ids (list[int]): Agent IDs to exclude from the analysis.
    """

    name: str = "temporal_dynamics"

    action_types: tuple[type[TimedRecord], ...] = DEFAULT_ACTION_TYPES
    period_steps: int = 24
    agent_type: Optional[str] = None
    exclude_agent_ids: list[int] = field(default_factory=list)

    def analyze(self, store: RecordStore) -> TemporalDynamicsResult:
        """Compute temporal-dynamics statistics from a record store.

        **Algorithm**

        1. Optionally filter agents by ``agent_type`` using
           :class:`~econsimulacra.log_analyses.records.AgentGenerationRecord`
           entries.
        2. For each action type in ``action_types``, collect events for
           qualifying agents.
        3. Per action type and per agent, sort event steps and compute
           inter-event intervals :math:`\\tau_i = t_{i+1} - t_i`. Compute
           :math:`B` and :math:`M` from each agent's interval sequence, then
           average across agents to obtain the per-action-type statistics.
        4. Build hourly histograms for each action type.
        5. Compute the circadian autocorrelation at lag ``period_steps``
           using the pooled event stream across all action types.

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

        all_action_records: list[TimedRecord] = []
        all_agent_ids: set[int] = set()
        total_events: int = 0

        by_action_type: dict[str, ActionTypeStats] = {}

        for action_type in self.action_types:
            type_name: str = action_type.__name__.removesuffix("Record")
            type_records: list[tuple[int, TimedRecord]] = []

            for record in store.typed(action_type):
                agent_id: Optional[int] = getattr(record, "agent_id", None)
                if agent_id is None:
                    continue
                if agent_id in self.exclude_agent_ids:
                    continue
                if allowed_agents is None or agent_id in allowed_agents:
                    type_records.append((int(agent_id), record))
                    all_action_records.append(record)
                    all_agent_ids.add(int(agent_id))

            if not type_records:
                continue

            total_events += len(type_records)

            steps_by_agent: defaultdict[int, list[int]] = defaultdict(list)
            for agent_id, record in type_records:
                steps_by_agent[agent_id].append(int(record.time_step))

            inter_event_times: list[int] = []
            burstiness_values: list[float] = []
            memory_values: list[float] = []
            for steps in steps_by_agent.values():
                sorted_steps: list[int] = sorted(steps)
                intervals: list[int] = [
                    later - earlier
                    for earlier, later in zip(sorted_steps, sorted_steps[1:])
                ]
                inter_event_times.extend(intervals)
                burstiness, memory = self._burstiness_and_memory(intervals)
                if burstiness is not None:
                    burstiness_values.append(burstiness)
                if memory is not None:
                    memory_values.append(memory)

            activity_by_hour: Counter[int] = Counter()
            for _, record in type_records:
                hour: Optional[int] = self._clock_hour(record)
                if hour is not None:
                    activity_by_hour[hour] += 1

            by_action_type[type_name] = ActionTypeStats(
                mean_burstiness=(
                    float(np.mean(burstiness_values)) if burstiness_values else None
                ),
                mean_memory=float(np.mean(memory_values)) if memory_values else None,
                inter_event_times=inter_event_times,
                activity_by_hour=dict(sorted(activity_by_hour.items())),
                n_events=len(type_records),
            )

        if not all_action_records:
            raise ValueError("No action records found.")

        return TemporalDynamicsResult(
            by_action_type=by_action_type,
            circadian_autocorr=self._circadian_autocorr(all_action_records),
            n_agents=len(all_agent_ids),
            n_events=total_events,
        )

    def analyze_stores(
        self, stores: list[RecordStore]
    ) -> TemporalDynamicsAggregateResult:
        """Aggregate temporal-dynamics statistics across multiple stores.

        Runs :meth:`analyze` on each store and aggregates per-action-type
        statistics (mean and standard deviation of burstiness, memory,
        mean inter-event time, and event count) across runs.

        Args:
            stores (list[RecordStore]): One store per simulation run.

        Returns:
            TemporalDynamicsAggregateResult: Aggregated statistics.
        """
        individual: list[TemporalDynamicsResult] = [
            self.analyze(store) for store in stores
        ]

        all_action_names: set[str] = set()
        for result in individual:
            all_action_names.update(result.by_action_type.keys())

        by_action_type: dict[str, ActionTypeAggregateStats] = {}
        for action_name in sorted(all_action_names):
            burstiness_vals: list[float] = []
            memory_vals: list[float] = []
            mean_iet_vals: list[float] = []
            n_events_vals: list[float] = []
            pooled_iets: list[int] = []
            hour_accumulator: defaultdict[int, list[int]] = defaultdict(list)

            for result in individual:
                stats: Optional[ActionTypeStats] = result.by_action_type.get(
                    action_name
                )
                if stats is None:
                    continue
                if stats.mean_burstiness is not None:
                    burstiness_vals.append(stats.mean_burstiness)
                if stats.mean_memory is not None:
                    memory_vals.append(stats.mean_memory)
                if stats.inter_event_times:
                    mean_iet_vals.append(float(np.mean(stats.inter_event_times)))
                    pooled_iets.extend(stats.inter_event_times)
                n_events_vals.append(float(stats.n_events))
                for hour, count in stats.activity_by_hour.items():
                    hour_accumulator[hour].append(count)

            mean_activity_by_hour: dict[int, float] = {
                hour: float(np.mean(counts))
                for hour, counts in sorted(hour_accumulator.items())
            }

            def _mean(vals: list[float]) -> Optional[float]:
                return float(np.mean(vals)) if vals else None

            def _std(vals: list[float]) -> Optional[float]:
                return float(np.std(vals, ddof=1)) if len(vals) >= 2 else None

            by_action_type[action_name] = ActionTypeAggregateStats(
                mean_burstiness_mean=_mean(burstiness_vals),
                mean_burstiness_std=_std(burstiness_vals),
                mean_memory_mean=_mean(memory_vals),
                mean_memory_std=_std(memory_vals),
                mean_iet_mean=_mean(mean_iet_vals),
                mean_iet_std=_std(mean_iet_vals),
                mean_n_events=float(np.mean(n_events_vals)) if n_events_vals else 0.0,
                std_n_events=_std(n_events_vals),
                pooled_inter_event_times=pooled_iets,
                mean_activity_by_hour=mean_activity_by_hour,
            )

        autocorr_vals: list[float] = [
            r.circadian_autocorr for r in individual if r.circadian_autocorr is not None
        ]
        return TemporalDynamicsAggregateResult(
            by_action_type=by_action_type,
            circadian_autocorr_mean=float(np.mean(autocorr_vals))
            if autocorr_vals
            else None,
            circadian_autocorr_std=float(np.std(autocorr_vals, ddof=1))
            if len(autocorr_vals) >= 2
            else None,
            n_runs=len(individual),
        )

    def draw_figs(self, result: TemporalDynamicsResult) -> dict[str, Figure]:
        """Draw temporal-dynamics figures.

        Produces one inter-event time line plot and one activity-by-hour bar
        chart for each action type that has recorded events.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Dictionary mapping figure names to Matplotlib figures. Keys follow
            the pattern ``"inter_event_times_{action_name}"`` and
            ``"activity_by_hour_{action_name}"``.
        """
        figures: dict[str, Figure] = {}
        hours: list[int] = list(range(24))

        for action_name, stats in result.by_action_type.items():
            if stats.inter_event_times:
                fig_iet: Figure
                ax_iet: Axes
                fig_iet, ax_iet = plt.subplots(figsize=(8, 6))
                iet_array: np.ndarray = np.asarray(stats.inter_event_times, dtype=float)
                min_val: float = max(float(iet_array.min()), 1.0)
                max_val: float = float(iet_array.max())
                bins: list[float] = np.logspace(
                    np.log10(min_val), np.log10(max_val + 1), 50
                ).tolist()
                counts, edges = np.histogram(iet_array, bins=bins)
                bin_centers = (edges[:-1] + edges[1:]) / 2
                mask = counts > 0
                ax_iet.plot(bin_centers[mask], counts[mask])
                ax_iet.set_xscale("log")
                ax_iet.set_yscale("log")
                ax_iet.set_title(f"Inter-event time distribution — {action_name}")
                ax_iet.set_xlabel("steps between actions")
                ax_iet.set_ylabel("frequency")
                fig_iet.tight_layout()
                figures[f"inter_event_times_{action_name}"] = fig_iet

            activity: list[int] = [
                stats.activity_by_hour.get(hour, 0) for hour in hours
            ]
            fig_hour: Figure
            ax_hour: Axes
            fig_hour, ax_hour = plt.subplots(figsize=(8, 6))
            ax_hour.bar(hours, activity)
            ax_hour.set_title(f"Activity by clock-hour — {action_name}")
            ax_hour.set_xlabel("hour of day")
            ax_hour.set_ylabel("event count")
            ax_hour.set_xticks(range(0, 24, 2))
            fig_hour.tight_layout()
            figures[f"activity_by_hour_{action_name}"] = fig_hour

        return figures

    def draw_figs_all(
        self, individual_results: list[TemporalDynamicsResult]
    ) -> dict[str, Figure]:
        """Draw temporal-dynamics figures pooled across multiple runs.

        Produces one inter-event time log-log line plot and one
        activity-by-hour bar chart (mean across runs) for each action type.

        Args:
            individual_results: List of results from :meth:`analyze`, one per run.

        Returns:
            Dictionary mapping figure names to Matplotlib figures. Keys follow
            the same pattern as :meth:`draw_figs`.
        """
        figures: dict[str, Figure] = {}
        hours: list[int] = list(range(24))

        pooled_iets: defaultdict[str, list[int]] = defaultdict(list)
        hour_accumulator: defaultdict[str, defaultdict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for result in individual_results:
            for action_name, stats in result.by_action_type.items():
                pooled_iets[action_name].extend(stats.inter_event_times)
                for hour, count in stats.activity_by_hour.items():
                    hour_accumulator[action_name][hour].append(count)

        for action_name in sorted(pooled_iets.keys()):
            iet_list: list[int] = pooled_iets[action_name]
            if iet_list:
                fig_iet: Figure
                ax_iet: Axes
                fig_iet, ax_iet = plt.subplots(figsize=(8, 6))
                iet_array: np.ndarray = np.asarray(iet_list, dtype=float)
                min_val: float = max(float(iet_array.min()), 1.0)
                max_val: float = float(iet_array.max())
                bins: list[float] = np.logspace(
                    np.log10(min_val), np.log10(max_val + 1), 50
                ).tolist()
                counts, edges = np.histogram(iet_array, bins=bins)
                bin_centers = (edges[:-1] + edges[1:]) / 2
                mask = counts > 0
                ax_iet.plot(bin_centers[mask], counts[mask])
                ax_iet.set_xscale("log")
                ax_iet.set_yscale("log")
                ax_iet.set_title(
                    f"Inter-event time distribution (all runs) — {action_name}"
                )
                ax_iet.set_xlabel("steps between actions")
                ax_iet.set_ylabel("frequency")
                fig_iet.tight_layout()
                figures[f"inter_event_times_{action_name}"] = fig_iet

            mean_activity: list[float] = []
            for hour in hours:
                run_counts: list[int] = hour_accumulator[action_name].get(hour, [])
                mean_activity.append(float(np.mean(run_counts)) if run_counts else 0.0)

            fig_hour: Figure
            ax_hour: Axes
            fig_hour, ax_hour = plt.subplots(figsize=(8, 6))
            ax_hour.bar(hours, mean_activity)
            ax_hour.set_title(f"Activity by clock-hour (all runs) — {action_name}")
            ax_hour.set_xlabel("hour of day")
            ax_hour.set_ylabel("mean event count")
            ax_hour.set_xticks(range(0, 24, 2))
            fig_hour.tight_layout()
            figures[f"activity_by_hour_{action_name}"] = fig_hour

        return figures

    def build_summary(self, result: TemporalDynamicsResult) -> Panel:
        """Build a Rich summary panel for temporal-dynamics analysis.

        Args:
            result: Output returned by :meth:`analyze`.

        Returns:
            Rich panel summarising per-action-type temporal statistics.
        """
        table: Table = Table(title="Temporal Dynamics Summary")
        table.add_column("Action type")
        table.add_column("Events")
        table.add_column("Burstiness B")
        table.add_column("Memory M")
        table.add_column("Mean IET (steps)")

        for action_name, stats in result.by_action_type.items():
            mean_iet: str = (
                f"{np.mean(stats.inter_event_times):.1f}"
                if stats.inter_event_times
                else "-"
            )
            table.add_row(
                action_name,
                str(stats.n_events),
                "-"
                if stats.mean_burstiness is None
                else f"{stats.mean_burstiness:.3f}",
                "-" if stats.mean_memory is None else f"{stats.mean_memory:.3f}",
                mean_iet,
            )

        table.add_section()
        table.add_row("Total", str(result.n_events), "", "", "")
        table.add_row("Agents", str(result.n_agents), "", "", "")
        table.add_row(
            f"Circadian autocorr (lag {self.period_steps})",
            "-"
            if result.circadian_autocorr is None
            else f"{result.circadian_autocorr:.3f}",
            "",
            "",
            "",
        )

        return Panel.fit(table, title="Analysis Summary", border_style="cyan")

    def build_summary_all(self, result: TemporalDynamicsAggregateResult) -> Panel:
        """Build a Rich summary panel for multi-run temporal-dynamics analysis.

        Shows mean ± standard deviation across runs for burstiness, memory,
        mean inter-event time, and event count per action type.

        Args:
            result: Output returned by :meth:`analyze_stores`.

        Returns:
            Rich panel summarising aggregated temporal statistics.
        """

        def _fmt(mean: Optional[float], std: Optional[float], decimals: int = 3) -> str:
            if mean is None:
                return "-"
            fmt = f".{decimals}f"
            if std is None:
                return f"{mean:{fmt}}"
            return f"{mean:{fmt}}±{std:{fmt}}"

        table: Table = Table(title=f"Temporal Dynamics Summary ({result.n_runs} runs)")
        table.add_column("Action type")
        table.add_column("Events (mean±std)")
        table.add_column("Burstiness B (mean±std)")
        table.add_column("Memory M (mean±std)")
        table.add_column("Mean IET (mean±std)")

        for action_name, stats in result.by_action_type.items():
            table.add_row(
                action_name,
                _fmt(stats.mean_n_events, stats.std_n_events, decimals=1),
                _fmt(stats.mean_burstiness_mean, stats.mean_burstiness_std),
                _fmt(stats.mean_memory_mean, stats.mean_memory_std),
                _fmt(stats.mean_iet_mean, stats.mean_iet_std, decimals=1),
            )

        table.add_section()
        table.add_row(
            f"Circadian autocorr (lag {self.period_steps})",
            _fmt(result.circadian_autocorr_mean, result.circadian_autocorr_std),
            "",
            "",
            "",
        )
        table.add_row("Runs", str(result.n_runs), "", "", "")

        return Panel.fit(table, title="Multi-Run Analysis Summary", border_style="cyan")

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

    def _clock_hour(self, record: TimedRecord) -> Optional[int]:
        """Clock-hour of a record, from datetime or step-within-period."""
        if isinstance(record.time, datetime):
            return record.time.hour
        if self.period_steps > 0:
            return int(record.time_step) % self.period_steps
        return None
