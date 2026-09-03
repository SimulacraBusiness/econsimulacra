from __future__ import annotations

from typing import Optional, TypeAlias, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from rich.console import RenderableType
from rich.panel import Panel
from rich.table import Table

from .base import AnalyzerBase
from .records import ObsRecord
from .store import RecordStore

StressData: TypeAlias = dict[int, dict[str, dict[int, float]]]


class StressAnalyzer(AnalyzerBase[StressData, None]):
    """Analyse per-agent stress levels over simulation time.

    EconSimulacra agents maintain internal stress values for different life
    domains (e.g., consumption, movement, economic condition, sleep). These
    values are periodically written to agent memory and recorded as
    :class:`~econsimulacra.log_analyses.records.ObsRecord` entries with
    ``obs_type == "memory"``.

    :class:`StressAnalyzer` extracts memory keys that end in
    ``"_history_stress"`` and returns a three-level dict::

        {
            agent_id: {
                stress_type: {time_step: stress_value, ...},
                ...
            },
            ...
        }

    where ``stress_type`` is the prefix before ``"_history_stress"``
    (e.g., ``"consumption"``, ``"movement"``, ``"economic"``, ``"sleep"``).
    Higher stress values indicate that an agent is further from its target
    state for that domain.

    :meth:`draw_figs` plots each stress type as a time-series overlay
    across all agents, allowing quick identification of agents under
    sustained stress and the time periods where stress peaks.

    Note:
        Agents listed in ``exclude_agent_ids`` (e.g., firm agents that do
        not have stress models) are silently skipped.
    """

    name = "stress"

    def __init__(
        self,
        exclude_agent_ids: Optional[list[int]] = None,
        max_stress: float = 100.0,
        high_stress_threshold: float = 0.7,
    ) -> None:
        """Initialization.

        Args:
            exclude_agent_ids (list[int], optional): A list of agent IDs to
                exclude from the analysis. Defaults to an empty list.
            max_stress (float, optional): Maximum value used to normalize
                stress into the interval ``[0, 1]``. Defaults to ``100.0``.
            high_stress_threshold (float, optional): Normalized threshold
                above which stress is considered high. Defaults to ``0.7``.

        Returns:
            None: This method does not return a value.

        Note:
            ``max_stress`` must be positive and ``high_stress_threshold``
            must be in the interval ``[0, 1]``.
        """
        if max_stress <= 0:
            raise ValueError("max_stress must be positive.")
        if not 0.0 <= high_stress_threshold <= 1.0:
            raise ValueError("high_stress_threshold must be between 0 and 1.")
        self.exclude_agent_ids = exclude_agent_ids or []
        self.max_stress = max_stress
        self.high_stress_threshold = high_stress_threshold

    def analyze(self, store: RecordStore) -> StressData:
        """Extract per-agent, per-domain stress time series from *store*.

        Scans all :class:`~econsimulacra.log_analyses.records.ObsRecord`
        entries whose ``obs_type`` is ``"memory"``. For each record the
        observation dict is searched for keys ending in
        ``"_history_stress"``; the suffix is stripped to obtain the
        ``stress_type`` label and the numeric value is recorded at the
        corresponding ``time_step``.

        Args:
            store (RecordStore): Record store containing memory observation
                records.

        Returns:
            StressData: Three-level dict of the form::

                {
                    agent_id: {
                        stress_type: {
                            time_step: stress_value,
                            ...
                        },
                        ...
                    },
                    ...
                }

            Agents listed in ``exclude_agent_ids`` are omitted.

        Note:
            When multiple memory observations exist for the same agent and
            time step, the last value encountered for each stress type is
            retained.
        """
        self._prepare_time_axis(store)
        result: dict[int, dict[str, dict[int, float]]] = {}
        for record in store.typed(ObsRecord):
            if record.obs_type != "memory":
                continue
            agent_id = record.agent_id
            if agent_id in self.exclude_agent_ids:
                continue
            memory_key: str
            for memory_key, memory_value in record.obs.items():
                if memory_key.endswith("_history_stress"):
                    stress_type = memory_key.removesuffix("_history_stress")
                    time = record.time_step
                    stress_value = float(cast(float | int, memory_value))
                    if agent_id not in result:
                        result[agent_id] = {}
                    if stress_type not in result[agent_id]:
                        result[agent_id][stress_type] = {}
                    result[agent_id][stress_type][time] = stress_value
        return result

    def analyze_stores(self, stores: list[RecordStore]) -> None:
        """Return no aggregate result for multiple stores.

        Args:
            stores (list[RecordStore]): Record stores from multiple
                simulation runs.

        Returns:
            None: Multi-run stress analysis is not implemented.

        Note:
            Single-run results remain available through :meth:`analyze`.
        """
        return None

    def draw_figs(
        self,
        result: StressData,
    ) -> dict[str, Figure]:
        """Draw per-category stress time series for all agents.

        Args:
            result (StressData): Output returned by :meth:`analyze`.

        Returns:
            dict[str, Figure]: Mapping from stress-category figure names to
                Matplotlib figures.

        Note:
            Excluded agents are absent because filtering occurs during
            analysis.
        """
        fig_dic: dict[str, Figure] = {}
        for agent_id, stress_type2time_stress in result.items():
            for stress_type, time_stress in stress_type2time_stress.items():
                fig_key: str = f"stress_{stress_type}"
                if fig_key not in fig_dic:
                    fig: Figure
                    ax: Axes
                    fig, ax = plt.subplots(figsize=(8, 6))
                    fig_dic[fig_key] = fig
                fig = fig_dic[fig_key]
                ax = fig.axes[0]
                times = list(time_stress.keys())
                stress_values = list(time_stress.values())
                last_time: int = max(times)
                if self._time_axis_config is not None:
                    max_time = int(self._time_axis_config.x_max)
                    if last_time < max_time:
                        times.append(max_time)
                        stress_values.append(0)
                ax.plot(times, stress_values, label=f"Agent {agent_id}")
                self._apply_time_axis(ax)
        return fig_dic

    def draw_figs_all(self, individual_results: list[StressData]) -> dict[str, Figure]:
        """Return figures for multiple stress-analysis results.

        Args:
            individual_results (list[StressData]): One stress result per
                simulation run.

        Returns:
            dict[str, Figure]: Empty mapping because multi-run stress figures
                are not implemented.

        Note:
            The argument is accepted to satisfy the analyzer interface.
        """
        return {}

    def build_summary(self, result: StressData) -> RenderableType:
        """Build a summary of high-stress exposure and persistence.

        Args:
            result (StressData): Output returned by :meth:`analyze`.

        Returns:
            RenderableType: Rich panel containing one row per agent and
                stress category.

        Note:
            Stress values are normalized by ``max_stress``. Exposure is the
            mean threshold exceedance. Persistence is the exceedance-weighted
            duration of consecutive high-stress observations. Persistence is
            zero when there is no threshold exceedance.
        """
        if not result:
            return Panel.fit(
                "No stress result is available.",
                title="Stress Summary",
                border_style="yellow",
            )

        table = Table(
            title="High-stress metrics",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Agent ID", justify="right")
        table.add_column("Stress Type", style="green")
        table.add_column("HighStressExposure", justify="right")
        table.add_column("StressPersistence", justify="right")

        for agent_id in sorted(result):
            for stress_type in sorted(result[agent_id]):
                exposure, persistence = self._calc_high_stress_metrics(
                    result[agent_id][stress_type]
                )
                table.add_row(
                    str(agent_id),
                    stress_type,
                    f"{exposure:.4f}",
                    f"{persistence:.4f}",
                )

        return Panel.fit(table, title="Stress Summary", border_style="cyan")

    def _calc_high_stress_metrics(
        self,
        time_stress: dict[int, float],
    ) -> tuple[float, float]:
        """Calculate high-stress exposure and persistence for one series.

        Args:
            time_stress (dict[int, float]): Mapping from simulation step to
                an unnormalized stress value.

        Returns:
            tuple[float, float]: ``HighStressExposure`` followed by
                ``StressPersistence``.

        Note:
            Observations are ordered by simulation step. A normalized value
            equal to the threshold has zero exceedance and resets the
            consecutive high-stress duration.
        """
        if not time_stress:
            return 0.0, 0.0

        total_exceedance = 0.0
        duration_weighted_exceedance = 0.0
        high_stress_duration = 0

        for time_step in sorted(time_stress):
            normalized_stress = min(
                1.0,
                max(0.0, time_stress[time_step] / self.max_stress),
            )
            exceedance = max(
                0.0,
                normalized_stress - self.high_stress_threshold,
            )
            if exceedance > 0.0:
                high_stress_duration += 1
            else:
                high_stress_duration = 0
            total_exceedance += exceedance
            duration_weighted_exceedance += exceedance * high_stress_duration

        exposure = total_exceedance / len(time_stress)
        persistence = (
            duration_weighted_exceedance / total_exceedance
            if total_exceedance > 0.0
            else 0.0
        )
        return exposure, persistence
