from __future__ import annotations

from typing import TypeAlias, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

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

    def __init__(self, exclude_agent_ids: list[int] = []):
        """Initialization.

        Args:
            exclude_agent_ids (list[int], optional): A list of agent IDs to exclude from the analysis. Defaults to an empty list.
        """
        self.exclude_agent_ids = exclude_agent_ids

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
        return None

    def draw_figs(
        self,
        result: StressData,
    ) -> dict[str, Figure]:
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
        return {}
