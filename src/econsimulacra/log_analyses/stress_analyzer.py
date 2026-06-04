from __future__ import annotations

from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import AnalyzerBase
from .records import ObsRecord
from .store import RecordStore

StressData: TypeAlias = dict[int, dict[str, dict[int, float]]]


class StressAnalyzer(AnalyzerBase[StressData, None]):
    """Stress analyzer.

    StressAnalyzer analyzes stress data for each household stress_type "*_history_stress".
    """

    name = "stress"

    def __init__(self, exclude_agent_ids: list[int] = []):
        """Initialization.

        Args:
            exclude_agent_ids (list[int], optional): A list of agent IDs to exclude from the analysis. Defaults to an empty list.
        """
        self.exclude_agent_ids = exclude_agent_ids

    def analyze(self, store: RecordStore) -> StressData:
        """Analyzes stress data for each household stress_type "*_history_stress".

        Args:
            store (RecordStore): The record store containing the records to analyze.

        Returns:
            A dictionary mapping agent IDs to a dictionary mapping stress types to
            a dictionary of timestamps and stress values.

        Note:
            Returned dictionary structure:
            {
                agent_id: {
                    stress_type: {
                        timestamp: stress_value,
                        ...
                    },
                    ...
                },
                ...
            }
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
            memory_value: dict[str, Any]
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
