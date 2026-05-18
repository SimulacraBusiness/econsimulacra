from __future__ import annotations

from datetime import datetime
from typing import Any, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .base import AnalyzerBase
from .records import ObsRecord
from .store import RecordStore


class StressAnalyzer(AnalyzerBase[dict[int, dict[str, dict[datetime | int, float]]]]):
    """Stress analyzer.

    StressAnalyzer analyzes stress data for each household stress_type "*_history_stress".
    """

    name = "stress"

    def analyze(
        self, store: RecordStore
    ) -> dict[int, dict[str, dict[datetime | int, float]]]:
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
        result: dict[int, dict[str, dict[datetime | int, float]]] = {}

        for record in store.typed(ObsRecord):
            if record.obs_type != "memory":
                continue
            memory_key: str
            memory_value: dict[str, Any]
            for memory_key, memory_value in record.obs.items():
                if memory_key.endswith("_history_stress"):
                    agent_id = record.agent_id
                    stress_type = memory_key.removesuffix("_history_stress")
                    time = record.time
                    stress_value = float(cast(float | int, memory_value))
                    if agent_id not in result:
                        result[agent_id] = {}
                    if stress_type not in result[agent_id]:
                        result[agent_id][stress_type] = {}
                    result[agent_id][stress_type][time] = stress_value

        return result

    def draw_figs(
        self,
        result: dict[int, dict[str, dict[datetime | int, float]]],
    ) -> dict[str, Figure]:
        fig_dic: dict[str, Figure] = {}
        for agent_id, stress_type2time_stress in result.items():
            for stress_type, time_stress in stress_type2time_stress.items():
                fig_key: str = f"stress_{stress_type}"
                if fig_key not in fig_dic:
                    fig: Figure = Figure(figsize=(15, 6))
                    ax: Axes = fig.add_subplot(1, 1, 1)
                    fig_dic[fig_key] = fig
                fig = fig_dic[fig_key]
                ax = fig.axes[0]
                times = list(time_stress.keys())
                stress_values = list(time_stress.values())
                x = np.arange(len(times))
                ax.plot(x, stress_values, label=f"Agent {agent_id}")
        return fig_dic
