from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TypeAlias, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from rich.console import Group, RenderableType
from rich.table import Table

from .base import AnalyzerBase
from .records import BaseRecord, ObsRecord, TweetRecord
from .store import RecordStore

StressActionData: TypeAlias = list[dict[str, int | bool | list[int]]]

STRESS_REASON_CATEGORIES = {
    0: "Acceptable consumption level",
    1: "You have not consumed any items",
    2: "You have not consumed enough items",
    3: "You have consumed too many items",
    4: "Acceptable movement level",
    5: "at home",
    6: "You have not moved at all",
    7: "You have not moved enough",
    8: "You have moved too much",
    9: "Your economic condition is acceptable",
    10: "You cannot buy enough goods",
    11: "You have less wealth than others",
    12: "Your wealth has recently decreased",
    13: "Acceptable sleep level",
    14: "You have not slept enough",
    15: "Your sleep rhythm is irregular",
}


@dataclass
class StressActionAnalyzer(AnalyzerBase[StressActionData, list[StressActionData]]):
    """Analyse how agent actions relate to the reasons behind their stress.

    EconSimulacra's stress model assigns a numeric stress level and a set
    of *reason codes* to each stress domain at every time step. Reason
    codes identify *why* an agent is stressed (e.g., ``"You have not
    consumed enough items"``). :class:`StressActionAnalyzer` cross-
    references these stress reasons with the actions the agent actually
    performed in the same time step, producing an *action-rate-by-reason*
    table.

    For each ``(stress_type, action_type, reason_id)`` triple the
    *action rate* is:

    .. math::

        r(\\text{stress\\_type},\\; \\text{action\\_type},\\; \\text{reason}) =
        \\frac{\\text{# steps where reason is active AND action was taken}}
              {\\text{# steps where reason is active}}

    This reveals which stress reasons actually trigger which actions.
    For example, if agents with ``consumption_history_stress_reason == 2``
    ("You have not consumed enough items") almost always perform a
    ``consumption`` action, that confirms the stress-action loop is
    functioning as designed.

    Attributes:
        name (str): Analyzer name.
        action_types (list[str]): Action types included in the analysis.
        stress_threshold (int): Informational threshold stored for
            reference; not used for filtering in the main pipeline.
        exclude_agent_ids (list[int]): Agent IDs excluded from the
            analysis.
        stress_reasons (dict[int, str]): Mapping from reason-code integer
            to human-readable description.
    """

    name = "stress_action"
    action_types: list[str] = field(
        default_factory=lambda: [
            "sleep_start",
            "move",
            "consumption",
            "order",
            "proposal",
            "tweet",
            "follow",
            "unfollow",
        ]
    )
    stress_threshold: int = 50
    exclude_agent_ids: list[int] = field(default_factory=list)
    stress_reasons: dict[int, str] = field(
        default_factory=lambda: STRESS_REASON_CATEGORIES
    )

    def analyze(self, store: RecordStore) -> StressActionData:
        """Build a per-step stress-and-action snapshot for all agents.

        For every ``(agent_id, time_step)`` pair found in the store, the
        method collects:

        * Stress levels and reason-code lists from memory observations
          (:class:`~econsimulacra.log_analyses.records.ObsRecord` with
          ``obs_type == "memory"``).
        * Boolean flags indicating whether each action type was performed.

        Args:
            store (RecordStore): Record store to analyse.

        Returns:
            StressActionData: A list of dicts, one per observed
            ``(agent_id, time_step)`` pair. Each dict contains:

            * ``"agent_id"`` (int)
            * ``"time_step"`` (int)
            * ``"{domain}_history_stress"`` (int) – stress level per domain
            * ``"{domain}_history_stress_reason"`` (list[int]) – active
              reason codes for the domain
            * ``"{action_type}"`` (bool) – whether the action was performed

            Entries with fewer than 3 populated keys are excluded (i.e.,
            time steps where no relevant records were found).
        """
        self._prepare_time_axis(store)
        if self._time_axis_config is not None:
            max_time_step: int = int(self._time_axis_config.x_max)
        agent_id2name: dict[int, str] = self.get_agent_id2name(store)
        stress_action_data: StressActionData = []
        for agent_id in agent_id2name.keys():
            for time_step in range(max_time_step):
                records: list[BaseRecord] = store.get_by_agent_time(agent_id, time_step)
                stress_action_value: Optional[dict[str, int | bool | list[int]]] = (
                    self._generate_agent_time_stress_action_value(records)
                )
                if stress_action_value is not None:
                    stress_action_data.append(stress_action_value)
        return stress_action_data

    def _generate_agent_time_stress_action_value(
        self, records: list[BaseRecord]
    ) -> Optional[dict[str, int | bool | list[int]]]:
        """Build the stress-action snapshot dict for a single agent–step.

        Scans the supplied records for the agent's memory observation and
        action records at one time step, then returns a unified dict.

        Args:
            records (list[BaseRecord]): All records for one agent at one
                time step.

        Returns:
            dict or None: A dict of the form::

                {
                    "agent_id": 0,
                    "time_step": 10,
                    "consumption_history_stress": 100,
                    "consumption_history_stress_reason": [3],
                    "sleep_start": True,
                    "move": False,
                    "consumption": True,
                    ...
                }

            Returns ``None`` if the agent is in ``exclude_agent_ids`` or
            if fewer than 3 keys were populated (e.g., no memory
            observation was found).
        """
        stress_action_value: dict[str, int | bool | list[int]] = {}
        for record in records:
            if hasattr(record, "agent_id"):
                agent_id = cast(int, getattr(record, "agent_id"))
                if agent_id in self.exclude_agent_ids:
                    return None
                else:
                    stress_action_value["agent_id"] = agent_id
            if hasattr(record, "time_step"):
                time_step = cast(int, getattr(record, "time_step"))
                stress_action_value["time_step"] = time_step
            if record.type == "observation":
                obs_record = cast(ObsRecord, record)
                if obs_record.obs_type == "memory":
                    memory_key: str
                    for memory_key, memory_value in obs_record.obs.items():
                        if memory_key.endswith("_history_stress"):
                            stress_action_value[memory_key] = int(memory_value)
                        elif memory_key.endswith("_history_stress_reason"):
                            reason_sentence: str = cast(str, memory_value)
                            for reason_id, reason_text in self.stress_reasons.items():
                                if reason_text in reason_sentence:
                                    if memory_key not in stress_action_value:
                                        stress_action_value[memory_key] = []
                                    value = stress_action_value[memory_key]
                                    assert isinstance(value, list)
                                    value.append(reason_id)
            elif record.type in self.action_types:
                if record.type == "tweet":
                    tweet_record = cast(TweetRecord, record)
                    if len(tweet_record.message) == 0:
                        continue
                stress_action_value[record.type] = True
        if len(stress_action_value) > 2:
            for action_type in self.action_types:
                if action_type not in stress_action_value:
                    stress_action_value[action_type] = False
            dic_to_add: dict[str, int | bool | list[int]] = {}
            for key in stress_action_value.keys():
                if key.endswith("_history_stress_reason"):
                    stress_reason_key = key
                    stress_key = key.replace("_reason", "")
                    if stress_key not in stress_action_value:
                        dic_to_add[stress_key] = 0
                if key.endswith("_history_stress"):
                    stress_key = key
                    stress_reason_key = key + "_reason"
                    if stress_reason_key not in stress_action_value:
                        dic_to_add[stress_reason_key] = []
            stress_action_value.update(dic_to_add)
            return stress_action_value
        else:
            return None

    def analyze_stores(self, stores: list[RecordStore]) -> list[StressActionData]:
        return [self.analyze(store) for store in stores]

    @staticmethod
    def _stress_types(result: StressActionData) -> list[str]:
        stress_types: set[str] = set()
        for row in result:
            for key in row:
                if key.endswith("_history_stress"):
                    stress_types.add(key)
        return sorted(stress_types)

    def _calc_action_rates_by_reason(
        self,
        result: StressActionData,
        stress_type: str,
        action_type: str,
    ) -> dict[int, float]:
        reason_key = f"{stress_type}_reason"
        true_counts: dict[int, int] = defaultdict(int)
        total_counts: dict[int, int] = defaultdict(int)
        for row in result:
            reasons = row.get(reason_key)
            action_value = row.get(action_type)
            if not isinstance(reasons, list) or len(reasons) == 0:
                continue
            if not isinstance(action_value, bool):
                continue
            for reason_id in reasons:
                if not isinstance(reason_id, int):
                    continue
                total_counts[reason_id] += 1
                true_counts[reason_id] += int(action_value)
        return {
            reason_id: true_counts[reason_id] / total_counts[reason_id]
            for reason_id in sorted(total_counts)
            if total_counts[reason_id] > 0
        }

    def draw_figs(
        self,
        result: StressActionData,
    ) -> dict[str, Figure]:
        figs: dict[str, Figure] = {}
        stress_types = self._stress_types(result)
        for stress_type in stress_types:
            for action_type in self.action_types:
                rates = self._calc_action_rates_by_reason(
                    result=result,
                    stress_type=stress_type,
                    action_type=action_type,
                )
                if len(rates) == 0:
                    continue
                reason_ids = list(rates.keys())
                values = [rates[r] for r in reason_ids]
                labels = [
                    f"{r}: {self.stress_reasons.get(r, 'Unknown')}" for r in reason_ids
                ]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(range(len(reason_ids)), values)
                ax.set_ylabel("Action rate")
                ax.set_ylim(0.0, 1.0)
                ax.set_xticks(range(len(reason_ids)))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                fig.tight_layout()
                fig_name = f"{stress_type}_{action_type}"
                figs[fig_name] = fig

        return figs

    def draw_figs_all(
        self,
        individual_results: list[StressActionData],
    ) -> dict[str, Figure]:
        figs: dict[str, Figure] = {}

        if len(individual_results) == 0:
            return figs

        stress_types = sorted(
            {
                stress_type
                for result in individual_results
                for stress_type in self._stress_types(result)
            }
        )

        for stress_type in stress_types:
            for action_type in self.action_types:
                per_result_rates: list[dict[int, float]] = [
                    self._calc_action_rates_by_reason(
                        result=result,
                        stress_type=stress_type,
                        action_type=action_type,
                    )
                    for result in individual_results
                ]

                reason_ids = sorted(
                    {reason_id for rates in per_result_rates for reason_id in rates}
                )

                if len(reason_ids) == 0:
                    continue

                means: list[float] = []
                stds: list[float] = []

                for reason_id in reason_ids:
                    values = [
                        rates[reason_id]
                        for rates in per_result_rates
                        if reason_id in rates
                    ]

                    means.append(float(np.mean(values)))
                    stds.append(
                        float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                    )

                labels = [
                    f"{r}: {self.stress_reasons.get(r, 'Unknown')}" for r in reason_ids
                ]

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(
                    range(len(reason_ids)),
                    means,
                    yerr=stds,
                    capsize=4,
                )

                ax.set_ylabel("Action rate")
                ax.set_ylim(0.0, 1.0)

                ax.set_xticks(range(len(reason_ids)))
                ax.set_xticklabels(labels, rotation=45, ha="right")

                fig.tight_layout()

                fig_name = f"{stress_type}_{action_type}"
                figs[fig_name] = fig

        return figs

    def build_summary(self, result: StressActionData) -> RenderableType:
        group_items: list[RenderableType] = []

        for stress_type in self._stress_types(result):
            table = Table(title=f"{stress_type}: action rate by stress reason")
            table.add_column("Reason", style="cyan")
            table.add_column("N", justify="right")
            for action_type in self.action_types:
                table.add_column(action_type, justify="right")

            reason_ids = sorted(
                {
                    reason_id
                    for action_type in self.action_types
                    for reason_id in self._calc_action_rates_by_reason(
                        result, stress_type, action_type
                    )
                }
            )

            if len(reason_ids) == 0:
                continue

            for reason_id in reason_ids:
                reason_label = self.stress_reasons.get(reason_id, "Unknown")
                reason_key = f"{stress_type}_reason"
                n = 0
                for row in result:
                    reasons = row.get(reason_key)
                    if not isinstance(reasons, list):
                        continue
                    if reason_id in reasons:
                        n += 1
                row_values: list[str] = [
                    f"{reason_id}: {reason_label}",
                    str(n),
                ]

                for action_type in self.action_types:
                    rates = self._calc_action_rates_by_reason(
                        result, stress_type, action_type
                    )
                    rate = rates.get(reason_id)
                    row_values.append("-" if rate is None else f"{rate:.3f}")

                table.add_row(*row_values)

            group_items.append(table)

        return Group(*group_items)

    def build_summary_all(
        self,
        results: list[StressActionData],
    ) -> RenderableType:
        group_items: list[RenderableType] = []

        if len(results) == 0:
            table = Table(title="Stress-action summary")
            table.add_column("Message")
            table.add_row("No results")
            return table

        stress_types = sorted(
            {
                stress_type
                for result in results
                for stress_type in self._stress_types(result)
            }
        )

        for stress_type in stress_types:
            table = Table(
                title=f"{stress_type}: mean ± std action rate by stress reason"
            )
            table.add_column("Reason", style="cyan")
            table.add_column("Runs", justify="right")
            for action_type in self.action_types:
                table.add_column(action_type, justify="right")

            reason_ids = sorted(
                {
                    reason_id
                    for result in results
                    for action_type in self.action_types
                    for reason_id in self._calc_action_rates_by_reason(
                        result, stress_type, action_type
                    )
                }
            )

            if len(reason_ids) == 0:
                continue

            for reason_id in reason_ids:
                reason_label = self.stress_reasons.get(reason_id, "Unknown")

                row_values: list[str] = [
                    f"{reason_id}: {reason_label}",
                    str(len(results)),
                ]

                for action_type in self.action_types:
                    values: list[float] = []

                    for result in results:
                        rates = self._calc_action_rates_by_reason(
                            result, stress_type, action_type
                        )
                        if reason_id in rates:
                            values.append(rates[reason_id])

                    if len(values) == 0:
                        row_values.append("-")
                    else:
                        mean = float(np.mean(values))
                        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                        row_values.append(f"{mean:.3f} ± {std:.3f}")

                table.add_row(*row_values)

            group_items.append(table)

        return Group(*group_items)
