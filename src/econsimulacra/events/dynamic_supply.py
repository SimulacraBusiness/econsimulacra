from __future__ import annotations

from datetime import datetime
from math import isfinite
from typing import TYPE_CHECKING, Any, Optional

from ..agents import Agent
from ..date_utils import get_corresponding_value, to_datetime
from ..logs import AgentGenerationLog, Log
from .base import Event, EventTrigger

if TYPE_CHECKING:
    from ..envs import Environment


SupplySpec = tuple[Optional[float], dict[str, float]]


class DynamicSupply(Event):
    """Supply inventories according to time-dependent configuration entries.

    The event follows the same two-phase lifecycle as :class:`ConstantSupply`:
    ``AgentGenerationLog`` calls capture the initial inventories, while step calls
    apply the supply selected for the environment's current display time.
    """

    def __init__(self, trigger: EventTrigger, config: dict[str, Any]) -> None:
        """Initialize a dynamic supply event.

        Args:
            trigger: Trigger used for agent-generation and periodic step calls.
            config: Event configuration containing ``suppliedAgentNames`` and
                ``supplies`` entries. Each supply entry has ``start``, ``end``,
                and exactly one of ``supplyRatio`` or ``itemAmounts``.

        Returns:
            None.

        Note:
            Time intervals use the half-open semantics of
            :func:`get_corresponding_value`: ``start <= time < end``.
        """
        super().__init__(trigger=trigger, config=config)
        self._validate_trigger(trigger)
        self.supplied_agent_names = self._parse_agent_names(config)
        self.time_span2supply = self._parse_supplies(config)
        self.agent_id2initial_supply: dict[int, dict[str, float | int]] = {}

    def _validate_trigger(self, trigger: EventTrigger) -> None:
        """Validate the two trigger modes required by dynamic supply.

        Args:
            trigger: Trigger to validate.

        Returns:
            None.

        Note:
            ``at`` and ``every`` may be combined. This permits a product launch
            exactly at one step followed by regular replenishment.
        """
        if len(trigger.logs) == 0:
            raise ValueError(
                "DynamicSupply requires an AgentGenerationLog trigger in 'with'."
            )
        if trigger.every is None and trigger.at is None:
            raise ValueError("DynamicSupply requires 'every' or 'at' in trigger.")
        if trigger.between is not None:
            raise ValueError(
                "DynamicSupply selects its active period from 'supplies'; "
                "do not use 'between' in trigger."
            )

    def _parse_agent_names(self, config: dict[str, Any]) -> tuple[str, ...]:
        """Parse configured target-agent name fragments.

        Args:
            config: Dynamic-supply configuration.

        Returns:
            Non-empty tuple of agent-name fragments.

        Note:
            Fragment matching preserves ``ConstantSupply`` behavior for names
            whose generated agent ID is appended to the configured base name.
        """
        raw_names = config.get("suppliedAgentNames")
        if not isinstance(raw_names, (list, tuple)) or not raw_names:
            raise ValueError("DynamicSupply requires non-empty suppliedAgentNames.")
        names = tuple(str(name) for name in raw_names)
        if any(not name for name in names):
            raise ValueError("suppliedAgentNames cannot contain empty names.")
        return names

    def _parse_supplies(
        self, config: dict[str, Any]
    ) -> dict[tuple[int | str, int | str], SupplySpec]:
        """Parse and validate time-dependent supply specifications.

        Args:
            config: Dynamic-supply configuration.

        Returns:
            Mapping accepted by :func:`get_corresponding_value`.

        Note:
            Ratio supplies use captured initial inventory. Absolute supplies add
            only the explicitly configured items.
        """
        raw_supplies = config.get("supplies")
        if not isinstance(raw_supplies, (list, tuple)) or not raw_supplies:
            raise ValueError("DynamicSupply requires non-empty supplies.")

        result: dict[tuple[int | str, int | str], SupplySpec] = {}
        normalized_spans: list[tuple[int | datetime, int | datetime]] = []
        for raw_supply in raw_supplies:
            if not isinstance(raw_supply, dict):
                raise TypeError("Each DynamicSupply entry must be a mapping.")
            if "start" not in raw_supply or "end" not in raw_supply:
                raise ValueError("Each DynamicSupply entry requires start and end.")
            start = raw_supply["start"]
            end = raw_supply["end"]
            if not isinstance(start, (int, str)) or isinstance(start, bool):
                raise TypeError("DynamicSupply start must be an integer or ISO time.")
            if not isinstance(end, (int, str)) or isinstance(end, bool):
                raise TypeError("DynamicSupply end must be an integer or ISO time.")
            normalized_start = to_datetime(start)
            normalized_end = to_datetime(end)
            if not self._is_ordered_span(normalized_start, normalized_end):
                raise ValueError("DynamicSupply requires start < end.")

            has_ratio = "supplyRatio" in raw_supply
            has_amounts = "itemAmounts" in raw_supply
            if has_ratio == has_amounts:
                raise ValueError(
                    "Each DynamicSupply entry requires exactly one of "
                    "supplyRatio or itemAmounts."
                )
            ratio: Optional[float] = None
            item_amounts: dict[str, float] = {}
            if has_ratio:
                ratio = self._validate_amount(raw_supply["supplyRatio"], "supplyRatio")
            else:
                raw_amounts = raw_supply["itemAmounts"]
                if not isinstance(raw_amounts, dict) or not raw_amounts:
                    raise ValueError("itemAmounts must be a non-empty mapping.")
                for item_name, amount in raw_amounts.items():
                    if not isinstance(item_name, str) or not item_name:
                        raise ValueError("itemAmounts keys must be non-empty strings.")
                    item_amounts[item_name] = self._validate_amount(
                        amount, f"itemAmounts[{item_name!r}]"
                    )

            span = (start, end)
            if span in result:
                raise ValueError(f"Duplicate DynamicSupply interval: {span}.")
            result[span] = (ratio, item_amounts)
            normalized_spans.append((normalized_start, normalized_end))

        for index, (start, end) in enumerate(normalized_spans):
            for other_start, other_end in normalized_spans[index + 1 :]:
                if self._spans_overlap(start, end, other_start, other_end):
                    raise ValueError("DynamicSupply intervals must not overlap.")
        return result

    @staticmethod
    def _is_ordered_span(start: int | datetime, end: int | datetime) -> bool:
        """Check that two like-typed boundaries form an increasing span.

        Args:
            start: Normalized inclusive start boundary.
            end: Normalized exclusive end boundary.

        Returns:
            Whether both boundaries have the same type and ``start < end``.

        Note:
            Explicit type narrowing keeps integer and datetime schedules separate.
        """
        if isinstance(start, int) and isinstance(end, int):
            return start < end
        if isinstance(start, datetime) and isinstance(end, datetime):
            return start < end
        raise TypeError("DynamicSupply start and end must use the same type.")

    @staticmethod
    def _spans_overlap(
        start: int | datetime,
        end: int | datetime,
        other_start: int | datetime,
        other_end: int | datetime,
    ) -> bool:
        """Check whether two normalized half-open spans overlap.

        Args:
            start: First span's inclusive boundary.
            end: First span's exclusive boundary.
            other_start: Second span's inclusive boundary.
            other_end: Second span's exclusive boundary.

        Returns:
            Whether the two spans overlap.

        Note:
            Mixing integer and datetime schedules is rejected.
        """
        if all(
            isinstance(value, int) for value in (start, end, other_start, other_end)
        ):
            return start < other_end and other_start < end  # type: ignore[operator]
        if all(
            isinstance(value, datetime)
            for value in (start, end, other_start, other_end)
        ):
            return start < other_end and other_start < end  # type: ignore[operator]
        raise TypeError("All DynamicSupply intervals must use the same time type.")

    @staticmethod
    def _validate_amount(value: Any, field_name: str) -> float:
        """Validate a nonnegative finite supply value.

        Args:
            value: Candidate numeric value.
            field_name: Configuration name used in an error message.

        Returns:
            Validated floating-point value.

        Note:
            Zero is accepted so a schedule may explicitly suspend supply.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_name} must be numeric.")
        result = float(value)
        if not isfinite(result) or result < 0:
            raise ValueError(f"{field_name} must be finite and nonnegative.")
        return result

    def execute(self, env: Environment[Any], log: Optional[Log] = None) -> None:
        """Capture initial inventory or apply the currently selected supply.

        Args:
            env: Environment whose agent inventories are updated.
            log: Agent-generation log during initialization, otherwise ``None``.

        Returns:
            None.

        Note:
            A time not covered by ``supplies`` performs no inventory mutation.
        """
        if log is not None:
            self._capture_initial_supply(env=env, log=log)
            return

        supply = get_corresponding_value(
            current_time=env.get_time(),
            time_span2value=self.time_span2supply,
            default_value=(None, {}),
        )
        if supply == (None, {}):
            return
        ratio, item_amounts = supply
        for agent_id, initial_supply in self.agent_id2initial_supply.items():
            agent: Agent[Any] = env.agent_id2agent[agent_id]
            additions = (
                {name: amount * ratio for name, amount in initial_supply.items()}
                if ratio is not None
                else item_amounts
            )
            for item_name, amount in additions.items():
                if item_name not in env.item_name2item:
                    raise ValueError(
                        f"DynamicSupply item {item_name!r} is not in the environment."
                    )
                agent.inventory_dic[item_name] = (
                    agent.inventory_dic.get(item_name, 0.0) + amount
                )

    def _capture_initial_supply(self, env: Environment[Any], log: Log) -> None:
        """Record non-cash initial inventory for one configured agent.

        Args:
            env: Environment providing the configured cash-item name.
            log: Log expected to be an :class:`AgentGenerationLog`.

        Returns:
            None.

        Note:
            Non-target agents are ignored. Unexpected log types are rejected to
            expose a misconfigured ``with`` trigger early.
        """
        if not isinstance(log, AgentGenerationLog):
            raise ValueError(
                f"DynamicSupply expected AgentGenerationLog, got {type(log).__name__}."
            )
        if not any(name in log.agent_name for name in self.supplied_agent_names):
            return
        self.agent_id2initial_supply[log.agent_id] = {
            item_name: amount
            for item_name, amount in log.inventory_dic.items()
            if item_name != env.cash_name
        }
