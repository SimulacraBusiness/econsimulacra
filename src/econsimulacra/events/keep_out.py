from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from ..date_utils import get_corresponding_value, to_datetime
from ..logs import Log
from .base import Event, EventTrigger

if TYPE_CHECKING:
    from ..envs import Environment


class KeepOut(Event):
    """Make configured grid cells non-traversable during scheduled periods."""

    def __init__(self, trigger: EventTrigger, config: dict[str, Any]) -> None:
        """Initialize a scheduled keep-out intervention.

        Args:
            trigger: Step trigger used to reevaluate the active period.
            config: Event configuration containing ``positions`` and
                ``keepOuts`` entries with ``start`` and ``end``.

        Returns:
            None.

        Note:
            The event records and restores each cell's original traversal flag;
            it never changes the cell's spawnability or custom attributes.
        """
        super().__init__(trigger=trigger, config=config)
        self._validate_trigger(trigger)
        self.positions = self._parse_positions(config)
        self.time_span2active = self._parse_periods(config)
        self.original_traversable: dict[tuple[int, ...], bool] = {}
        self.is_active: Optional[bool] = None

    def _validate_trigger(self, trigger: EventTrigger) -> None:
        """Validate that keep-out state is reevaluated on simulation steps.

        Args:
            trigger: Trigger to validate.

        Returns:
            None.

        Note:
            ``every: 1`` is recommended. State mutations occur only when the
            active period changes, so per-step checks do not rewrite cells.
        """
        if trigger.logs:
            raise ValueError("KeepOut does not support log triggers.")
        if trigger.every is None and trigger.at is None:
            raise ValueError("KeepOut requires 'every' or 'at' in trigger.")
        if trigger.between is not None:
            raise ValueError(
                "KeepOut selects its active period from 'keepOuts'; "
                "do not use 'between' in trigger."
            )

    def _parse_positions(self, config: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
        """Parse grid positions affected by the event.

        Args:
            config: Keep-out event configuration.

        Returns:
            Non-empty tuple of integer coordinate tuples.

        Note:
            Bounds are validated by ``GridSpace.get_cell`` on first execution.
        """
        raw_positions = config.get("positions")
        if not isinstance(raw_positions, (list, tuple)) or not raw_positions:
            raise ValueError("KeepOut requires non-empty positions.")
        positions: list[tuple[int, ...]] = []
        for raw_position in raw_positions:
            if not isinstance(raw_position, (list, tuple)) or not raw_position:
                raise TypeError("Each KeepOut position must be a coordinate array.")
            position = tuple(raw_position)
            if any(
                not isinstance(coordinate, int) or isinstance(coordinate, bool)
                for coordinate in position
            ):
                raise TypeError("KeepOut coordinates must be integers.")
            positions.append(position)
        if len(set(positions)) != len(positions):
            raise ValueError("KeepOut positions must not contain duplicates.")
        return tuple(positions)

    def _parse_periods(
        self, config: dict[str, Any]
    ) -> dict[tuple[int | str, int | str], bool]:
        """Parse and validate keep-out time spans.

        Args:
            config: Keep-out event configuration.

        Returns:
            Time-span mapping accepted by :func:`get_corresponding_value`.

        Note:
            Intervals are half-open and must not overlap.
        """
        raw_periods = config.get("keepOuts")
        if not isinstance(raw_periods, (list, tuple)) or not raw_periods:
            raise ValueError("KeepOut requires non-empty keepOuts.")
        result: dict[tuple[int | str, int | str], bool] = {}
        normalized_spans: list[tuple[int | datetime, int | datetime]] = []
        for raw_period in raw_periods:
            if not isinstance(raw_period, dict):
                raise TypeError("Each KeepOut entry must be a mapping.")
            if "start" not in raw_period or "end" not in raw_period:
                raise ValueError("Each KeepOut entry requires start and end.")
            start = raw_period["start"]
            end = raw_period["end"]
            if not isinstance(start, (int, str)) or isinstance(start, bool):
                raise TypeError("KeepOut start must be an integer or ISO time.")
            if not isinstance(end, (int, str)) or isinstance(end, bool):
                raise TypeError("KeepOut end must be an integer or ISO time.")
            normalized_start = to_datetime(start)
            normalized_end = to_datetime(end)
            if not self._is_ordered_span(normalized_start, normalized_end):
                raise ValueError("KeepOut requires start < end.")
            span = (start, end)
            if span in result:
                raise ValueError(f"Duplicate KeepOut interval: {span}.")
            result[span] = True
            normalized_spans.append((normalized_start, normalized_end))

        for index, (start, end) in enumerate(normalized_spans):
            for other_start, other_end in normalized_spans[index + 1 :]:
                if self._spans_overlap(start, end, other_start, other_end):
                    raise ValueError("KeepOut intervals must not overlap.")
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
            Integer and datetime schedules cannot be mixed.
        """
        if isinstance(start, int) and isinstance(end, int):
            return start < end
        if isinstance(start, datetime) and isinstance(end, datetime):
            return start < end
        raise TypeError("KeepOut start and end must use the same type.")

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
            Whether the spans overlap.

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
        raise TypeError("All KeepOut intervals must use the same time type.")

    def execute(self, env: Environment[Any], log: Optional[Log] = None) -> None:
        """Apply or restore traversal access for the current time.

        Args:
            env: Environment containing the target grid space.
            log: Unused log argument required by the event interface.

        Returns:
            None.

        Note:
            Agents already inside a newly closed cell are not displaced. They
            may leave because access checks apply to the destination cell.
        """
        if log is not None:
            raise ValueError("KeepOut must be triggered by simulation steps.")
        if not self.original_traversable:
            self.original_traversable = {
                position: env.grid_space.get_cell(position).access.traversable
                for position in self.positions
            }
        active = get_corresponding_value(
            current_time=env.get_time(),
            time_span2value=self.time_span2active,
            default_value=False,
        )
        if self.is_active == active:
            return
        for position in self.positions:
            env.grid_space.update_cell_access(
                pos=position,
                traversable=(False if active else self.original_traversable[position]),
            )
        self.is_active = active
