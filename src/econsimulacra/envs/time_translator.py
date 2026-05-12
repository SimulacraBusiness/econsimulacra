from __future__ import annotations

import random
from datetime import datetime, time, timedelta
from typing import Any, Optional, Type


class TimeTranslator:
    """Time Translator class. Usually used as environment service."""

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the TimeTranslator. This must include parameters such as:
                - numSteps: The total number of steps in the simulation (e.g., 1000).
                - startDatetime: The starting datetime of the simulation in ISO format (e.g., "2025-01-01 00:00:00").
                - endDatetime: The ending datetime of the simulation in ISO format (e.g., "2025-12-31 23:59:59").
                and may optionally include:
                - activeTimeRanges: A list of active time ranges within a day, specified as pairs of start and end times in ISO format
                    (e.g., [["09:00:00", "17:00:00"]]).
            prng (random.Random, optional): An optional instance of random.Random for reproducibility. If not provided, a new instance will be created.
            registered_classes (list[Type]): A list of classes that are registered for use.
        """
        if "numSteps" not in config:
            raise ValueError("TimeTranslator: 'numSteps' must be specified.")
        self.num_steps: int = config["numSteps"]

        if "startDatetime" not in config:
            raise ValueError("TimeTranslator: 'startDatetime' must be specified.")
        self.start_datetime: datetime = datetime.strptime(
            config["startDatetime"], "%Y-%m-%d %H:%M:%S"
        )

        if "endDatetime" not in config:
            raise ValueError("TimeTranslator: 'endDatetime' must be specified.")
        self.end_datetime: datetime = datetime.strptime(
            config["endDatetime"], "%Y-%m-%d %H:%M:%S"
        )

        if self.end_datetime <= self.start_datetime:
            raise ValueError("'endDatetime' must be after 'startDatetime'.")

        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = registered_classes

        self.active_time_ranges: list[tuple[time, time]] = (
            self._parse_active_time_ranges(config.get("activeTimeRanges", []))
        )

        if self.active_time_ranges:
            self.step_datetimes: list[datetime] = self._build_active_step_datetimes()
        else:
            self.time_delta: timedelta = (
                self.end_datetime - self.start_datetime
            ) / self.num_steps
            self.step_datetimes = [
                self.start_datetime + self.time_delta * step
                for step in range(self.num_steps + 1)
            ]

        self.min_time_delta: timedelta = self._calc_min_time_delta()

    def _parse_active_time_ranges(
        self,
        raw_ranges: list[list[str]] | list[tuple[str, str]],
    ) -> list[tuple[time, time]]:
        ranges: list[tuple[time, time]] = []

        for start_str, end_str in raw_ranges:
            start = datetime.strptime(start_str, "%H:%M:%S").time()
            end = datetime.strptime(end_str, "%H:%M:%S").time()

            if end <= start:
                raise ValueError(
                    "TimeTranslator: activeTimeRanges crossing midnight "
                    "is not supported yet."
                )

            ranges.append((start, end))

        return sorted(ranges, key=lambda x: x[0])

    def _is_active_datetime(self, dt: datetime) -> bool:
        current_time = dt.time()
        return any(
            start <= current_time < end for start, end in self.active_time_ranges
        )

    def _next_active_datetime(self, dt: datetime) -> datetime:
        current_date = dt.date()
        current_time = dt.time()

        for start, _ in self.active_time_ranges:
            if current_time < start:
                return datetime.combine(current_date, start)

        next_date = current_date + timedelta(days=1)
        return datetime.combine(next_date, self.active_time_ranges[0][0])

    def _get_active_range_end(self, dt: datetime) -> datetime:
        current_time = dt.time()

        for start, end in self.active_time_ranges:
            if start <= current_time < end:
                return datetime.combine(dt.date(), end)

        raise ValueError(f"{dt} is not in an active time range.")

    def _calc_total_active_seconds(self) -> float:
        total = 0.0
        current = self.start_datetime

        while current < self.end_datetime:
            if not self._is_active_datetime(current):
                current = self._next_active_datetime(current)
                continue

            active_end = min(self._get_active_range_end(current), self.end_datetime)
            total += (active_end - current).total_seconds()
            current = active_end

        return total

    def _advance_active_time(self, dt: datetime, delta: timedelta) -> datetime:
        current = dt

        if not self._is_active_datetime(current):
            current = self._next_active_datetime(current)

        remaining_seconds = delta.total_seconds()

        while remaining_seconds > 0:
            active_end = self._get_active_range_end(current)
            seconds_until_end = (active_end - current).total_seconds()

            if remaining_seconds <= seconds_until_end:
                return current + timedelta(seconds=remaining_seconds)

            remaining_seconds -= seconds_until_end
            current = self._next_active_datetime(active_end)

        return current

    def _build_active_step_datetimes(self) -> list[datetime]:
        total_active_seconds = self._calc_total_active_seconds()

        if total_active_seconds <= 0:
            raise ValueError(
                "TimeTranslator: No active time exists between "
                "startDatetime and endDatetime."
            )

        active_delta = timedelta(seconds=total_active_seconds / self.num_steps)

        step_datetimes: list[datetime] = []
        current = self.start_datetime

        if not self._is_active_datetime(current):
            current = self._next_active_datetime(current)

        for _ in range(self.num_steps + 1):
            step_datetimes.append(current)
            current = self._advance_active_time(current, active_delta)

        return step_datetimes

    def _calc_min_time_delta(self) -> timedelta:
        if len(self.step_datetimes) < 2:
            return timedelta(0)

        deltas = [
            self.step_datetimes[i + 1] - self.step_datetimes[i]
            for i in range(len(self.step_datetimes) - 1)
        ]

        return min(deltas)

    def step_to_datetime(self, step: int) -> str:
        """Convert a simulation step to a datetime.

        Args:
            step (int): The simulation step to convert. Must be between 0 and numSteps - 1.

        Returns:
            datetime: The corresponding datetime for the given simulation step.
        """
        if step < -1 or step > self.num_steps:
            raise ValueError(
                f"TimeTranslator: 'step' must be between -1 and {self.num_steps}."
            )

        return self.step_datetimes[step].strftime("%Y-%m-%d %H:%M:%S")

    def get_timedelta(self) -> str:
        """Get the mimimum time delta for each simulation step.

        Returns:
            str: The time delta for each simulation step in ISO format (e.g., "0:00:01" for 1 second).
        """
        return str(self.min_time_delta)
