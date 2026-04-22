import random
from datetime import datetime, timedelta
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
        """
        if "numSteps" not in config:
            raise ValueError(
                "TimeTranslator: 'numSteps' must be specified in the config."
            )
        self.num_steps: int = config["numSteps"]
        if "startDatetime" not in config:
            raise ValueError(
                "TimeTranslator: 'startDatetime' must be specified in the config."
            )
        self.start_datetime: datetime = datetime.strptime(
            config["startDatetime"], "%Y-%m-%d %H:%M:%S"
        )
        if "endDatetime" not in config:
            raise ValueError(
                "TimeTranslator: 'endDatetime' must be specified in the config."
            )
        self.end_datetime: datetime = datetime.strptime(
            config["endDatetime"], "%Y-%m-%d %H:%M:%S"
        )
        if self.end_datetime <= self.start_datetime:
            raise ValueError(
                "TimeTranslator: 'endDatetime' must be after 'startDatetime'."
            )
        self.time_delta: timedelta = (
            self.end_datetime - self.start_datetime
        ) / self.num_steps
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = registered_classes

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
        current_datetime: datetime = self.start_datetime + self.time_delta * step
        return current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    def get_timedelta(self) -> str:
        """Get the time delta for each simulation step.

        Returns:
            str: The time delta for each simulation step in ISO format (e.g., "0:00:01" for 1 second).
        """
        return str(self.time_delta)
