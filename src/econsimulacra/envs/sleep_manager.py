from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Optional, Type

from ..logs import SleepEndLog, SleepStartLog


class SleepManager:
    """Sleep manager class. Usually used as environment service."""

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the SleepManager.
            prng (random.Random, optional): An optional instance of random.Random for reproducibility. If not provided, a new instance will be created.
            registered_classes (list[Type]): A list of classes that are registered for use.
        """
        self.config: dict[str, Any] = config
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = registered_classes
        self.agent_id2is_sleeping: dict[int, bool] = {}
        self.agent_id2since: dict[int, Optional[str | int]] = {}
        self.agent_id2until: dict[int, Optional[str | int]] = {}

    def update_sleep_status(
        self,
        agent_id: int,
        current_time: str | int,
        current_time_step: int,
        sleep_duration: Optional[str | int] = None,
    ) -> Optional[SleepStartLog | SleepEndLog]:
        """Update the sleep status of an agent.

        Args:
            agent_id (int): The ID of the agent.
            current_time (str | int): The current time in ISO format or as an integer timestamp.
            current_time_step (int): The current time step in the environment.
            sleep_duration (str | int, optional): The duration of sleep in timedelta or as an integer timestamp.
                Sleep duration as a string must be end with 'h' or 'm' (e.g., "1h" for 1 hour, "30m" for 30 minutes).
                Sleep duration as an integer is the number of time steps to sleep.
                Ex) If the current time is "2024-01-01T00:00:00Z" and the sleep duration is "1h", the agent will wake up at "2024-01-01T01:00:00Z".
                If not provided, the agent will wake up immediately.

        Note:
            This method updates the sleep status of the agent based on the provided current time and sleep duration.
            If the agent is currently sleeping, it will compare the current time with the wake-up time and update the status accordingly.
            Otherwise, it will set the agent to sleeping status and calculate the wake-up time based on the current time and sleep duration.
        """
        log: Optional[SleepStartLog | SleepEndLog] = None
        if agent_id not in self.agent_id2is_sleeping:
            self.agent_id2is_sleeping[agent_id] = False
            self.agent_id2since[agent_id] = None
            self.agent_id2until[agent_id] = None
        if sleep_duration is not None and self.agent_id2is_sleeping[agent_id]:
            raise ValueError(
                "The agent is already sleeping. Cannot set sleep_duration for a sleeping agent."
            )
        if self.agent_id2is_sleeping[agent_id]:
            until: Optional[str | int] = self.agent_id2until[agent_id]
            since: Optional[str | int] = self.agent_id2since[agent_id]
            if until is None:
                raise ValueError("until is None even though the agent is sleeping.")
            if since is None:
                raise ValueError("since is None even though the agent is sleeping.")
            elif isinstance(until, str):
                if isinstance(current_time, int):
                    raise ValueError(
                        "current_time should be in ISO format when the agent specified a ISO format wake-up time."
                    )
                elif isinstance(current_time, str):
                    until_dt = datetime.fromisoformat(until)
                    current_time_dt = datetime.fromisoformat(current_time)
                    if until_dt <= current_time_dt:
                        self._wakeup_agent(agent_id)
                        log = SleepEndLog(
                            time=current_time,
                            time_step=current_time_step,
                            agent_id=agent_id,
                            since=since,
                        )
            elif isinstance(until, int):
                if until <= current_time_step:
                    self._wakeup_agent(agent_id)
                    log = SleepEndLog(
                        time=current_time,
                        time_step=current_time_step,
                        agent_id=agent_id,
                        since=since,
                    )
            else:
                raise ValueError(f"Unknown type of until: {type(until)}")
        else:
            if sleep_duration is not None:
                self.agent_id2is_sleeping[agent_id] = True
                self.agent_id2since[agent_id] = current_time
                if isinstance(sleep_duration, str):
                    sleep_duration_td = self._resolve_sleep_duration(sleep_duration)
                    if isinstance(current_time, str):
                        current_time_dt = datetime.fromisoformat(current_time)
                        until_dt = current_time_dt + sleep_duration_td
                        until_str: str = until_dt.strftime("%Y-%m-%d %H:%M:%S")
                        self.agent_id2until[agent_id] = until_str
                        log = SleepStartLog(
                            time=current_time,
                            time_step=current_time_step,
                            agent_id=agent_id,
                            until=until_str,
                        )
                    elif isinstance(current_time, int):
                        raise ValueError(
                            "current_time should be in ISO format when the agent specified a ISO format sleep_duration."
                        )
                elif isinstance(sleep_duration, int):
                    until_step: int = current_time_step + sleep_duration
                    self.agent_id2until[agent_id] = until_step
                    log = SleepStartLog(
                        time=current_time,
                        time_step=current_time_step,
                        agent_id=agent_id,
                        until=until_step,
                    )
                else:
                    raise ValueError(
                        f"Unknown type of sleep_duration: {type(sleep_duration)}"
                    )
        return log

    def _resolve_sleep_duration(self, sleep_duration: str) -> timedelta:
        """Resolve the sleep duration from a string format to a timedelta object.

        Args:
            sleep_duration (str): The sleep duration as a string. It must end with 'h' for hours or 'm' for minutes (e.g., "1h" for 1 hour, "30m" for 30 minutes).

        Returns:
            timedelta: The resolved sleep duration as a timedelta object.
        """
        if sleep_duration.endswith("h"):
            hours = float(sleep_duration[:-1])
            return timedelta(hours=hours)
        elif sleep_duration.endswith("m"):
            minutes = float(sleep_duration[:-1])
            return timedelta(minutes=minutes)
        else:
            raise ValueError(
                f"Invalid sleep duration format: {sleep_duration}. It must end with 'h' for hours or 'm' for minutes."
            )

    def get_sleep_status(self, agent_id: int) -> bool:
        """Get the sleep status of an agent.

        Args:
            agent_id (int): The ID of the agent.

        Returns:
            bool: The sleep status of the agent. True if the agent is sleeping, False otherwise.
        """
        return self.agent_id2is_sleeping.get(agent_id, False)

    def _wakeup_agent(self, agent_id: int) -> None:
        """Wake up the agent.

        Args:
            agent_id (int): The ID of the agent to wake up.

        Note:
            This method sets the sleep status of the specified agent to False and resets the since and until times to None.
        """
        if (
            agent_id not in self.agent_id2is_sleeping
            or self.agent_id2is_sleeping[agent_id] is False
        ):
            raise ValueError("Sleeping agent not found in agent_id2is_sleeping.")
        self.agent_id2is_sleeping[agent_id] = False
        if agent_id not in self.agent_id2since or self.agent_id2since[agent_id] is None:
            raise ValueError("Sleeping agent not found in agent_id2since.")
        self.agent_id2since[agent_id] = None
        if agent_id not in self.agent_id2until or self.agent_id2until[agent_id] is None:
            raise ValueError("Sleeping agent not found in agent_id2until.")
        self.agent_id2until[agent_id] = None
