from __future__ import annotations
from ..logs import Log
import random
from ..sim_utils import find_class
from typing import Any
from typing import Optional
from typing import Type
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Environment


class EventManager:
    def __init__(
        self,
        event_names: list[str],
        events_dic: dict[str, dict[str, Any]],
        registered_classes: list[Type],
        prng: Optional[random.Random] = None,
    ):
        """Initialization.

        Args:
            event_names: A list of event names to be generated in the environment.
            events_dic: An optional dictionary containing event details and parameters.
            registered_classes: A list of classes that can be used to generate events.
                See also: econsimulacra.envs.base.Environment.register_classes(class_list: list[Type])
            prng: An optional random number generator for reproducibility.

        Note:
            events_dic example:
            {
                "event_name_1": { # must be one of the event_names
                    "type": ..., # required
                    "trigger": { # config for EventTrigger. If not provided, the event will never be triggered.
                        "at": tuple[int], # optional
                        "every": int, # optional
                        "with": tuple["Log"], # optional
                        "between": tuple[int, int], # optional,
                        "probability": float, # optional
                    },
                    "other_parameters": ...
                },
                "event_name_2": {
                    ...
                },
                ...
            }
        """
        self.prng = prng if prng is not None else random.Random()
        self.events: list[Event] = self._generate_events(
            event_names=event_names,
            events_dic=events_dic,
            registered_classes=registered_classes,
        )

    def _generate_events(
        self,
        event_names: list[str],
        events_dic: dict[str, dict[str, Any]],
        registered_classes: list[Type],
    ) -> list[Event]:
        """Generates event instances based on the provided event names and configurations.

        Args:
            event_names: A list of event names to be generated.
            events_dic: A dictionary containing event details and parameters.
            registered_classes: A list of classes that can be used to generate events.

        Returns:
            A list of generated event instances.
        """
        events: list[Event] = []
        for event_name in event_names:
            if event_name not in events_dic:
                raise ValueError(f"Event '{event_name}' is not defined in events_dic.")
            event_dic: dict[str, Any] = events_dic[event_name]
            if "type" not in event_dic:
                raise ValueError(
                    f"Couldn't find 'type' key for event '{event_name}' in events_dic."
                )
            event_type: str = event_dic["type"]
            event_class: Type[Event] = find_class(event_type, registered_classes)
            event_instance: Event = event_class(
                trigger=EventTrigger(
                    config=event_dic.get("trigger", {}),
                    registered_classes=registered_classes,
                    prng=self.prng,
                ),
                config=event_dic,
            )
            events.append(event_instance)
        return events

    def trigger_events_after_step(self, time_step: int, env: Environment) -> None:
        """Checks and triggers events based on the current time step.

        Args:
            time_step: The current time step in the environment.
            env: The environment instance where the events will be executed.

        Note:
            See also: econsimulacra.envs.base.Environment.step() where this method is called after each step.
        """
        for event in self.events:
            if event.trigger.check_trigger_after_step(time_step):
                event.execute(env=env, log=None)

    def trigger_events_after_log(self, log: Log, env: Environment) -> None:
        """Checks and triggers events based on the provided log.

        Args:
            log: The log instance that may trigger events.
            env: The environment instance where the events will be executed.
        """
        for event in self.events:
            if event.trigger.check_trigger_after_log(log):
                event.execute(env=env, log=log)


class Event:
    """Event class.

    Usage:
    1. Define a new event class that inherits from Event and implements the execute method.
        >>> class MyEvent(Event):
        >>>     def execute(self, env: Environment, log: Optional[Log] = None) -> None:
        >>>         # Implement the event logic here
    2. Write simulation configuration to include the new event.
        >>> {
        >>>     "simulation": {
        >>>         ...
        >>>         "events": ["myEvent"]
        >>>     },
        >>>     "myEvent": {
        >>>         "type": "MyEvent",
        >>>         "trigger": {
        >>>             "at": [1, 5, 10]
        >>>         }
        >>>     }
        >>> }
    3. Register the new event class in the environment.
        >>> env.register_classes([MyEvent])
    4. Run the simulation and observe the event being triggered at the specified time steps.
    """

    def __init__(
        self,
        trigger: EventTrigger,
        config: dict[str, Any],
    ) -> None:
        """Initialization.

        Args:
            trigger: An instance of EventTrigger that defines when the event should be triggered.
            config: A dictionary containing event-specific parameters and configurations.
        """
        self.trigger = trigger
        self.config = config

    def execute(self, env: Environment, log: Optional[Log] = None) -> None:
        pass


class EventTrigger:
    """EventTrigger class that defines the conditions for triggering events."""

    def __init__(
        self,
        config: dict[str, Any],
        registered_classes: list[Type],
        prng: random.Random,
    ) -> None:
        """Initialization.

        Args:
            config: A dictionary containing trigger conditions and parameters.
            registered_classes: A list of classes that can be used to define trigger conditions.
            prng: A random number generator for handling probabilistic triggers.
        """
        self.at: Optional[tuple[int]] = config.get("at")
        self.every: Optional[int] = config.get("every")
        self.logs: list[Type[Log]] = []
        log_strs: Optional[tuple[str]] = config.get("with")
        if log_strs is not None:
            for log_str in log_strs:
                log_class: Type[Log] = find_class(log_str, registered_classes)
                self.logs.append(log_class)
        self.between: Optional[tuple[int, int]] = config.get("between")
        self.probability: Optional[float] = config.get("probability")
        self.prng = prng

    def check_trigger_after_step(self, time_step: int) -> bool:
        """Checks whether the event should be triggered based on the current time step."""
        if self.at is not None and time_step not in self.at:
            return False
        if self.every is not None and time_step % self.every != 0:
            return False
        if self.between is not None and not (
            self.between[0] <= time_step <= self.between[1]
        ):
            return False
        if self.probability is not None and self.prng.random() >= self.probability:
            return False
        return True

    def check_trigger_after_log(self, log: Log) -> bool:
        """Checks whether the event should be triggered based on the provided log."""
        if self.logs is not None and not any(
            isinstance(log, log_class) for log_class in self.logs
        ):
            return False
        if self.probability is not None and self.prng.random() >= self.probability:
            return False
        return True
