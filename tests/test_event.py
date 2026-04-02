from econsimulacra.envs import Environment
from econsimulacra.envs import Event
from econsimulacra.envs import EventTrigger
from econsimulacra.envs import EventManager
import random
from typing import Any


class DummyEvent(Event):
    def __init__(
        self,
        trigger: EventTrigger,
        config: dict[str, Any],
    ) -> None:
        super().__init__(trigger=trigger, config=config)
        self.num_executions = 0

    def execute(self, env, log=None):
        self.num_executions += 1


class TestEvents:
    event_names: list[str] = [
        "DummyEvent1",
        "DummyEvent2",
        "DummyEvent3",
        "DummyEvent4",
        "DummyEvent5",
    ]
    event_configs: dict[str, dict[str, Any]] = {
        "DummyEvent1": {
            "trigger": {
                "at": (1, 3, 5),
            },
            "type": "DummyEvent",
        },
        "DummyEvent2": {
            "trigger": {
                "every": 2,
            },
            "type": "DummyEvent",
        },
        "DummyEvent3": {
            "trigger": {
                "between": (2, 4),
            },
            "type": "DummyEvent",
        },
        "DummyEvent4": {
            "trigger": {
                "probability": 0.0,
                "at": (1, 3, 5)
            },
            "type": "DummyEvent",
        },
        "DummyEvent5": {
            "trigger": {
                "probability": 1.0,
                "at": (1, 3, 5)
            },
            "type": "DummyEvent",
        },
    }

    def test_event_trigger(self):
        prng = random.Random(0)
        event_manager = EventManager(
            event_names=self.event_names,
            events_dic=self.event_configs,
            registered_classes=[DummyEvent],
            prng=prng,
        )
        env = Environment(config={"environment": {"cashName": "Cash"}})
        for time_step in range(1, 7):
            event_manager.trigger_events_after_step(time_step, env)
        assert event_manager.events[0].num_executions == 3
        assert event_manager.events[1].num_executions == 3
        assert event_manager.events[2].num_executions == 3
        assert event_manager.events[3].num_executions == 0
        assert event_manager.events[4].num_executions == 3
