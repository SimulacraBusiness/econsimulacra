from random import Random
from types import SimpleNamespace
from typing import Any, cast

import pytest

from econsimulacra.events import DynamicSupply, EventTrigger
from econsimulacra.logs import AgentGenerationLog


def _trigger(config: dict | None = None) -> EventTrigger:
    """Build a valid dynamic-supply trigger for unit tests.

    Args:
        config: Optional trigger configuration override.

    Returns:
        Configured event trigger.

    Note:
        The helper uses the package's real class lookup for AgentGenerationLog.
    """
    return EventTrigger(
        config=config or {"with": ["AgentGenerationLog"], "every": 24},
        registered_classes=[],
        prng=Random(1),
    )


def _generation_log(agent_id: int = 1, name: str = "Daily Mart1") -> AgentGenerationLog:
    """Build an agent-generation log with deterministic inventory.

    Args:
        agent_id: Generated agent identifier.
        name: Generated agent name.

    Returns:
        Agent-generation log used to seed DynamicSupply.

    Note:
        Cash is included to verify that it is excluded from replenishment.
    """
    return AgentGenerationLog(
        time="2025-03-01 07:00:00",
        time_step=0,
        agent_id=agent_id,
        agent_type="RuleBasedRetailer",
        agent_name=name,
        wealth=100.0,
        inventory_dic={"Yen": 10.0, "Rice": 20.0},
        persona_dic=None,
    )


def test_dynamic_supply_selects_ratio_by_display_time() -> None:
    """Verify the half-open time transition between two supply ratios."""
    event = DynamicSupply(
        trigger=_trigger(),
        config={
            "suppliedAgentNames": ["Daily Mart"],
            "supplies": [
                {
                    "start": "2025-03-01 07:00:00",
                    "end": "2025-03-07 07:00:00",
                    "supplyRatio": 0.5,
                },
                {
                    "start": "2025-03-07 07:00:00",
                    "end": "2025-03-31 07:00:00",
                    "supplyRatio": 0.25,
                },
            ],
        },
    )
    agent = SimpleNamespace(inventory_dic={"Yen": 10.0, "Rice": 20.0})
    env = SimpleNamespace(
        cash_name="Yen",
        agent_id2agent={1: agent},
        item_name2item={"Yen": object(), "Rice": object()},
        get_time=lambda: "2025-03-06 07:00:00",
    )
    event.execute(env=cast(Any, env), log=_generation_log())
    event.execute(env=cast(Any, env))
    assert agent.inventory_dic == {"Yen": 10.0, "Rice": 30.0}

    env.get_time = lambda: "2025-03-07 07:00:00"
    event.execute(env=cast(Any, env))
    assert agent.inventory_dic == {"Yen": 10.0, "Rice": 35.0}


def test_dynamic_supply_can_launch_an_initially_absent_item() -> None:
    """Verify absolute supply independently of captured initial inventory."""
    event = DynamicSupply(
        trigger=_trigger({"with": ["AgentGenerationLog"], "at": [144]}),
        config={
            "suppliedAgentNames": ["Car Dealer"],
            "supplies": [
                {
                    "start": 144,
                    "end": 720,
                    "itemAmounts": {"HighPerformanceCar": 2},
                }
            ],
        },
    )
    agent = SimpleNamespace(inventory_dic={"Yen": 0.0, "GasolineCar": 2.0})
    env = SimpleNamespace(
        cash_name="Yen",
        agent_id2agent={1: agent},
        item_name2item={"HighPerformanceCar": object()},
        get_time=lambda: 144,
    )
    event.execute(env=cast(Any, env), log=_generation_log(name="Car Dealer1"))
    event.execute(env=cast(Any, env))
    assert agent.inventory_dic["HighPerformanceCar"] == 2.0


def test_dynamic_supply_rejects_overlapping_periods() -> None:
    """Verify ambiguous supply schedules fail during construction."""
    with pytest.raises(ValueError, match="must not overlap"):
        DynamicSupply(
            trigger=_trigger(),
            config={
                "suppliedAgentNames": ["Daily Mart"],
                "supplies": [
                    {"start": 0, "end": 10, "supplyRatio": 0.5},
                    {"start": 9, "end": 20, "supplyRatio": 0.25},
                ],
            },
        )
