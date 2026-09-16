from random import Random
from types import SimpleNamespace
from typing import Any, cast

from econsimulacra.events import EventTrigger, KeepOut
from econsimulacra.spaces import GridSpace


def test_keep_out_closes_and_restores_configured_cells() -> None:
    """Verify scheduled closure and restoration preserve original access."""
    event = KeepOut(
        trigger=EventTrigger(
            config={"every": 1}, registered_classes=[], prng=Random(1)
        ),
        config={
            "positions": [[1, 1], [1, 2]],
            "keepOuts": [{"start": 10, "end": 20}],
        },
    )
    grid = GridSpace(
        config={
            "gridSize": [3, 3],
            "cells": [{"pos": [1, 2], "access": {"traversable": False}}],
        },
        registered_classes=[],
        prng=Random(1),
    )
    current_time = 9
    env = SimpleNamespace(grid_space=grid, get_time=lambda: current_time)

    event.execute(env=cast(Any, env))
    assert grid.get_cell((1, 1)).access.traversable
    assert not grid.get_cell((1, 2)).access.traversable

    current_time = 10
    event.execute(env=cast(Any, env))
    assert not grid.get_cell((1, 1)).access.traversable
    assert not grid.get_cell((1, 2)).access.traversable

    current_time = 20
    event.execute(env=cast(Any, env))
    assert grid.get_cell((1, 1)).access.traversable
    assert not grid.get_cell((1, 2)).access.traversable


def test_agent_inside_keep_out_cell_can_leave_but_cannot_reenter() -> None:
    """Verify destination-based traversal permits evacuation without reentry."""
    event = KeepOut(
        trigger=EventTrigger(
            config={"every": 1}, registered_classes=[], prng=Random(1)
        ),
        config={
            "positions": [[1, 1]],
            "keepOuts": [{"start": 0, "end": 10}],
        },
    )
    grid = GridSpace(config={"gridSize": [3, 3]}, registered_classes=[], prng=Random(1))
    grid.place_agent(agent_id=1, pos=(1, 1))
    env = SimpleNamespace(grid_space=grid, get_time=lambda: 0)
    event.execute(env=cast(Any, env))
    grid.move_agent(agent_id=1, new_pos=(0, 1))
    assert grid.get_pos(1) == (0, 1)
    try:
        grid.move_agent(agent_id=1, new_pos=(1, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Agent unexpectedly reentered a keep-out cell.")
