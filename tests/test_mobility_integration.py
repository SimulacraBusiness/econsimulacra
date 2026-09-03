"""Integration tests for mobility across Environment, memory, and Simulator."""

import asyncio
from collections import deque
from copy import deepcopy
from typing import Any

import pytest

from econsimulacra.agents import Agent
from econsimulacra.envs import Environment
from econsimulacra.memory import MoveHistoryItem, calc_stress_from_move_history
from econsimulacra.mobility import MobilityManager
from econsimulacra.simulator import Simulator


class CountingHousehold(Agent):
    """Minimal household that records active decisions for simulator tests."""

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Select an electric-car trip on the first active decision.

        Args:
            obs (dict[str, Any]): Current environment observation.

        Returns:
            dict[str, Any]: First trip request or an empty action afterward.

        Note:
            ``act_calls`` allows tests to distinguish automatic movement from a
            model decision without depending on an external LLM.
        """
        self.act_calls = getattr(self, "act_calls", 0) + 1
        if self.act_calls == 1 and "ElectricCar" in obs["available_mobility"]:
            return {"move": (15, 0), "mobility": "ElectricCar"}
        return {}


def make_config(*, with_mobility: bool = True, num_steps: int = 1) -> dict[str, Any]:
    """Build a small deterministic Environment configuration.

    Args:
        with_mobility (bool, optional): Whether to configure an electric car.
        num_steps (int, optional): Number of simulator steps.

    Returns:
        dict[str, Any]: Independent configuration mapping.

    Note:
        Every mobility-related inventory name is registered as a regular Item.
    """
    config: dict[str, Any] = {
        "simulation": {"numSteps": num_steps, "events": []},
        "environment": {
            "space": "gridSpace",
            "socialNetwork": "socialNetwork",
            "cashName": "Yen",
            "agents": ["CountingHousehold"],
            "items": [
                "Yen",
                "ElectricCar",
                "DriverLicense",
                "Electricity",
            ],
            "service": ["memoryHandler"],
        },
        "gridSpace": {"type": "GridSpace", "gridSize": [20, 1]},
        "socialNetwork": {
            "type": "SocialNetwork",
            "recSys": {"type": "TwoHopRecommenderSystem"},
        },
        "CountingHousehold": {
            "isHousehold": True,
            "initialCoords": (0, 0),
            "inventory": {
                "Yen": 100,
                "ElectricCar": 1,
                "DriverLicense": 1,
                "Electricity": 2,
            },
        },
        "Yen": {"type": "Item", "initialPrice": 1},
        "ElectricCar": {"type": "Item", "initialPrice": 100},
        "DriverLicense": {"type": "Item", "initialPrice": 1},
        "Electricity": {"type": "Item", "initialPrice": 1},
        "memoryHandler": {
            "type": "MemoryHandler",
            "memoryLength": 10,
            "memorySummarizer": {"type": "MemorySummarizer"},
        },
    }
    if with_mobility:
        config["environment"]["service"].append("mobilityManager")
        config["mobilityManager"] = {
            "type": "MobilityManager",
            "defaultMode": "Walking",
            "modes": {
                "Walking": {"velocity": 1},
                "ElectricCar": {
                    "itemName": "ElectricCar",
                    "velocity": 10,
                    "requiredItems": {"DriverLicense": 1},
                    "consumptionPerCell": {"Electricity": 0.1},
                },
            },
        }
    return deepcopy(config)


def make_environment(*, with_mobility: bool = True) -> Environment:
    """Create and reset a configured integration-test Environment.

    Args:
        with_mobility (bool, optional): Whether to configure an electric car.

    Returns:
        Environment: Reset environment containing one household.

    Note:
        A fixed seed makes initial state and path selection deterministic.
    """
    env = Environment(make_config(with_mobility=with_mobility))
    env.register_classes([CountingHousehold])
    env.reset(seed=1)
    return env


def test_missing_config_uses_walking_without_changing_legacy_movement() -> None:
    """Test automatic Walking manager and legacy movement behavior.

    Args:
        None.

    Returns:
        None.

    Note:
        Neither a mobility service entry nor action mobility field is supplied.
    """
    env = make_environment(with_mobility=False)
    agent_id = env.household_ids[0]

    assert isinstance(env.get_mobility_manager(), MobilityManager)
    assert env.get_mobility_manager().get_default_mode().name == "Walking"
    env.apply_action_to_env(agent_id, {"move": (3, 0)})

    assert env.grid_space.get_pos(agent_id) == (1, 0)
    assert env.agent_id2is_moving[agent_id]
    assert env.agent_id2destination[agent_id] == (3, 0)
    assert env.agent_id2movement_state[agent_id].mobility_name == "Walking"


def test_observation_defaults_to_canonical_state_and_supports_legacy_opt_in() -> None:
    """Test canonical defaults while retaining explicit legacy observations.

    Args:
        None.

    Returns:
        None.

    Note:
        Legacy fields remain available during migration but are not in ``all``.
    """
    env = make_environment()
    agent_id = env.household_ids[0]

    obs = env.get_observations(agent_id)

    assert obs["movement_state"] == {
        "is_moving": False,
        "destination": None,
        "mobility_name": None,
    }
    assert "self_is_moving" not in obs
    assert "self_destination" not in obs
    assert set(obs["available_mobility"]) == {"Walking", "ElectricCar"}
    assert obs["available_mobility"]["ElectricCar"]["velocity"] == 10
    env.agent_id2agent[agent_id].request_obses = [
        "self_is_moving",
        "self_destination",
    ]
    legacy_obs = env.get_observations(agent_id)
    assert legacy_obs == {"self_is_moving": False, "self_destination": None}


def test_check_move_validates_mobility_from_inventory() -> None:
    """Test that mobility validity is enforced inside ``_check_move``.

    Args:
        None.

    Returns:
        None.

    Note:
        Removing the mobility Item immediately locks the corresponding mode.
    """
    env = make_environment()
    agent_id = env.household_ids[0]

    assert env._check_move(agent_id, (2, 0), "ElectricCar")
    env.agent_id2agent[agent_id].inventory_dic["ElectricCar"] = 0

    assert not env._check_move(agent_id, (2, 0), "ElectricCar")
    assert not env._check_move(agent_id, (2, 0), "UnknownVehicle")
    assert env._check_move(agent_id, (2, 0))


def test_electric_car_uses_velocity_and_consumes_actual_path_length() -> None:
    """Test high-velocity movement, state synchronization, and fuel deduction.

    Args:
        None.

    Returns:
        None.

    Note:
        Required vehicle and license Items remain durable after movement.
    """
    env = make_environment()
    agent_id = env.household_ids[0]

    env.apply_action_to_env(agent_id, {"move": (15, 0), "mobility": "ElectricCar"})

    agent = env.agent_id2agent[agent_id]
    assert env.grid_space.get_pos(agent_id) == (10, 0)
    assert agent.get_item_amount("Electricity") == pytest.approx(1)
    assert agent.get_item_amount("ElectricCar") == 1
    assert agent.get_item_amount("DriverLicense") == 1
    assert env.agent_id2movement_state[agent_id].mobility_name == "ElectricCar"
    assert env.agent_id2is_moving[agent_id]
    assert env.agent_id2destination[agent_id] == (15, 0)
    assert env.should_skip_decision(agent_id)


def test_fuel_exhaustion_stops_trip_before_decision_and_is_remembered() -> None:
    """Test same-step recovery from resource depletion with durable memory.

    Args:
        None.

    Returns:
        None.

    Note:
        The interrupted agent becomes eligible to choose another action immediately.
    """
    env = make_environment()
    agent_id = env.household_ids[0]
    env.apply_action_to_env(agent_id, {"move": (15, 0), "mobility": "ElectricCar"})
    env.agent_id2agent[agent_id].inventory_dic["Electricity"] = 0.05

    env.prepare_agent_decision(agent_id)

    assert env.agent_id2movement_state[agent_id] is None
    assert env.agent_id2is_moving[agent_id] is False
    assert env.agent_id2destination[agent_id] is None
    assert not env.should_skip_decision(agent_id)
    memory = env.get_memory_handler().get_memory(agent_id)
    assert "ElectricCar" in memory["movement_interruption_history"]
    assert "Electricity" in memory["movement_interruption_history"]
    assert memory["consumption_history"] == "You have no consumption history."
    assert "ElectricCar" not in env.get_observations(agent_id)["available_mobility"]


def test_simulator_skips_decision_during_nonwalking_trip() -> None:
    """Test that automatic non-walking continuation avoids an agent call.

    Args:
        None.

    Returns:
        None.

    Note:
        The agent decides at trip start and again after arrival, but not in transit.
    """
    simulator = Simulator(
        config=make_config(with_mobility=True, num_steps=3),
        env_class=Environment,
    )
    simulator.register_classes([CountingHousehold])

    asyncio.run(simulator.simulate(seed=1))

    agent = simulator.env.agent_id2agent[simulator.env.household_ids[0]]
    assert agent.act_calls == 2
    assert simulator.env.grid_space.get_pos(agent.agent_id) == (15, 0)


def test_move_stress_counts_walking_only_and_preserves_legacy_fallback() -> None:
    """Test mobility-aware stress without changing old MoveHistory behavior.

    Args:
        None.

    Returns:
        None.

    Note:
        Recorded moved_cells avoids attributing a preceding car trip to walking.
    """
    history = deque(
        [
            MoveHistoryItem((0, 0), None, (0, 0), None, -1),
            MoveHistoryItem((10, 0), None, (0, 0), 1, 1, "ElectricCar", 10),
            MoveHistoryItem((11, 0), None, (0, 0), 2, 2, "Walking", 1),
        ]
    )

    stress, reason = calc_stress_from_move_history(
        move_history=history,
        current_time_step=2,
        max_stress=100,
        target_distance=1,
        window_size=10,
        time_decay=1,
        tolerance_threshold=0,
        home_comfort=0,
    )

    assert stress == 0
    assert "distance: 10.0" not in reason
