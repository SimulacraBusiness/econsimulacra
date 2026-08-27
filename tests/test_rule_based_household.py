import asyncio
import math
from random import Random
from typing import Any

import pytest

from econsimulacra.agents import Agent
from econsimulacra.agents.households import (
    ActionCapabilities,
    DecisionContext,
    HouseholdDecisionPolicy,
    HouseholdState,
    MobilityModel,
    PhysiologyModel,
    RuleBasedHousehold,
    ShoppingModel,
)
from econsimulacra.envs import Environment
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import Simulator


def _context(
    *,
    hour: float = 12.0,
    pos: tuple[int, int] = (0, 0),
    inventory: dict[str, float] | None = None,
    **obs: Any,
) -> DecisionContext:
    observation = {"self_is_sleeping": False, **obs}
    return DecisionContext(
        obs=observation,
        time_step=0,
        hour=hour,
        current_pos=pos,
        inventory=inventory or {"Yen": 100.0, "Rice": 2.0},
    )


def _policy(
    config: dict[str, Any] | None = None,
    capabilities: ActionCapabilities | None = None,
) -> HouseholdDecisionPolicy:
    household_config = config or {}
    return HouseholdDecisionPolicy(
        physiology=PhysiologyModel(household_config, ("Rice",), 1.0),
        mobility=MobilityModel(),
        shopping=ShoppingModel(household_config, ("Rice",), "Yen", Random(42)),
        capabilities=capabilities,
    )


def test_physiology_updates_awake_and_asleep_stocks() -> None:
    model = PhysiologyModel({}, ("Rice",), step_hours=1.0)
    state = model.initialize_state()

    model.update_state(state, 0)
    model.update_state(state, 2)

    expected_awake_pressure = 1.0 - (1.0 - 0.35) * math.exp(-2.0 / 18.0)
    assert state.sleep_pressure == pytest.approx(expected_awake_pressure)
    assert state.hunger == pytest.approx(0.25 + 2 * 0.07)

    state.has_been_sleeping = True
    model.update_state(state, 4)

    expected_asleep_pressure = 0.05 + (expected_awake_pressure - 0.05) * math.exp(
        -2.0 / 4.0
    )
    assert state.sleep_pressure == pytest.approx(expected_asleep_pressure)
    assert state.hunger == pytest.approx(0.25 + 2 * 0.07 + 2 * 0.025)


def test_meal_is_inventory_bounded_and_resets_meal_state() -> None:
    config = {
        "mealRule": {
            "composition": {"Rice": 1.0},
            "energyPerUnit": {"Rice": 2.0},
        }
    }
    model = PhysiologyModel(config, ("Rice",), step_hours=1.0)
    state = HouseholdState(
        sleep_pressure=0.4,
        hunger=0.8,
        last_meal_elapsed=6.0,
        destination=(3, 3),
        mode="TRAVEL_STORE",
    )

    action = model.generate_eat_action(state, {"Rice": 0.25})

    assert action == {"consumptions": ({"item_name": "Rice", "item_amount": 0.25},)}
    assert state.hunger == pytest.approx(0.3)
    assert state.last_meal_elapsed == 0.0
    assert state.destination is None
    assert state.mode == "HOME"


def test_shopping_basket_obeys_target_stock_seller_stock_and_budget() -> None:
    config = {
        "sSinventoryRule": {
            "reorderPoints": {"Rice": 1.0},
            "targetStocks": {"Rice": 8.0},
        },
        "budgetRule": {"cashReserve": 4.0, "maxBasketShare": 0.5},
    }
    model = ShoppingModel(config, ("Rice",), "Yen", Random(42))
    context = _context(
        inventory={"Yen": 10.0, "Rice": 1.0},
        others_inventory=(
            {
                "agent_id": 9,
                "Rice": {"price": 2.0, "amount": 10.0},
            },
        ),
    )

    assert model.should_shop(context.inventory)
    assert model.get_budget(context.inventory) == 5.0
    assert model.generate_order_action(context) == {
        "orders": (
            {
                "counterparty_id": 9,
                "item_name": "Rice",
                "item_amount": 2.5,
                "ttl": 2,
            },
        )
    }


def test_disabled_high_priority_action_does_not_block_enabled_meal() -> None:
    config = {
        "sleepRule": {"initialPressure": 1.0},
        "mealRule": {
            "initialHunger": 0.8,
            "composition": {"Rice": 1.0},
        },
    }
    policy = _policy(
        config,
        ActionCapabilities(frozenset({"sleep_duration"})),
    )
    state = policy.physiology.initialize_state()
    state.home = (0, 0)

    action = policy.decide(_context(hour=2.0), state)

    assert "sleep_duration" not in action
    assert action["consumptions"][0]["item_name"] == "Rice"


def test_household_composes_core_action_with_proposal_rejections() -> None:
    household = RuleBasedHousehold(
        agent_id=0,
        agent_name="Household0",
        env_service_dic={},
        prng=Random(42),
        config={
            "inventory": {"Yen": 100.0, "Rice": 2.0},
            "foodItems": ("Rice",),
            "startHour": 8.0,
            "sleepRule": {"onsetThreshold": 2.0},
            "mealRule": {
                "initialHunger": 0.5,
                "mealThreshold": 0.5,
                "composition": {"Rice": 1.0},
            },
        },
    )
    obs = {
        "time": 0,
        "self_pos": (0, 0),
        "self_init_pos": (0, 0),
        "self_inventory": {"Yen": 100.0, "Rice": 2.0},
        "self_is_sleeping": False,
        "incoming_proposals": ({"proposal_id": 17},),
        "others_pos": (),
    }

    action = asyncio.run(household.act(obs))

    assert action["consumptions"] == ({"item_name": "Rice", "item_amount": 0.5},)
    assert action["reactions"] == ({"kind": "proposal", "id": 17, "accept": False},)
    assert household.state.home == (0, 0)
    assert household.state.last_step == 0


def test_action_composition_concatenates_sequences_and_rejects_conflicts() -> None:
    household = RuleBasedHousehold(
        agent_id=0,
        agent_name="Household0",
        env_service_dic={},
        config={"foodItems": ("Rice",)},
    )

    assert household._compose_fragments(
        [
            {"reactions": ({"id": 1},), "tweet": "same"},
            {"reactions": ({"id": 2},), "tweet": "same"},
        ]
    ) == {
        "reactions": ({"id": 1}, {"id": 2}),
        "tweet": "same",
    }
    with pytest.raises(ValueError, match="Conflicting action fragments"):
        household._compose_fragments([{"tweet": "first"}, {"tweet": "second"}])
        

def test_critical_hunger_takes_priority_over_sleep_as_documented() -> None:
    config = {
        "sleepRule": {"initialPressure": 1.0},
        "mealRule": {
            "initialHunger": 0.95,
            "criticalHunger": 0.9,
            "composition": {"Rice": 1.0},
        },
    }
    policy = _policy(config)
    state = policy.physiology.initialize_state()
    state.home = (0, 0)

    action = policy.decide(_context(hour=2.0), state)

    assert "consumptions" in action
    assert "sleep_duration" not in action


class AcceptingMarket(Agent[dict[str, Any]]):
    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {
            "reactions": tuple(
                {
                    "kind": "order",
                    "id": order["order_id"],
                    "accept_amount": order["item_amount"],
                }
                for order in obs["incoming_orders"]
            )
        }


def test_rule_based_household_completes_a_shopping_simulation() -> None:
    config = {
        "simulation": {"numSteps": 7, "parallelBatchSize": 2},
        "environment": {
            "space": "gridSpace",
            "socialNetwork": "socialNetwork",
            "cashName": "Yen",
            "agents": ("Household", "Market"),
            "items": ("Yen", "Rice"),
        },
        "gridSpace": {"type": "GridSpace", "gridSize": (4, 2)},
        "socialNetwork": {
            "type": "SocialNetwork",
            "followCap": 2,
            "recSys": {
                "type": "TwoHopRecommenderSystem",
                "maxRecommendations": 2,
            },
        },
        "Household": {
            "type": "RuleBasedHousehold",
            "isHousehold": True,
            "numAgents": 1,
            "initialCoords": (0, 0),
            "inventory": {"Yen": 100.0, "Rice": 0.0},
            "foodItems": ("Rice",),
            "disabledActions": ("sleep_duration", "consumptions"),
            "sSinventoryRule": {
                "reorderPoints": {"Rice": 1.0},
                "targetStocks": {"Rice": 4.0},
            },
            "budgetRule": {"cashReserve": 0.0, "maxBasketShare": 1.0},
            "pricePriors": {"Rice": 2.0},
            "storeChoice": {"sellerNamePrefixes": ("Market",)},
        },
        "Market": {
            "type": "AcceptingMarket",
            "isHousehold": False,
            "numAgents": 1,
            "initialCoords": (2, 0),
            "inventory": {"Yen": 0.0, "Rice": 20.0},
            "provideInfo4AllAgents": ("self_pos",),
            "provideInfo4CoLocatedAgents": ("inventory",),
        },
        "Yen": {"type": "Item", "initialPrice": 1.0, "weightInBasket": 0},
        "Rice": {"type": "Item", "initialPrice": 2.0, "weightInBasket": 1},
    }
    simulator = Simulator(config=config, env_class=Environment, logger=DictLogger())
    simulator.register_classes([AcceptingMarket])

    asyncio.run(simulator.simulate(seed=42))

    env = simulator.env
    household_id = env.household_ids[0]
    market_id = env.others_ids[0]
    household = env.agent_id2agent[household_id]
    market = env.agent_id2agent[market_id]
    assert household.get_item_amount("Rice") == pytest.approx(4.0)
    assert household.get_item_amount("Yen") == pytest.approx(92.0)
    assert market.get_item_amount("Rice") == pytest.approx(16.0)
    assert market.get_item_amount("Yen") == pytest.approx(8.0)
    assert env.grid_space.get_pos(household_id) == (0, 0)
    assert env.invalid_action_dic == {
        "sleep": 0,
        "move": 0,
        "consumptions": 0,
        "orders": 0,
        "proposals": 0,
        "reactions": 0,
        "set_prices": 0,
        "follow": 0,
        "unfollow": 0,
    }
