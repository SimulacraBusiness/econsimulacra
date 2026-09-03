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


def test_mobility_model_defaults_to_walking_without_mobility_observation() -> None:
    model = MobilityModel()

    assert model.select_mobility(_context()) == "Walking"


def test_mobility_model_selects_highest_effective_velocity() -> None:
    model = MobilityModel()
    context = _context(
        available_mobility={
            "Walking": {"velocity": 1, "max_velocity": 1},
            "GasolineCar": {"velocity": 7, "max_velocity": 10},
            "ElectricCar": {"velocity": 10, "max_velocity": 10},
        }
    )

    assert model.select_mobility(context) == "ElectricCar"


def test_mobility_model_keeps_active_available_mobility() -> None:
    model = MobilityModel()
    context = _context(
        movement_state={
            "is_moving": True,
            "destination": (5, 5),
            "mobility_name": "GasolineCar",
        },
        available_mobility={
            "Walking": {"velocity": 1},
            "GasolineCar": {"velocity": 7},
            "ElectricCar": {"velocity": 10},
        },
    )

    assert model.select_mobility(context) == "GasolineCar"


def test_mobility_model_generates_move_with_selected_mobility() -> None:
    model = MobilityModel()
    state = HouseholdState(
        sleep_pressure=0.4,
        hunger=0.2,
        last_meal_elapsed=1.0,
        has_been_sleeping=True,
    )
    context = _context(
        available_mobility={
            "Walking": {"velocity": 1},
            "ElectricCar": {"velocity": 10},
        }
    )

    action = model.generate_move_action(
        context,
        state,
        destination=(4, 2),
        mode="TRAVEL_STORE",
    )

    assert action == {"move": (4, 2), "mobility": "ElectricCar"}
    assert state.destination == (4, 2)
    assert state.mode == "TRAVEL_STORE"
    assert not state.has_been_sleeping


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
                "item_amount": 2,
                "ttl": 2,
            },
        )
    }


def test_household_can_purchase_mobility_items_without_treating_them_as_food() -> None:
    household = RuleBasedHousehold(
        agent_id=0,
        agent_name="Household0",
        env_service_dic={},
        config={
            "foodItems": ("Rice",),
            "shoppingItems": ("Rice", "GasolineCar", "Gasoline"),
            "sSinventoryRule": {
                "reorderPoints": {
                    "Rice": 1.0,
                    "GasolineCar": 0.49,
                    "Gasoline": 5.0,
                },
                "targetStocks": {
                    "Rice": 4.0,
                    "GasolineCar": 1.5,
                    "Gasoline": 20.0,
                },
            },
            "budgetRule": {"maxBasketShare": 1.0},
            "pricePriors": {
                "Rice": 500.0,
                "GasolineCar": 8000.0,
                "Gasoline": 180.0,
            },
        },
    )
    shopping = household.decision_policy.shopping
    context = _context(
        inventory={
            "Yen": 100000.0,
            "Rice": 2.0,
            "GasolineCar": 0.2,
            "Gasoline": 10.0,
        },
        others_inventory=(
            {
                "agent_id": 9,
                "is_household": False,
                "GasolineCar": {"price": 8000.0, "amount": 20.0},
            },
        ),
    )

    action = shopping.generate_order_action(context)

    assert household.decision_policy.physiology.food_items == ("Rice",)
    assert shopping.shopping_items == ("Rice", "GasolineCar", "Gasoline")
    assert action == {
        "orders": (
            {
                "counterparty_id": 9,
                "item_name": "GasolineCar",
                "item_amount": 1,
                "ttl": 2,
            },
        )
    }


def test_store_choice_favors_availability_of_important_shortages() -> None:
    config = {
        "sSinventoryRule": {
            "targetStocks": {"Rice": 1.0, "Chocolate": 1.0},
        },
        "pricePriors": {"Rice": 1.0, "Chocolate": 1.0},
        "itemImportance": {"Rice": 10.0, "Chocolate": 1.0},
        "storeChoice": {
            "betaPrice": 0.0,
            "betaAvailability": 20.0,
            "betaDistance": 0.0,
        },
    }
    model = ShoppingModel(config, ("Rice", "Chocolate"), "Yen", Random(42))
    stores = [
        {"agent_id": 1, "agent_name": "StapleMarket", "pos": (1, 0)},
        {"agent_id": 2, "agent_name": "TreatMarket", "pos": (1, 0)},
    ]
    model._ensure_store_beliefs(1)
    model._ensure_store_beliefs(2)
    model.expected_availability[1] = {"Rice": 1.0, "Chocolate": 0.0}
    model.expected_availability[2] = {"Rice": 0.0, "Chocolate": 1.0}

    chosen = model.choose_store(
        (0, 0), stores, {"Yen": 100.0, "Rice": 0.0, "Chocolate": 0.0}
    )

    assert chosen == stores[0]


def test_store_choice_favors_lower_expected_price_at_equal_coverage() -> None:
    config = {
        "sSinventoryRule": {"targetStocks": {"Rice": 1.0}},
        "pricePriors": {"Rice": 2.0},
        "storeChoice": {
            "betaPrice": 20.0,
            "betaAvailability": 1.0,
            "betaDistance": 0.0,
        },
    }
    model = ShoppingModel(config, ("Rice",), "Yen", Random(42))
    stores = [
        {"agent_id": 1, "agent_name": "CheapMarket", "pos": (1, 0)},
        {"agent_id": 2, "agent_name": "ExpensiveMarket", "pos": (1, 0)},
    ]
    for seller_id in (1, 2):
        model._ensure_store_beliefs(seller_id)
        model.expected_availability[seller_id]["Rice"] = 1.0
    model.expected_price[1]["Rice"] = 1.0
    model.expected_price[2]["Rice"] = 3.0

    chosen = model.choose_store((0, 0), stores, {"Yen": 100.0, "Rice": 0.0})

    assert chosen == stores[0]


def test_budget_pressure_makes_limited_household_avoid_expensive_store() -> None:
    config = {
        "sSinventoryRule": {
            "targetStocks": {"Rice": 1.0, "Chocolate": 1.0},
        },
        "budgetRule": {"cashReserve": 0.0, "maxBasketShare": 1.0},
        "pricePriors": {"Rice": 10.0, "Chocolate": 100.0},
        "storeChoice": {
            "betaPrice": 0.0,
            "betaAvailability": 10.0,
            "betaBudgetPressure": 2.0,
            "betaDistance": 0.0,
        },
    }
    stores = [
        {"agent_id": 1, "agent_name": "BudgetMarket", "pos": (1, 0)},
        {"agent_id": 2, "agent_name": "PrimeDiner", "pos": (1, 0)},
    ]

    def model_with_beliefs(seed: int) -> ShoppingModel:
        model = ShoppingModel(config, ("Rice", "Chocolate"), "Yen", Random(seed))
        for seller_id in (1, 2):
            model._ensure_store_beliefs(seller_id)
        model.expected_availability[1] = {"Rice": 1.0, "Chocolate": 0.0}
        model.expected_availability[2] = {"Rice": 1.0, "Chocolate": 1.0}
        return model

    limited_choice = model_with_beliefs(42).choose_store(
        (0, 0),
        stores,
        {"Yen": 20.0, "Rice": 0.0, "Chocolate": 0.0},
    )
    wealthy_choice = model_with_beliefs(42).choose_store(
        (0, 0),
        stores,
        {"Yen": 1000.0, "Rice": 0.0, "Chocolate": 0.0},
    )

    assert limited_choice == stores[0]
    assert wealthy_choice == stores[1]


def test_store_beliefs_learn_from_visits_and_remain_household_specific() -> None:
    config = {
        "pricePriors": {"Rice": 2.0, "Chocolate": 5.0},
        "storeChoice": {
            "initialAvailability": 0.2,
            "beliefLearningRate": 0.5,
        },
    }
    first = ShoppingModel(config, ("Rice", "Chocolate"), "Yen", Random(1))
    second = ShoppingModel(config, ("Rice", "Chocolate"), "Yen", Random(2))
    context = _context(
        others_inventory=(
            {
                "agent_id": 7,
                "agent_name": "Market7",
                "is_household": False,
                "Rice": {"price": 4.0, "amount": 3.0},
            },
            {
                "agent_id": 8,
                "agent_name": "Household8",
                "is_household": True,
                "Rice": {"price": 1.0, "amount": 10.0},
            },
        )
    )

    first.update_beliefs(context)

    assert first.expected_price[7]["Rice"] == pytest.approx(3.0)
    assert first.expected_availability[7]["Rice"] == pytest.approx(0.6)
    assert first.expected_availability[7]["Chocolate"] == pytest.approx(0.1)
    assert 8 not in first.expected_price
    assert 7 not in second.expected_price


def test_all_visible_non_household_agents_are_store_candidates() -> None:
    model = ShoppingModel({}, ("Rice",), "Yen", Random(42))
    context = _context(
        others_pos=(
            {
                "agent_id": 1,
                "agent_name": "Market1",
                "is_household": False,
                "pos": (1, 0),
            },
            {
                "agent_id": 2,
                "agent_name": "Restaurant2",
                "is_household": False,
                "pos": (2, 0),
            },
            {
                "agent_id": 3,
                "agent_name": "Household3",
                "is_household": True,
                "pos": (3, 0),
            },
        )
    )

    assert model.get_stores(context) == [
        {
            "agent_id": 1,
            "agent_name": "Market1",
            "is_household": False,
            "pos": (1, 0),
        },
        {
            "agent_id": 2,
            "agent_name": "Restaurant2",
            "is_household": False,
            "pos": (2, 0),
        },
    ]


def test_tight_budget_replenishes_more_important_item_first() -> None:
    config = {
        "sSinventoryRule": {
            "targetStocks": {"Chocolate": 2.0, "Rice": 2.0},
        },
        "budgetRule": {"cashReserve": 0.0, "maxBasketShare": 1.0},
        "itemImportance": {"Chocolate": 1.0, "Rice": 10.0},
    }
    model = ShoppingModel(config, ("Chocolate", "Rice"), "Yen", Random(42))
    context = _context(
        inventory={"Yen": 2.0, "Chocolate": 0.0, "Rice": 0.0},
        others_inventory=(
            {
                "agent_id": 8,
                "is_household": True,
                "Chocolate": {"price": 0.5, "amount": 10.0},
                "Rice": {"price": 0.5, "amount": 10.0},
            },
            {
                "agent_id": 9,
                "is_household": False,
                "Chocolate": {"price": 1.0, "amount": 10.0},
                "Rice": {"price": 1.0, "amount": 10.0},
            },
        ),
    )

    action = model.generate_order_action(context)

    assert action["orders"] == (
        {
            "counterparty_id": 9,
            "item_name": "Rice",
            "item_amount": 2.0,
            "ttl": 2,
        },
    )


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


def test_household_context_uses_environment_step_and_iso_clock() -> None:
    household = RuleBasedHousehold(
        agent_id=0,
        agent_name="Household0",
        env_service_dic={},
        config={"foodItems": ("Rice",), "startHour": 7.0, "stepHours": 1.0},
    )
    household.state.last_step = 3

    context = household._context(
        {
            "time": "2025-03-02 23:30:00",
            "time_step": 40,
            "self_pos": (0, 0),
            "self_inventory": {"Yen": 100.0, "Rice": 2.0},
        }
    )

    assert context.time_step == 40
    assert context.hour == pytest.approx(23.5)


def test_household_context_retains_legacy_time_fallback() -> None:
    household = RuleBasedHousehold(
        agent_id=0,
        agent_name="Household0",
        env_service_dic={},
        config={"foodItems": ("Rice",), "startHour": 7.0, "stepHours": 1.0},
    )

    context = household._context(
        {
            "time": 5,
            "self_pos": (0, 0),
            "self_inventory": {"Yen": 100.0, "Rice": 2.0},
        }
    )

    assert context.time_step == 5
    assert context.hour == pytest.approx(12.0)


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


def test_rule_based_household_uses_elapsed_environment_steps_after_sleep() -> None:
    config = {
        "simulation": {"numSteps": 3, "events": []},
        "environment": {
            "space": "gridSpace",
            "socialNetwork": "socialNetwork",
            "cashName": "Yen",
            "agents": ("Household",),
            "items": ("Yen", "Rice"),
            "service": ("timeTranslator", "sleepManager"),
        },
        "gridSpace": {"type": "GridSpace", "gridSize": (1, 1)},
        "socialNetwork": {
            "type": "SocialNetwork",
            "recSys": {"type": "TwoHopRecommenderSystem"},
        },
        "timeTranslator": {
            "type": "TimeTranslator",
            "numSteps": 3,
            "startDatetime": "2025-01-01 00:00:00",
            "endDatetime": "2025-01-01 03:00:00",
        },
        "sleepManager": {"type": "SleepManager"},
        "Household": {
            "type": "RuleBasedHousehold",
            "isHousehold": True,
            "initialCoords": (0, 0),
            "inventory": {"Yen": 100.0, "Rice": 100.0},
            "foodItems": ("Rice",),
            "socialRule": {"enabled": False},
            "sleepRule": {
                "initialPressure": 1.0,
                "lowerAsymptote": 0.05,
                "tauSleepHours": 1.0,
                "circadianAmplitude": 0.0,
                "onsetThreshold": 0.9,
                "wakeThreshold": 0.2,
                "maxSleepSteps": 4,
            },
        },
        "Yen": {"type": "Item", "initialPrice": 1.0, "weightInBasket": 0},
        "Rice": {"type": "Item", "initialPrice": 1.0, "weightInBasket": 1},
    }
    logger = DictLogger()
    simulator = Simulator(config=config, env_class=Environment, logger=logger)

    asyncio.run(simulator.simulate(seed=42))

    household = simulator.env.agent_id2agent[simulator.env.household_ids[0]]
    assert household.state.last_step == 2
    assert household.state.sleep_pressure == pytest.approx(
        0.05 + (1.0 - 0.05) * math.exp(-2.0)
    )
    assert [log["type"] for log in logger.logs].count("sleep_start") == 1
    assert [log["type"] for log in logger.logs].count("sleep_end") == 1
