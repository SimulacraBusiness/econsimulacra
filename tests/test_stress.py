"""Tests for the stress ObsProviders (REQ-ENVIRONMENT-010).

Covers:
- Item.stress_effects attribute initialization from config.
- Environment stress state initialization via _init_stress_state().
- FinancialStressProvider, SocialStressProvider, LifeStressProvider, PhysicalStressProvider.
- Hunger (steps since last consumption) and fatigue (consecutive moves) tracking.
- add_disease_stress() method.
- StateEvaluationLog and StateEvaluationItem include stress fields.
- MemoryHandler stores and formats stress in state_evaluation_history.
- _apply_item_stress_effects reduces stress components defined in stressEffects config.
"""

from typing import Any, Optional

from econsimulacra.agents import Agent
from econsimulacra.envs import (
    Environment,
    FinancialStressProvider,
    LifeStressProvider,
    MemoryHandler,
    PhysicalStressProvider,
    SocialStressProvider,
)
from econsimulacra.items import Item
from econsimulacra.logs import StateEvaluationLog


# ---------------------------------------------------------------------------
# Minimal test agent / dummy classes
# ---------------------------------------------------------------------------


class SimpleHousehold(Agent):
    """A minimal household agent that does nothing each step."""

    def _initialize_inventory(self, config) -> dict[str, float | int]:
        return {"Yen": 10000, "Rice": 50, "Bread": 20}

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {}


class SimpleRetailer(Agent):
    """A minimal retailer agent that does nothing each step."""

    def _initialize_inventory(self, config) -> dict[str, float | int]:
        return {"Yen": 100000, "Rice": 1000, "Bread": 500}

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {}


class SimpleMemoryHandler(MemoryHandler):
    """A memory handler that always returns an empty dict (avoids LLM deps)."""

    def get_memory(self, agent_id: int) -> Optional[dict[str, Any]]:
        return {}


# ---------------------------------------------------------------------------
# Environment config used throughout
# ---------------------------------------------------------------------------

BASE_CONFIG: dict[str, Any] = {
    "simulation": {"numSteps": 20},
    "environment": {
        "space": "gridSpace",
        "socialNetwork": "socialNetwork",
        "cashName": "Yen",
        "agents": ["SimpleHousehold", "SimpleRetailer"],
        "items": ["Yen", "Rice", "Bread"],
        "service": ["memoryHandler"],
    },
    "gridSpace": {
        "type": "GridSpace",
        "gridSize": [5, 5],
    },
    "socialNetwork": {
        "type": "SocialNetwork",
        "followCap": 5,
        "recSys": {
            "type": "TwoHopRecommenderSystem",
            "maxRecommendations": 2,
        },
    },
    "SimpleHousehold": {
        "isHousehold": True,
        "numAgents": 2,
        "inventory": {"Yen": 10000, "Rice": 50, "Bread": 20},
    },
    "SimpleRetailer": {
        "isHousehold": False,
        "numAgents": 1,
        "inventory": {"Yen": 100000, "Rice": 1000, "Bread": 500},
    },
    "Yen": {"type": "Item", "initialPrice": 1.0},
    "Rice": {
        "type": "Item",
        "initialPrice": 500.0,
        "stressEffects": {"hunger": 20.0, "life": 5.0},
    },
    "Bread": {
        "type": "Item",
        "initialPrice": 200.0,
        "stressEffects": {"hunger": 10.0, "fatigue": 2.0},
    },
    "memoryHandler": {
        "type": "SimpleMemoryHandler",
        "memoryLength": 5,
    },
}


def make_env() -> Environment:
    env = Environment(config=BASE_CONFIG)
    env.register_classes([SimpleHousehold, SimpleRetailer, SimpleMemoryHandler])
    env.reset(seed=0)
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestItemStressEffects:
    def test_stress_effects_from_config(self) -> None:
        item = Item(item_id=0, item_name="Rice", config={"initialPrice": 500.0, "stressEffects": {"hunger": 20.0, "life": 5.0}})
        assert item.stress_effects == {"hunger": 20.0, "life": 5.0}

    def test_no_stress_effects_by_default(self) -> None:
        item = Item(item_id=1, item_name="Yen", config={"initialPrice": 1.0})
        assert item.stress_effects == {}

    def test_stress_effects_empty_config(self) -> None:
        item = Item(item_id=2, item_name="Gold")
        assert item.stress_effects == {}

    def test_stress_effects_in_env_items(self) -> None:
        env = make_env()
        rice = env.item_name2item["Rice"]
        assert rice.stress_effects.get("hunger") == 20.0
        assert rice.stress_effects.get("life") == 5.0
        bread = env.item_name2item["Bread"]
        assert bread.stress_effects.get("hunger") == 10.0
        assert bread.stress_effects.get("fatigue") == 2.0
        yen = env.item_name2item["Yen"]
        assert yen.stress_effects == {}


class TestStressStateInitialization:
    def test_stress_state_initialized_for_all_agents(self) -> None:
        env = make_env()
        for agent_id in env.agent_ids:
            assert agent_id in env.agent_id2stress
            stress = env.agent_id2stress[agent_id]
            assert "financial" in stress
            assert "social" in stress
            assert "life" in stress
            assert "physical" in stress
            assert "affordance" in stress["financial"]
            assert "relative_financial_status" in stress["financial"]
            assert "reputation" in stress["social"]
            assert "satisfaction" in stress["social"]
            assert "hunger" in stress["physical"]
            assert "fatigue" in stress["physical"]
            assert "disease" in stress["physical"]

    def test_stress_state_starts_at_zero(self) -> None:
        env = make_env()
        for agent_id in env.agent_ids:
            stress = env.agent_id2stress[agent_id]
            assert stress["life"] == 0.0
            assert stress["physical"]["disease"] == 0.0

    def test_consecutive_moves_initialized_to_zero(self) -> None:
        env = make_env()
        for agent_id in env.agent_ids:
            assert env.agent_id2consecutive_moves[agent_id] == 0

    def test_last_consumption_step_initialized(self) -> None:
        env = make_env()
        for agent_id in env.agent_ids:
            assert agent_id in env.agent_id2last_consumption_step


class TestPhysicalStress:
    def test_hunger_increases_with_steps(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        # Step the env without any consumption
        env.step({aid: {} for aid in env.agent_ids})
        env.evaluate_agent_state(household_id)
        hunger = env.agent_id2stress[household_id]["physical"]["hunger"]
        assert hunger >= 1.0

    def test_hunger_resets_on_consumption(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        # Advance a few steps without consumption to build up hunger
        for _ in range(3):
            env.step({aid: {} for aid in env.agent_ids})
        env.evaluate_agent_state(household_id)
        hunger_before = env.agent_id2stress[household_id]["physical"]["hunger"]
        assert hunger_before >= 3.0
        # Now consume an item
        env._consume_items(
            agent_id=household_id,
            consumptions=[{"item_name": "Rice", "item_amount": 1}],
        )
        assert env.agent_id2stress[household_id]["physical"]["hunger"] == 0.0

    def test_consumption_always_resets_hunger(self) -> None:
        """Consuming any item resets hunger to 0, regardless of stressEffects config."""
        env = make_env()
        household_id = env.household_ids[0]
        # Manually set hunger high
        env.agent_id2stress[household_id]["physical"]["hunger"] = 50.0
        # Consume Rice (has stressEffects hunger=20.0 which is ignored for hunger reset)
        env._apply_item_stress_effects(agent_id=household_id, item_name="Rice")
        # Hunger is always reset to 0 since it measures steps since last consumption
        assert env.agent_id2stress[household_id]["physical"]["hunger"] == 0.0

    def test_fatigue_increases_with_consecutive_moves(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        initial_moves = env.agent_id2consecutive_moves[household_id]
        # Issue a move action
        action_with_move = {
            "move": (min(env.grid_space.get_pos(household_id)[0] + 1, 4), 0)
        }
        env.apply_action_to_env(agent_id=household_id, action_dic=action_with_move)
        assert env.agent_id2consecutive_moves[household_id] == initial_moves + 1

    def test_fatigue_resets_when_no_move(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        # Force some consecutive moves
        env.agent_id2consecutive_moves[household_id] = 5
        # Apply no-move action
        env.apply_action_to_env(agent_id=household_id, action_dic={})
        assert env.agent_id2consecutive_moves[household_id] == 0

    def test_fatigue_reflected_in_stress(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        env.agent_id2consecutive_moves[household_id] = 7
        env.evaluate_agent_state(household_id)
        fatigue = env.agent_id2stress[household_id]["physical"]["fatigue"]
        assert fatigue == 7.0

    def test_disease_stress_accumulates(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        env.add_disease_stress(agent_id=household_id, amount=10.0)
        env.add_disease_stress(agent_id=household_id, amount=5.0)
        assert env.agent_id2stress[household_id]["physical"]["disease"] == 15.0

    def test_add_disease_stress_negative_raises(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        try:
            env.add_disease_stress(agent_id=household_id, amount=-1.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestFinancialStress:
    def test_financial_stress_keys(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        stress = env.agent_id2stress[household_id]["financial"]
        assert "affordance" in stress
        assert "relative_financial_status" in stress

    def test_affordance_stress_zero_when_rich(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        # Give agent lots of cash so affordance stress should be 0
        env.agent_id2agent[household_id].inventory_dic["Yen"] = 10_000_000
        env.evaluate_agent_state(household_id)
        assert env.agent_id2stress[household_id]["financial"]["affordance"] == 0.0

    def test_relative_financial_status_range(self) -> None:
        env = make_env()
        for agent_id in env.agent_ids:
            env.evaluate_agent_state(agent_id)
            rfs = env.agent_id2stress[agent_id]["financial"]["relative_financial_status"]
            assert 0.0 <= rfs <= 1.0

    def test_financial_stress_provider(self) -> None:
        env = make_env()
        provider = FinancialStressProvider(env=env)
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        obs = provider.get_obs(agent_id=household_id)
        assert isinstance(obs, dict)
        assert "affordance" in obs
        assert "relative_financial_status" in obs


class TestSocialStress:
    def test_social_stress_keys(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        stress = env.agent_id2stress[household_id]["social"]
        assert "reputation" in stress
        assert "satisfaction" in stress

    def test_reputation_zero_when_max_followers(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        # Make this agent have as many followers as any agent
        for other_id in env.agent_ids:
            if other_id != household_id:
                env.social_network.follow_agent(
                    agent_id=other_id, target_agent_id=household_id
                )
        env.evaluate_agent_state(household_id)
        assert env.agent_id2stress[household_id]["social"]["reputation"] == 0.0

    def test_satisfaction_zero_when_follows_tweet(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        retailer_id = env.others_ids[0]
        # Follow retailer and make sure they have a tweet
        env.social_network.follow_agent(
            agent_id=household_id, target_agent_id=retailer_id
        )
        env.social_network.tweet(agent_id=retailer_id, message="Hello!")
        env.evaluate_agent_state(household_id)
        # Retailer has a tweet, so satisfaction stress should be 0
        assert env.agent_id2stress[household_id]["social"]["satisfaction"] == 0.0

    def test_satisfaction_high_when_follows_no_tweet(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        retailer_id = env.others_ids[0]
        # Follow retailer with no tweet (empty tweet)
        env.social_network.follow_agent(
            agent_id=household_id, target_agent_id=retailer_id
        )
        env.evaluate_agent_state(household_id)
        # Retailer has no tweet so satisfaction stress should be 1.0
        assert env.agent_id2stress[household_id]["social"]["satisfaction"] == 1.0

    def test_social_stress_provider(self) -> None:
        env = make_env()
        provider = SocialStressProvider(env=env)
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        obs = provider.get_obs(agent_id=household_id)
        assert isinstance(obs, dict)
        assert "reputation" in obs
        assert "satisfaction" in obs


class TestLifeStress:
    def test_life_stress_increases_without_consumption(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        life_before = env.agent_id2stress[household_id]["life"]
        # No consumption means life stress increases
        env.evaluate_agent_state(household_id)
        life_after = env.agent_id2stress[household_id]["life"]
        assert life_after >= life_before

    def test_life_stress_decreases_with_diverse_consumption(self) -> None:
        env = make_env()
        household_id = env.household_ids[0]
        # Pre-set life stress to something measurable
        env.agent_id2stress[household_id]["life"] = 0.5
        # Consume two different items (diverse diet) to reduce life stress
        env.agent_id2recent_item_names[household_id] = ["Rice", "Bread"]
        env.evaluate_agent_state(household_id)
        life_after = env.agent_id2stress[household_id]["life"]
        assert life_after < 0.5

    def test_life_stress_provider(self) -> None:
        env = make_env()
        provider = LifeStressProvider(env=env)
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        obs = provider.get_obs(agent_id=household_id)
        assert isinstance(obs, float)
        assert 0.0 <= obs <= 1.0


class TestPhysicalStressProvider:
    def test_physical_stress_provider(self) -> None:
        env = make_env()
        provider = PhysicalStressProvider(env=env)
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        obs = provider.get_obs(agent_id=household_id)
        assert isinstance(obs, dict)
        assert "hunger" in obs
        assert "fatigue" in obs
        assert "disease" in obs


class TestStressInObservations:
    def test_stress_keys_in_obs(self) -> None:
        env = make_env()
        env.step({aid: {} for aid in env.agent_ids})
        for agent_id in env.agent_ids:
            obs = env.get_observations(agent_id=agent_id)
            assert "financial_stress" in obs
            assert "social_stress" in obs
            assert "life_stress" in obs
            assert "physical_stress" in obs

    def test_stress_obs_types(self) -> None:
        env = make_env()
        env.step({aid: {} for aid in env.agent_ids})
        for agent_id in env.agent_ids:
            obs = env.get_observations(agent_id=agent_id)
            assert isinstance(obs["financial_stress"], dict)
            assert isinstance(obs["social_stress"], dict)
            assert isinstance(obs["life_stress"], float)
            assert isinstance(obs["physical_stress"], dict)


class TestStateEvaluationLogWithStress:
    def test_state_evaluation_log_includes_stress(self) -> None:
        log = StateEvaluationLog(
            time=0,
            time_step=0,
            agent_id=0,
            wealth=1000.0,
            financial_stress={"affordance": 0.3, "relative_financial_status": 0.2},
            social_stress={"reputation": 0.5, "satisfaction": 0.1},
            life_stress=0.4,
            physical_stress={"hunger": 2.0, "fatigue": 1.0, "disease": 0.0},
        )
        assert log.financial_stress == {
            "affordance": 0.3,
            "relative_financial_status": 0.2,
        }
        assert log.social_stress == {"reputation": 0.5, "satisfaction": 0.1}
        assert log.life_stress == 0.4
        assert log.physical_stress == {"hunger": 2.0, "fatigue": 1.0, "disease": 0.0}

    def test_state_evaluation_log_defaults_to_none(self) -> None:
        log = StateEvaluationLog(
            time=0,
            time_step=0,
            agent_id=0,
            wealth=500.0,
        )
        assert log.financial_stress is None
        assert log.social_stress is None
        assert log.life_stress is None
        assert log.physical_stress is None

    def test_state_evaluation_includes_stress_from_env(self) -> None:
        """StateEvaluationLog generated by env.evaluate_agent_state should include stress."""
        from econsimulacra.logs import DictLogger

        env = make_env()
        env.logger = DictLogger()
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        env.logger.process_logs()
        state_eval_logs = [
            log for log in env.logger.logs if log.get("type") == "state_evaluation"
        ]
        assert len(state_eval_logs) >= 1
        log_dict = state_eval_logs[-1]
        assert "financial_stress" in log_dict
        assert "social_stress" in log_dict
        assert "life_stress" in log_dict
        assert "physical_stress" in log_dict


class TestMemoryWithStress:
    def test_memory_includes_stress_in_state_evaluation(self) -> None:
        """MemoryHandler should store stress in StateEvaluationItem."""
        env = Environment(
            config={
                **BASE_CONFIG,
                "memoryHandler": {
                    "type": "MemoryHandler",
                    "memoryLength": 5,
                },
            }
        )
        env.register_classes([SimpleHousehold, SimpleRetailer])
        env.reset(seed=0)
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        memory_handler = env.get_memory_handler()
        assert memory_handler is not None
        agent_memory = memory_handler.agent_id2memory[household_id]
        assert len(agent_memory.state_evaluation_history) >= 1
        item = agent_memory.state_evaluation_history[-1]
        assert item.financial_stress is not None
        assert item.social_stress is not None
        assert item.life_stress is not None
        assert item.physical_stress is not None

    def test_memory_summary_includes_stress_text(self) -> None:
        """get_memory() string should contain stress information."""
        env = Environment(
            config={
                **BASE_CONFIG,
                "memoryHandler": {
                    "type": "MemoryHandler",
                    "memoryLength": 5,
                },
            }
        )
        env.register_classes([SimpleHousehold, SimpleRetailer])
        env.reset(seed=0)
        household_id = env.household_ids[0]
        env.evaluate_agent_state(household_id)
        memory_handler = env.get_memory_handler()
        assert memory_handler is not None
        mem = memory_handler.get_memory(agent_id=household_id)
        assert "state_evaluation_history" in mem
        summary = mem["state_evaluation_history"]
        assert "FinancialStress" in summary
        assert "SocialStress" in summary
        assert "LifeStress" in summary
        assert "PhysicalStress" in summary
