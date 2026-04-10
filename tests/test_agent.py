import copy
import random
from typing import Any

import pytest

from econsimulacra.agents import Agent


class DummyAgent(Agent):
    def act(self, obs):
        pass


class DummyAgentwName(Agent):
    def self_assign_name(self, config) -> None:
        self.agent_name = f"DummyAgent{self.agent_id}"

    def act(self, obs):
        pass


class TestAgent:
    agent_config: dict[str, Any] = {
        "inventory": {
            "cash": 100,
            "rice": 50,
        }
    }

    def test__init__(self) -> None:
        agent = DummyAgent(
            agent_id=1,
            agent_name="TestAgent",
            env_service_dic={},
            config=self.agent_config,
        )
        assert agent.agent_id == 1
        assert agent.agent_type == "DummyAgent"
        assert agent.agent_name == "TestAgent"
        assert not agent.is_rich_info_allowed
        assert agent.inventory_dic == {"cash": 100, "rice": 50}
        agent = DummyAgentwName(
            agent_id=2,
            agent_name="IgnoredName",
            env_service_dic={},
            config=self.agent_config,
        )
        assert agent.agent_id == 2
        assert agent.agent_type == "DummyAgentwName"
        assert agent.agent_name == "DummyAgent2"
        random_config = copy.deepcopy(self.agent_config)
        random_config["inventory"]["cash"] = [50, 100]
        random_config["inventory"]["rice"] = [30, 70]
        random_config["isRichInfoAllowed"] = True
        agent = DummyAgent(
            agent_id=3,
            agent_name="TestAgent3",
            env_service_dic={},
            prng=random.Random(42),
            config=random_config,
        )
        assert agent.agent_id == 3
        assert agent.agent_type == "DummyAgent"
        assert agent.agent_name == "TestAgent3"
        assert agent.is_rich_info_allowed
        cash_amount = agent.inventory_dic["cash"]
        rice_amount = agent.inventory_dic["rice"]
        assert 50 <= cash_amount <= 100
        assert 30 <= rice_amount <= 70
        random_config = copy.deepcopy(self.agent_config)
        random_config["inventory"]["cash"] = [101, 150]
        random_config["inventory"]["rice"] = [71, 110]
        agent = DummyAgent(
            agent_id=4,
            agent_name="TestAgent3",
            env_service_dic={},
            prng=random.Random(42),
            config=random_config,
        )
        assert agent.agent_id == 4
        assert agent.agent_name == "TestAgent3"
        cash_amount = agent.inventory_dic["cash"]
        rice_amount = agent.inventory_dic["rice"]
        assert 101 <= cash_amount <= 150
        assert 71 <= rice_amount <= 110

    def test_exchange_goods(self) -> None:
        agent = DummyAgent(
            agent_id=1,
            agent_name="TestAgent",
            env_service_dic={},
            config=self.agent_config,
        )
        agent.exchange_goods(
            get_item_name="cash",
            get_item_amount=20,
        )
        assert agent.inventory_dic == {"cash": 120, "rice": 50}
        agent.exchange_goods(
            give_item_name="cash",
            give_item_amount=10,
        )
        assert agent.inventory_dic == {"cash": 110, "rice": 50}
        agent.exchange_goods(
            get_item_name="rice",
            get_item_amount=5,
            give_item_name="cash",
            give_item_amount=15,
        )
        assert agent.inventory_dic == {"cash": 95, "rice": 55}
        agent.exchange_goods(
            get_item_name="gold",
            get_item_amount=10,
        )
        assert agent.inventory_dic == {"cash": 95, "rice": 55, "gold": 10}
        with pytest.raises(ValueError):
            agent.exchange_goods(
                give_item_name="bread",
                give_item_amount=10,
            )
        with pytest.raises(ValueError):
            agent.exchange_goods(
                get_item_name="cash",
            )
        with pytest.raises(ValueError):
            agent.exchange_goods(
                give_item_name="rice",
            )
