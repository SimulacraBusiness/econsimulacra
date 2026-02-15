import random
from econsimulacra.agents import Agent
from econsimulacra.envs import JsonRandom
import pytest


class DummyAgent(Agent):
    def _initialize_inventory(self, config) -> dict[str, float | int]:
        return {"cash": 100, "rice": 50}

    def act(self, obs):
        pass


class DummyAgentwName(Agent):
    def _initialize_inventory(self, config) -> dict[str, float | int]:
        return {"cash": 100, "rice": 50}

    def self_assign_name(self, config) -> None:
        self.agent_name = f"DummyAgent{self.agent_id}"

    def act(self, obs):
        pass


class DummyAgentwRandomInventory(Agent):
    def _initialize_inventory(self, config) -> dict[str, float | int]:
        json_random = JsonRandom(prng=self.prng)
        cash_amount = json_random.random(json_value=config["cashAmount"])
        rice_amount = json_random.random(json_value=config["riceAmount"])
        return {"cash": cash_amount, "rice": rice_amount}

    def act(self, obs):
        pass


class TestAgent:
    def test__init__(self) -> None:
        agent = DummyAgent(agent_id=1, agent_name="TestAgent")
        assert agent.agent_id == 1
        assert agent.agent_type == "DummyAgent"
        assert agent.agent_name == "TestAgent"
        assert agent.is_rich_info_allowed == False
        assert agent.inventory_dic == {"cash": 100, "rice": 50}
        agent = DummyAgentwName(agent_id=2, agent_name="IgnoredName")
        assert agent.agent_id == 2
        assert agent.agent_type == "DummyAgentwName"
        assert agent.agent_name == "DummyAgent2"
        agent = DummyAgentwRandomInventory(
            agent_id=3,
            agent_name="TestAgent3",
            prng=random.Random(42),
            config={
                "cashAmount": [50, 100],
                "riceAmount": [30,70],
                "isRichInfoAllowed": True,
            },
        )
        assert agent.agent_id == 3
        assert agent.agent_type == "DummyAgentwRandomInventory"
        assert agent.agent_name == "TestAgent3"
        assert agent.is_rich_info_allowed == True
        cash_amount = agent.inventory_dic["cash"]
        rice_amount = agent.inventory_dic["rice"]
        assert 50 <= cash_amount <= 100
        assert 30 <= rice_amount <= 70
        agent = DummyAgentwRandomInventory(
            agent_id=4,
            agent_name="TestAgent3",
            prng=random.Random(42),
            config={
                "cashAmount": [101, 150],
                "riceAmount": [71,110]
            },
        )
        assert agent.agent_id == 4
        assert agent.agent_name == "TestAgent3"
        cash_amount = agent.inventory_dic["cash"]
        rice_amount = agent.inventory_dic["rice"]
        assert 101 <= cash_amount <= 150
        assert 71 <= rice_amount <= 110

    def test_exchange_goods(self) -> None:
        agent = DummyAgent(agent_id=1, agent_name="TestAgent")
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
        with pytest.raises(ValueError):
            agent.exchange_goods(
                get_item_name="gold",
                get_item_amount=10,
            )
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
