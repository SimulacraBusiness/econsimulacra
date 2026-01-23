from econsimulacra.agents import Agent
import pytest


class DummyAgent(Agent):
    def _initialize_inventory(self) -> dict[str, float | int]:
        return {"cash": 100, "rice": 50}

    def act(self, obs):
        pass


class TestAgent:
    def test__init__(self) -> None:
        agent = DummyAgent(agent_id=1, agent_name="TestAgent")
        assert agent.agent_id == 1
        assert agent.agent_name == "TestAgent"
        assert agent.inventory_dic == {"cash": 100, "rice": 50}

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
