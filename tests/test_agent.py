from econsimulacra.agents import Agent


class DummyAgent(Agent):
    def _initialize_inventory(self) -> dict[str, float | int]:
        return {"cash": 100, "rice": 50}


class TestAgent:
    def test__init__(self) -> None:
        agent = DummyAgent(agent_id=1, agent_name="TestAgent")
        assert agent.agent_id == 1
        assert agent.agent_name == "TestAgent"
        assert agent.inventory_dic == {"cash": 100, "rice": 50}
