from econsimulacra.agents import Agent
from econsimulacra.envs import GridSpace
from econsimulacra.envs import Environment
import pytest


class TestGridSpace:
    def test___init__(self) -> None:
        grid_space = GridSpace(space_size=(10, 10))
        assert grid_space.space_size == (10, 10)
        assert grid_space.pos2agent_ids == {}
        assert grid_space.agent_id2pos == {}
        grid_space_3d = GridSpace(space_size=(5, 5, 5))
        assert grid_space_3d.space_size == (5, 5, 5)
        grid_space_1d = GridSpace(space_size=(20,))
        assert grid_space_1d.space_size == (20,)
        grid_space_ = GridSpace(space_size=(10, 10))
        assert grid_space == grid_space_

    def test_place_agent(self) -> None:
        grid_space = GridSpace(space_size=(10, 10))
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        assert grid_space.get_pos(agent_id=0) == (2, 3)
        assert grid_space.get_agents(pos=(2, 3)) == {0}
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=0, pos=(4, 5))
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=2, pos=(10, 5))
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=3, pos=(-1, 0))
        grid_space_3d = GridSpace(space_size=(5, 5, 5))
        grid_space_3d.place_agent(agent_id=0, pos=(1, 2, 3))
        assert grid_space_3d.get_pos(agent_id=0) == (1, 2, 3)
        assert grid_space_3d.get_agents(pos=(1, 2, 3)) == {0}
        with pytest.raises(ValueError):
            grid_space_3d.place_agent(agent_id=1, pos=(5, 2))

    def test_remove_agent(self) -> None:
        grid_space = GridSpace(space_size=(10, 10))
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.remove_agent(agent_id=0)
        assert grid_space.pos2agent_ids == {}
        assert grid_space.agent_id2pos == {}
        with pytest.raises(ValueError):
            grid_space.remove_agent(agent_id=0)

    def test_move_agent(self) -> None:
        grid_space = GridSpace(space_size=(10, 10))
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.move_agent(agent_id=0, new_pos=(5, 6))
        assert grid_space.get_pos(agent_id=0) == (5, 6)
        assert grid_space.get_agents(pos=(5, 6)) == {0}
        assert grid_space.get_agents(pos=(2, 3)) == set()
        with pytest.raises(ValueError):
            grid_space.move_agent(agent_id=1, new_pos=(4, 5))
        with pytest.raises(ValueError):
            grid_space.move_agent(agent_id=0, new_pos=(10, 5))

    def test_move_many_agents(self) -> None:
        grid_space = GridSpace(space_size=(10, 10))
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.place_agent(agent_id=1, pos=(4, 5))
        grid_space.move_many_agents(agent_id2new_pos={0: (6, 7), 1: (8, 9)})
        assert grid_space.get_pos(agent_id=0) == (6, 7)
        assert grid_space.get_pos(agent_id=1) == (8, 9)
        assert grid_space.get_agents(pos=(2, 3)) == set()
        assert grid_space.get_agents(pos=(4, 5)) == set()


class DummyHousehold(Agent):
    def _initialize_inventory(self) -> dict[str, float | int]:
        return {"cash": 100, "rice": 50}

    def act(self, obs):
        pass


class TestEnvironment:
    config = {
        "gridSpace": (10, 10),
        "simulation": {
            "numSteps": 100,
            "agents": ["DummyHousehold"],
        },
        "DummyHousehold": {
            "isHousehold": True,
        },
    }

    def test___init__(self) -> None:
        env = Environment(config=self.config)
        assert env.space_size == (10, 10)
        assert env.config == self.config
        assert isinstance(env.prng, type(pytest.importorskip("random").Random()))

    def test_register_classes(self) -> None:
        env = Environment(config=self.config)
        env.register_classes([DummyHousehold])
        assert DummyHousehold in env.registered_classes

    def test_generate_agents(self) -> None:
        env = Environment(config=self.config)
        env.register_classes([DummyHousehold])
        env.reset(seed=42)
        assert len(env.agent_id2agent) == 1
        agent = env.agent_id2agent[0]
        assert isinstance(agent, DummyHousehold)
        assert agent.agent_id == 0
        assert agent.agent_name == "DummyHousehold0"
        pos = env.grid_space.get_pos(agent_id=0)
        assert all(0 <= coord < 10 for coord in pos)
        config_ = self.config.copy()
        config_["DummyHousehold"]["numAgents"] = 3
        env = Environment(config=config_)
        env.register_classes([DummyHousehold])
        env.reset(seed=42)
        assert len(env.agent_id2agent) == 3
        for agent_id in range(3):
            agent = env.agent_id2agent[agent_id]
            assert isinstance(agent, DummyHousehold)
            assert agent.agent_id == agent_id
            assert agent.agent_name == f"DummyHousehold{agent_id}"
            pos = env.grid_space.get_pos(agent_id=agent_id)
            assert all(0 <= coord < 10 for coord in pos)
        config_ = self.config.copy()
        config_["DummyHousehold"]["initialCoords"] = (1, 1)
        env = Environment(config=config_)
        env.register_classes([DummyHousehold])
        env.reset(seed=42)
        agent = env.agent_id2agent[0]
        assert isinstance(agent, DummyHousehold)
        pos = env.grid_space.get_pos(agent_id=0)
        assert pos == (1, 1)
