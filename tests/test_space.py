from econsimulacra.envs import GridSpace
import pytest


class TestGridSpace:
    config = {"gridSize": (10, 10)}

    def test___init__(self) -> None:
        grid_space = GridSpace(config=self.config)
        assert grid_space.space_size == (10, 10)
        assert grid_space.pos2agent_ids == {}
        assert grid_space.agent_id2pos == {}
        grid_space_3d = GridSpace(config={"gridSize": (5, 5, 5)})
        assert grid_space_3d.space_size == (5, 5, 5)
        grid_space_1d = GridSpace(config={"gridSize": (20,)})
        assert grid_space_1d.space_size == (20,)

    def test_place_agent(self) -> None:
        grid_space = GridSpace(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        assert grid_space.get_pos(agent_id=0) == (2, 3)
        assert grid_space.get_agents(pos=(2, 3)) == {0}
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=0, pos=(4, 5))
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=2, pos=(10, 5))
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=3, pos=(-1, 0))
        grid_space_3d = GridSpace(config={"gridSize": (5, 5, 5)})
        grid_space_3d.place_agent(agent_id=0, pos=(1, 2, 3))
        assert grid_space_3d.get_pos(agent_id=0) == (1, 2, 3)
        assert grid_space_3d.get_agents(pos=(1, 2, 3)) == {0}
        with pytest.raises(ValueError):
            grid_space_3d.place_agent(agent_id=1, pos=(5, 2))

    def test_remove_agent(self) -> None:
        grid_space = GridSpace(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.remove_agent(agent_id=0)
        assert grid_space.pos2agent_ids == {}
        assert grid_space.agent_id2pos == {}
        with pytest.raises(ValueError):
            grid_space.remove_agent(agent_id=0)

    def test_move_agent(self) -> None:
        grid_space = GridSpace(config=self.config)
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
        grid_space = GridSpace(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.place_agent(agent_id=1, pos=(4, 5))
        grid_space.move_many_agents(agent_id2new_pos={0: (6, 7), 1: (8, 9)})
        assert grid_space.get_pos(agent_id=0) == (6, 7)
        assert grid_space.get_pos(agent_id=1) == (8, 9)
        assert grid_space.get_agents(pos=(2, 3)) == set()
        assert grid_space.get_agents(pos=(4, 5)) == set()
