from random import Random

import pytest

from econsimulacra.spaces import GridSpace


class TestGridSpace:
    config = {"gridSize": (10, 10)}

    @staticmethod
    def make_grid_space(config: dict) -> GridSpace:
        return GridSpace(config=config, registered_classes=[], prng=Random(0))

    def test___init__(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
        assert grid_space.space_size == (10, 10)
        assert grid_space.pos2agent_ids == {}
        assert grid_space.agent_id2pos == {}
        grid_space_3d = self.make_grid_space(config={"gridSize": (5, 5, 5)})
        assert grid_space_3d.space_size == (5, 5, 5)
        grid_space_1d = self.make_grid_space(config={"gridSize": (20,)})
        assert grid_space_1d.space_size == (20,)

    def test_place_agent(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        assert grid_space.get_pos(agent_id=0) == (2, 3)
        assert grid_space.get_agents(pos=(2, 3)) == {0}
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=0, pos=(4, 5))
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=2, pos=(10, 5))
        with pytest.raises(ValueError):
            grid_space.place_agent(agent_id=3, pos=(-1, 0))
        grid_space_3d = self.make_grid_space(config={"gridSize": (5, 5, 5)})
        grid_space_3d.place_agent(agent_id=0, pos=(1, 2, 3))
        assert grid_space_3d.get_pos(agent_id=0) == (1, 2, 3)
        assert grid_space_3d.get_agents(pos=(1, 2, 3)) == {0}
        with pytest.raises(ValueError):
            grid_space_3d.place_agent(agent_id=1, pos=(5, 2))

    def test_get_colocated_agents(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.place_agent(agent_id=1, pos=(2, 3))
        grid_space.place_agent(agent_id=2, pos=(4, 5))
        assert grid_space.get_colocated_agents(agent_id=0) == {1}
        assert grid_space.get_colocated_agents(agent_id=1) == {0}
        assert grid_space.get_colocated_agents(agent_id=2) == set()
        with pytest.raises(ValueError):
            grid_space.get_colocated_agents(agent_id=3)

    def test_get_near_agents(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.place_agent(agent_id=1, pos=(2, 4))
        grid_space.place_agent(agent_id=2, pos=(3, 4))
        grid_space.place_agent(agent_id=3, pos=(5, 5))
        assert grid_space.get_near_agents(center_pos=(2, 3), max_distance=1) == {0, 1}
        assert grid_space.get_near_agents(center_pos=(2, 3), max_distance=2) == {0, 1, 2}
        assert grid_space.get_near_agents(center_pos=(2, 4), max_distance=1) == {0, 1, 2}
        assert grid_space.get_near_agents(center_pos=(3, 4), max_distance=1) == {1, 2}
        assert grid_space.get_near_agents(center_pos=(5, 5), max_distance=1) == {3}
        with pytest.raises(ValueError):
            grid_space.get_near_agents(center_pos=(10, 10), max_distance=1)

    def test_remove_agent(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.remove_agent(agent_id=0)
        assert grid_space.pos2agent_ids == {}
        assert grid_space.agent_id2pos == {}
        with pytest.raises(ValueError):
            grid_space.remove_agent(agent_id=0)

    def test_move_agent(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
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
        grid_space = self.make_grid_space(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.place_agent(agent_id=1, pos=(4, 5))
        grid_space.move_many_agents(agent_id2new_pos={0: (6, 7), 1: (8, 9)})
        assert grid_space.get_pos(agent_id=0) == (6, 7)
        assert grid_space.get_pos(agent_id=1) == (8, 9)
        assert grid_space.get_agents(pos=(2, 3)) == set()
        assert grid_space.get_agents(pos=(4, 5)) == set()

    @pytest.mark.parametrize(
        "config",
        [
            {"gridSize": (2, 2)},
            {"gridSize": (2, 2), "allowColocatedAgents": True},
        ],
    )
    def test_colocated_agents_remain_allowed_by_default_and_when_enabled(
        self, config: dict
    ) -> None:
        grid_space = self.make_grid_space(config=config)

        grid_space.place_agent(agent_id=0, pos=(0, 0))
        grid_space.place_agent(agent_id=1, pos=(0, 0))

        assert grid_space.get_agents(pos=(0, 0)) == {0, 1}
        assert grid_space.get_colocated_agents(agent_id=0) == {1}

    def test_colocated_agents_are_rejected_when_disabled(self) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (2, 2), "allowColocatedAgents": False}
        )
        grid_space.place_agent(agent_id=0, pos=(0, 0))

        with pytest.raises(ValueError, match="already occupied"):
            grid_space.place_agent(agent_id=1, pos=(0, 0))

    def test_distinct_initial_positions_remain_allowed_when_colocation_disabled(
        self,
    ) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (2, 2), "allowColocatedAgents": False}
        )

        grid_space.place_agent(agent_id=0, pos=(0, 0))
        grid_space.place_agent(agent_id=1, pos=(0, 1))

        assert grid_space.get_pos(agent_id=0) != grid_space.get_pos(agent_id=1)

    def test_random_initial_positions_do_not_overlap_when_colocation_disabled(
        self,
    ) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (2, 1), "allowColocatedAgents": False}
        )

        grid_space.place_agent(agent_id=0, pos=None)
        grid_space.place_agent(agent_id=1, pos=None)

        assert len(set(grid_space.agent_id2pos.values())) == 2

    def test_random_initial_positions_can_overlap_when_colocation_enabled(
        self,
    ) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (1, 1), "allowColocatedAgents": True}
        )

        grid_space.place_agent(agent_id=0, pos=None)
        grid_space.place_agent(agent_id=1, pos=None)

        assert grid_space.get_agents(pos=(0, 0)) == {0, 1}
