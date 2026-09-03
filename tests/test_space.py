from random import Random

import pytest

from econsimulacra.spaces import Cell, CellAccess, GridSpace


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

    def test_cell_configuration_and_arbitrary_attributes(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (2, 2),
                "cellDefaults": {
                    "access": {"spawnable": False},
                    "attrs": {"district": "central"},
                },
                "cells": [
                    {
                        "pos": [0, 1],
                        "access": {"traversable": False, "spawnable": True},
                        "attrs": {"kind": "house"},
                    }
                ],
            }
        )

        assert grid_space.get_cell((0, 0)) == Cell(
            access=CellAccess(traversable=True, spawnable=False),
            attrs={"district": "central"},
        )
        assert grid_space.get_cell((0, 1)) == Cell(
            access=CellAccess(traversable=False, spawnable=True),
            attrs={"district": "central", "kind": "house"},
        )
        attrs = grid_space.get_cell_attrs((0, 1))
        attrs["kind"] = "road"
        assert grid_space.get_cell_attrs((0, 1))["kind"] == "house"

    def test_cell_configuration_accepts_simulator_normalized_tuple(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (2, 2),
                "cells": (
                    {
                        "pos": (1, 1),
                        "access": {"spawnable": False},
                        "attrs": {"kind": "common_space"},
                    },
                ),
            }
        )

        assert not grid_space.get_cell((1, 1)).access.spawnable
        assert grid_space.get_cell_attrs((1, 1)) == {"kind": "common_space"}

    def test_update_cell_information(self) -> None:
        grid_space = self.make_grid_space(config={"gridSize": (2, 2)})

        grid_space.update_cell_access(pos=(1, 1), traversable=False, spawnable=False)
        grid_space.update_cell_attrs(pos=(1, 1), updates={"kind": "common_space"})

        cell = grid_space.get_cell((1, 1))
        assert cell.access == CellAccess(traversable=False, spawnable=False)
        assert cell.attrs == {"kind": "common_space"}

    def test_invalid_cell_configuration_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="configured more than once"):
            self.make_grid_space(
                config={
                    "gridSize": (1, 1),
                    "cells": [{"pos": [0, 0]}, {"pos": [0, 0]}],
                }
            )

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
        assert grid_space.get_near_agents(center_pos=(2, 3), max_distance=2) == {
            0,
            1,
            2,
        }
        assert grid_space.get_near_agents(center_pos=(2, 4), max_distance=1) == {
            0,
            1,
            2,
        }
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

    def test_move_agent_rejects_non_traversable_cell(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (2, 1),
                "cells": [{"pos": [1, 0], "access": {"traversable": False}}],
            }
        )
        grid_space.place_agent(agent_id=0, pos=(0, 0))

        with pytest.raises(ValueError, match="not traversable"):
            grid_space.move_agent(agent_id=0, new_pos=(1, 0))

        assert grid_space.get_pos(agent_id=0) == (0, 0)

    def test_move_many_agents(self) -> None:
        grid_space = self.make_grid_space(config=self.config)
        grid_space.place_agent(agent_id=0, pos=(2, 3))
        grid_space.place_agent(agent_id=1, pos=(4, 5))
        grid_space.move_many_agents(agent_id2new_pos={0: (6, 7), 1: (8, 9)})
        assert grid_space.get_pos(agent_id=0) == (6, 7)
        assert grid_space.get_pos(agent_id=1) == (8, 9)
        assert grid_space.get_agents(pos=(2, 3)) == set()
        assert grid_space.get_agents(pos=(4, 5)) == set()

    def test_move_many_agents_validates_before_updating(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (3, 1),
                "cells": [{"pos": [2, 0], "access": {"traversable": False}}],
            }
        )
        grid_space.place_agent(agent_id=0, pos=(0, 0))
        grid_space.place_agent(agent_id=1, pos=(1, 0))

        with pytest.raises(ValueError, match="not traversable"):
            grid_space.move_many_agents(agent_id2new_pos={0: (1, 0), 1: (2, 0)})

        assert grid_space.get_pos(agent_id=0) == (0, 0)
        assert grid_space.get_pos(agent_id=1) == (1, 0)

    def test_movement_colocation_is_allowed(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (2, 1),
                "allowInitialColocatedAgents": False,
            }
        )
        grid_space.place_agent(agent_id=0, pos=(0, 0))
        grid_space.place_agent(agent_id=1, pos=(1, 0))

        grid_space.move_agent(agent_id=0, new_pos=(1, 0))

        assert grid_space.get_agents((1, 0)) == {0, 1}

    def test_calc_next_pos_avoids_non_traversable_cells(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (3, 3),
                "cells": [{"pos": [1, 1], "access": {"traversable": False}}],
            }
        )

        next_pos = grid_space.calc_next_pos(current_pos=(0, 1), destination_pos=(2, 1))

        assert next_pos in {(1, 0), (1, 2)}

    def test_calc_next_pos_respects_velocity(self) -> None:
        grid_space = self.make_grid_space(config={"gridSize": (5, 1)})

        assert grid_space.calc_next_pos(
            current_pos=(0, 0), destination_pos=(4, 0), velocity=2
        ) == (2, 0)
        assert grid_space.calc_next_pos(
            current_pos=(0, 0), destination_pos=(1, 0), velocity=3
        ) == (1, 0)

    def test_calc_next_pos_returns_none_when_no_path_exists(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (3, 1),
                "cells": [{"pos": [1, 0], "access": {"traversable": False}}],
            }
        )

        assert (
            grid_space.calc_next_pos(current_pos=(0, 0), destination_pos=(2, 0)) is None
        )

    def test_calc_next_path_preserves_endpoint_and_reports_distance(self) -> None:
        """Test path-segment output used for mobility resource accounting.

        Args:
            None.

        Returns:
            None.

        Note:
            ``calc_next_pos`` remains equal to the final segment position.
        """
        grid_space = self.make_grid_space(config={"gridSize": [6, 1]})

        path = grid_space.calc_next_path((0, 0), (5, 0), velocity=3)

        assert path == [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert grid_space.calc_next_pos((0, 0), (5, 0), velocity=3) == path[-1]

    @pytest.mark.parametrize("velocity", [0, -1, 1.5, True])
    def test_calc_next_pos_rejects_invalid_velocity(self, velocity: object) -> None:
        grid_space = self.make_grid_space(config={"gridSize": (2, 1)})

        with pytest.raises(ValueError, match="velocity"):
            grid_space.calc_next_pos(
                current_pos=(0, 0),
                destination_pos=(1, 0),
                velocity=velocity,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize(
        "config",
        [
            {"gridSize": (2, 2)},
            {"gridSize": (2, 2), "allowInitialColocatedAgents": True},
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
            config={"gridSize": (2, 2), "allowInitialColocatedAgents": False}
        )
        grid_space.place_agent(agent_id=0, pos=(0, 0))

        with pytest.raises(ValueError, match="already occupied"):
            grid_space.place_agent(agent_id=1, pos=(0, 0))

    def test_distinct_initial_positions_remain_allowed_when_colocation_disabled(
        self,
    ) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (2, 2), "allowInitialColocatedAgents": False}
        )

        grid_space.place_agent(agent_id=0, pos=(0, 0))
        grid_space.place_agent(agent_id=1, pos=(0, 1))

        assert grid_space.get_pos(agent_id=0) != grid_space.get_pos(agent_id=1)

    def test_random_initial_positions_do_not_overlap_when_colocation_disabled(
        self,
    ) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (2, 1), "allowInitialColocatedAgents": False}
        )

        grid_space.place_agent(agent_id=0, pos=None)
        grid_space.place_agent(agent_id=1, pos=None)

        assert len(set(grid_space.agent_id2pos.values())) == 2

    def test_random_initial_positions_can_overlap_when_colocation_enabled(
        self,
    ) -> None:
        grid_space = self.make_grid_space(
            config={"gridSize": (1, 1), "allowInitialColocatedAgents": True}
        )

        grid_space.place_agent(agent_id=0, pos=None)
        grid_space.place_agent(agent_id=1, pos=None)

        assert grid_space.get_agents(pos=(0, 0)) == {0, 1}

    def test_random_initial_position_is_selected_from_spawnable_cells(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (3, 1),
                "cellDefaults": {"access": {"spawnable": False}},
                "cells": [{"pos": [1, 0], "access": {"spawnable": True}}],
            }
        )

        grid_space.place_agent(agent_id=0, pos=None)

        assert grid_space.get_pos(agent_id=0) == (1, 0)

    def test_explicit_initial_position_must_be_spawnable(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (2, 1),
                "cells": [{"pos": [1, 0], "access": {"spawnable": False}}],
            }
        )

        with pytest.raises(ValueError, match="not spawnable"):
            grid_space.place_agent(agent_id=0, pos=(1, 0))

    def test_random_placement_fails_when_no_spawnable_cell_exists(self) -> None:
        grid_space = self.make_grid_space(
            config={
                "gridSize": (2, 1),
                "cellDefaults": {"access": {"spawnable": False}},
            }
        )

        with pytest.raises(ValueError, match="valid spawnable position"):
            grid_space.place_agent(agent_id=0, pos=None)
