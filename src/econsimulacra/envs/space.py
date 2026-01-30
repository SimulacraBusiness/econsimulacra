from collections import defaultdict
from typing import DefaultDict


class GridSpace:
    def __init__(self, space_size: tuple[int, ...]) -> None:
        self.space_size: tuple[int, ...] = space_size
        self.pos2agent_ids: DefaultDict[tuple[int, ...], set[int]] = defaultdict(set)
        self.agent_id2pos: dict[int, tuple[int, ...]] = {}

    def _check_bounds(self, pos: tuple[int, ...]) -> None:
        if len(pos) != len(self.space_size):
            raise ValueError(
                f"Position {pos} has different dimension than space size {self.space_size}."
            )
        for i, coord in enumerate(pos):
            if coord < 0 or coord >= self.space_size[i]:
                raise ValueError(
                    f"Coordinate {coord} at dimension {i} is out of bounds for space size {self.space_size}."
                )

    def get_pos(self, agent_id: int) -> tuple[int, ...]:
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        return self.agent_id2pos[agent_id]

    def get_agents(self, pos: tuple[int, ...]) -> set[int]:
        self._check_bounds(pos=pos)
        return self.pos2agent_ids.get(pos, set())

    def place_agent(self, agent_id: int, pos: tuple[int, ...]) -> None:
        self._check_bounds(pos=pos)
        if agent_id in self.agent_id2pos:
            raise ValueError(
                f"Agent ID {agent_id} is already placed in the grid space."
            )
        self.agent_id2pos[agent_id] = pos
        self.pos2agent_ids[pos].add(agent_id)

    def remove_agent(self, agent_id: int) -> None:
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        pos: tuple[int, ...] = self.agent_id2pos[agent_id]
        del self.agent_id2pos[agent_id]
        self.pos2agent_ids[pos].remove(agent_id)
        if len(self.pos2agent_ids[pos]) == 0:
            del self.pos2agent_ids[pos]

    def move_agent(self, agent_id: int, new_pos: tuple[int, ...]) -> None:
        self._check_bounds(pos=new_pos)
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        old_pos: tuple[int, ...] = self.agent_id2pos[agent_id]
        self.pos2agent_ids[old_pos].remove(agent_id)
        if len(self.pos2agent_ids[old_pos]) == 0:
            del self.pos2agent_ids[old_pos]
        self.agent_id2pos[agent_id] = new_pos
        self.pos2agent_ids[new_pos].add(agent_id)

    def move_many_agents(self, agent_id2new_pos: dict[int, tuple[int, ...]]) -> None:
        for agent_id, new_pos in agent_id2new_pos.items():
            self.move_agent(agent_id=agent_id, new_pos=new_pos)
