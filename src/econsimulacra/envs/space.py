from collections import defaultdict
from typing import Any, DefaultDict


class GridSpace:
    """Grid Space class.

    The grid space represents the spatial environment where agents are located.
    Each agent can be placed at a specific position in the grid,
    and multiple agents can occupy the same position.
    The GridSpace provides methods to manage agent placements and movements.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the GridSpace.
                This must include:
                - "gridSize": A tuple representing the dimensions of the grid.

        Note:
            See also:
                econsimulacra.envs.base.Environment._generate_space(space_key: str)
        """
        self.config: dict[str, Any] = config
        if "gridSize" not in config:
            raise ValueError("GridSpace requires 'gridSize' in config.")
        self.space_size: tuple[int, ...] = config["gridSize"]
        self.pos2agent_ids: DefaultDict[tuple[int, ...], set[int]] = defaultdict(set)
        self.agent_id2pos: dict[int, tuple[int, ...]] = {}

    def get_space_size(self) -> tuple[int, ...]:
        """Get the shape of the grid space.

        Returns:
            tuple[int, ...]: The dimensions of the grid space.
        """
        return self.space_size

    def _check_bounds(self, pos: tuple[int, ...]) -> None:
        """Check whether the given position is within the bounds of the grid space.

        Args:
            pos (tuple[int, ...]): The position to check.
        """
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
        """Get the position of the agent with the given ID.

        Args:
            agent_id (int): The ID of the agent to get the position for.

        Returns:
            tuple[int, ...]: The position of the agent.
        """
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        return self.agent_id2pos[agent_id]

    def get_agents(self, pos: tuple[int, ...]) -> set[int]:
        """Get the set of agent IDs located at the given position.

        Args:
            pos (tuple[int, ...]): The position to check.

        Returns:
            set[int]: The set of agent IDs located at the given position.
        """
        self._check_bounds(pos=pos)
        return self.pos2agent_ids.get(pos, set())

    def place_agent(self, agent_id: int, pos: tuple[int, ...]) -> None:
        """Place an agent with the given ID at the specified position in the grid space.

        Args:
            agent_id (int): The ID of the agent to place.
            pos (tuple[int, ...]): The position to place the agent at.

        Note:
            See also: ``econsimulacra.envs.base.Environment._assign_agent_to_space``
        """
        self._check_bounds(pos=pos)
        if agent_id in self.agent_id2pos:
            raise ValueError(
                f"Agent ID {agent_id} is already placed in the grid space."
            )
        self.agent_id2pos[agent_id] = pos
        self.pos2agent_ids[pos].add(agent_id)

    def get_colocated_agents(self, agent_id: int) -> set[int]:
        """Get the set of agent IDs located at the same position as the given agent.

        Args:
            agent_id (int): The ID of the agent to check.

        Returns:
            set[int]: The set of agent IDs located at the same position as the given agent.
        """
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        pos: tuple[int, ...] = self.agent_id2pos[agent_id]
        return self.pos2agent_ids.get(pos, set()) - {agent_id}

    def get_near_agents(self, agent_id: int, max_distance: int = 1) -> set[int]:
        """Get agent IDs within the given grid distance from the specified agent.

        Args:
            agent_id (int): The ID of the reference agent.
            max_distance (int): Maximum Manhattan distance to regard as nearby. Default to 1.

        Returns:
            set[int]: Agent IDs within ``max_distance`` from the reference agent,
                excluding the reference agent itself.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        center_pos: tuple[int, ...] = self.agent_id2pos[agent_id]
        near_agent_ids: set[int] = set()
        for other_agent_id, other_pos in self.agent_id2pos.items():
            if other_agent_id == agent_id:
                continue
            distance = sum(
                abs(center_coord - other_coord)
                for center_coord, other_coord in zip(center_pos, other_pos)
            )
            if distance <= max_distance:
                near_agent_ids.add(other_agent_id)
        return near_agent_ids

    def remove_agent(self, agent_id: int) -> None:
        """Remove the agent with the given ID from the grid space.

        Args:
            agent_id (int): The ID of the agent to remove.
        """
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        pos: tuple[int, ...] = self.agent_id2pos[agent_id]
        del self.agent_id2pos[agent_id]
        self.pos2agent_ids[pos].remove(agent_id)
        if len(self.pos2agent_ids[pos]) == 0:
            del self.pos2agent_ids[pos]

    def move_agent(self, agent_id: int, new_pos: tuple[int, ...]) -> None:
        """Move the agent with the given ID to a new position in the grid space.

        Args:
            agent_id (int): The ID of the agent to move.
            new_pos (tuple[int, ...]): The new position to move the agent to.

        Note:
            See also: ``econsimulacra.envs.base.Environment._move``
        """
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
        """Move multiple agents to new positions in the grid space.

        Args:
            agent_id2new_pos (dict[int, tuple[int, ...]]):
                A dictionary mapping agent IDs to their new positions.
        """
        for agent_id, new_pos in agent_id2new_pos.items():
            self.move_agent(agent_id=agent_id, new_pos=new_pos)
