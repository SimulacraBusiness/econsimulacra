from ..agents import Agent
from collections import defaultdict
from dataclasses import dataclass
from .env_utils import find_class
import random
from random import Random
from typing import Any
from typing import DefaultDict
from typing import NewType
from typing import Optional
from typing import Type
from typing import Generic, TypeVar

ObsT = TypeVar("ObsT")


@dataclass
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


class Environment(Generic[ObsT]):
    def __init__(self, config: dict[str, Any]) -> None:
        """Initialization.

        config example:
        {
            "gridSpace": [width: int, height: int],
            "simulation": {
                "numSteps": int,
                "agents": ["Household", "Retailer", "Restaurant", ...],
            },
            "Household": {
                isHousehold: bool,
                ...
            },
            "Retailer": {
                isHousehold: bool, # Optional, default False
                ...
            },
            "Restaurant": {
                isHousehold: bool, # Optional, default False
                ...
            },
        }
        Args:
            config (dict[str, Any]): Environment configuration dictionary.
        """
        if "gridSpace" not in config:
            raise ValueError("Environment config must include 'gridSpace' key.")
        self.space_size: tuple[int, ...] = config["gridSpace"]
        self.config: dict[str, Any] = config
        self.prng: Random = random.Random()
        self.registered_classes: list[Type] = []

    def register_classes(self, class_list: list[Type]) -> None:
        self.registered_classes.extend(class_list)

    def reset(self, seed: int) -> None:
        self.prng.seed(seed)
        self.grid_space: GridSpace = GridSpace(space_size=self.space_size)
        assert "simulation" in self.config, "Config must include 'simulation' key."
        assert "agents" in self.config["simulation"], (
            "Simulation config must include 'agents' key."
        )
        agent_types: list[str] = self.config["simulation"]["agents"]
        self._generate_agents(agent_types=agent_types)

    def _generate_agents(self, agent_types: list[str]) -> None:
        """generate agents and place them in the grid space.

        Args:
            agent_types (list[str]): name list of agent types to be generated.
        """
        current_agent_id: int = 0
        self.agent_ids: list[int] = []
        self.household_ids: list[int] = []
        self.agent_id2agent: dict[int, Agent] = {}
        for agent_type in agent_types:
            agent_config: dict[str, Any] = self.config.get(agent_type, {})
            num_agents: int = agent_config.get("numAgents", 1)
            is_household: bool = agent_config.get("isHousehold", False)
            for _ in range(num_agents):
                agent_class: Type[Agent] = find_class(
                    name=agent_type, optional_class_list=self.registered_classes
                )
                agent_instance: Agent = agent_class(
                    agent_id=current_agent_id,
                    agent_name=agent_type + str(current_agent_id),
                    prng=self.prng,
                    config=agent_config,
                )
                self.agent_ids.append(current_agent_id)
                if is_household:
                    self.household_ids.append(current_agent_id)
                self.agent_id2agent[current_agent_id] = agent_instance
                self._assign_agent_to_space(
                    agent_id=current_agent_id,
                    coords=agent_config.get("initialCoords", None),
                )
                current_agent_id += 1

    def _assign_agent_to_space(
        self, agent_id: int, coords: Optional[tuple[int, ...]] = None
    ) -> None:
        if coords is None:
            coords = tuple(
                self.prng.randint(0, self.space_size[dim] - 1)
                for dim in range(len(self.space_size))
            )
        self.grid_space.place_agent(agent_id=agent_id, pos=coords)
