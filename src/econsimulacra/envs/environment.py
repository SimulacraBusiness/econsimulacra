from abc import ABC, abstractmethod
from ..agents import Agent
from .env_utils import find_class
import random
from random import Random
from .social_network import SocialNetwork
from .space import GridSpace
from typing import Any
from typing import Optional
from typing import Type
from typing import Generic, TypeVar

ObsT = TypeVar("ObsT")


class Environment(ABC, Generic[ObsT]):
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
                "isHousehold": bool,
                "numAgents": int, # Optional, default 1
                ...
            },
            "Retailer": {
                "isHousehold": bool, # Optional, default False
                ...
            },
            "Restaurant": {
                "isHousehold": bool, # Optional, default False
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
        self.social_network: SocialNetwork = SocialNetwork()
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
        self.agent_name2agent_id: dict[str, int] = {}
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
                agent_name: str = agent_instance.get_self_name()
                while True:
                    if agent_name not in self.agent_name2agent_id:
                        break
                    agent_name += "_"
                self.agent_name2agent_id[agent_name] = current_agent_id
                self.agent_ids.append(current_agent_id)
                if is_household:
                    self.household_ids.append(current_agent_id)
                self.agent_id2agent[current_agent_id] = agent_instance
                self._assign_agent_to_space(
                    agent_id=current_agent_id,
                    coords=agent_config.get("initialCoords", None),
                )
                self.social_network.add_agent(current_agent_id)
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

    def step(self, action_dic: dict[int, Any]) -> None:
        for agent_id, action in action_dic.items():
            # move <- household側でどこに行きたいかを継続的に指定させて，stepでは1マスずつ動かす
            # add_orders
            # execute_orders
            # follow/unfollow
            # tweet
            ...


    @abstractmethod
    def get_observations(self, agent_id: int) -> ObsT:
        pass
