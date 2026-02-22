from abc import ABC, abstractmethod
from ..sim_utils import JsonRandom
import random
from random import Random
from typing import Any
from typing import Optional
from typing import Generic, TypeVar


ObsT = TypeVar("ObsT")


class Agent(ABC, Generic[ObsT]):
    """Agent class.

    Once you define the agent class inheriting this agent ABC class, environment automatically generate agents using the class you define.
    act(self, obs: ObsT) method is the only method that must be implemented in the agent class you define, and it will be called by environment at each step to get the action of the agent.
    """

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        env_service_dic: dict[str, Any],
        prng: Optional[Random] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """initialization.

        Args:
            agent_id (int): agent id, which is unique in the environment.
            agent_name (str): agent name, which is used for identification and display purposes.
            prng (Optional[Random], optional): pseudo-random number generator for the agent. Defaults to None.
            config (Optional[dict[str, Any]], optional): configuration dictionary for the agent. Defaults to None.

        Note:
            config example:
            {
                "isHousehold": True,
                "numAgents": 10,
                "inventory": {
                    "Yen": [1000000, 10000000],
                    "Rice": [3, 10],
                    "Fish": [3, 10]
                },
                "isRichInfoAllowed": False,
                "requestObs": ["all"],
                # built-in options: "time", "self_agent_id", "self_name", "self_pos", "self_init_pos", "self_is_moving",
                # "self_destination", "others_pos", "self_tweet", "visible_tl", "incoming_orders", "incoming_proposals",
                # "item_name2price", "others_inventory",
                "provideInfo4AllAgents": [], # built-in option: "self_pos",
                "provideInfo4CoLocatedAgents": [], # built-in option: "inventory"
                "provideInfo4AllowedAgents": [], # built-in option: None,
                "availableServices": ["prompt_builder", "llm_client", ...] # Optional, default []
            }
        """
        self.agent_id: int = agent_id
        self.agent_type: str = self.__class__.__name__
        self.agent_name: str = agent_name
        self.prng: Random = prng if prng is not None else random.Random()
        self.config: dict[str, Any] = config if config is not None else {}
        self.inventory_dic: dict[str, float | int] = self._initialize_inventory(
            self.config
        )
        self.is_rich_info_allowed: bool
        if self.config is not None and "isRichInfoAllowed" in self.config:
            self.is_rich_info_allowed = self.config["isRichInfoAllowed"]
        else:
            self.is_rich_info_allowed = False
        self._setup_request_obs()
        self._setup_infos_to_provide()
        self._setup_env_services(env_service_dic)
        self.self_assign_name(self.config)

    def get_self_name(self) -> str:
        return self.agent_name

    def self_assign_name(self, config: dict[str, Any]) -> None:
        pass

    def _setup_request_obs(self) -> None:
        if self.config is not None and "requestObs" in self.config:
            self.request_obses: list[str] = list(self.config["requestObs"])
        else:
            self.request_obses = ["all"]

    def _setup_infos_to_provide(self) -> None:
        if self.config is not None:
            self.info4all_agents: list[str] = list(
                self.config.get("provideInfo4AllAgents", [])
            )
            self.info4co_located_agents: list[str] = list(
                self.config.get("provideInfo4CoLocatedAgents", [])
            )
            self.info4allowed_agents: list[str] = list(
                self.config.get("provideInfo4AllowedAgents", [])
            )
        else:
            self.info4all_agents = []
            self.info4co_located_agents = []
            self.info4allowed_agents = []

    def _setup_env_services(self, env_service_dic: dict[str, Any]) -> None:
        """setup environment services for the agent based on self.service_dic.

        See also:
            econsimulacra.envs.base._generate_service_providers
            econsimulacra.agents.LLMAgent
        """
        pass

    def _initialize_inventory(self, config: dict[str, Any]) -> dict[str, float | int]:
        json_random = JsonRandom(prng=self.prng)
        inventory_config: dict[str, Any] = config.get("inventory", {})
        inventory_dic: dict[str, Any] = {}
        for item_name, json_value in inventory_config.items():
            amount = json_random.random(json_value=json_value)
            inventory_dic[item_name] = amount
        return inventory_dic

    @abstractmethod
    async def act(self, obs: ObsT) -> dict[str, Any]:
        pass

    def exchange_goods(
        self,
        get_item_name: Optional[str] = None,
        get_item_amount: Optional[float | int] = None,
        give_item_name: Optional[str] = None,
        give_item_amount: Optional[float | int] = None,
    ) -> None:
        if get_item_name is not None:
            if get_item_amount is None:
                raise ValueError(
                    "get_item_amount must be provided when get_item_name is provided."
                )
            if get_item_name not in self.inventory_dic:
                raise ValueError(
                    f"Agent {self.agent_name} does not have {get_item_name} in inventory."
                )
            self.inventory_dic[get_item_name] += get_item_amount
        if give_item_name is not None:
            if give_item_amount is None:
                raise ValueError(
                    "give_item_amount must be provided when give_item_name is provided."
                )
            if give_item_name not in self.inventory_dic:
                raise ValueError(
                    f"Agent {self.agent_name} does not have {give_item_name} in inventory."
                )
            self.inventory_dic[give_item_name] -= give_item_amount

    def provide_info4all_agents(self) -> list[str]:
        """provide information for all agents.

        Returns:
            list[str]: a list of information keys that the agent can provide for all agents.

        Note:
            Usually called by observation providers registered in econsimulacra.environment.base._build_observation_registry.
            Currently, the following built-in observation provider is supported: econsimulacra.environment.base._obs_others_pos
            If another agent is requesting "others_pos" information (i.e., "others_pos" is in the self.request_obses of the another agent),
            the agent can provide its position information to them by adding "self_pos" in self.info4all_agents.
        """
        return self.info4all_agents

    def provide_info4co_located_agents(self) -> list[str]:
        """provide information for those agents who are co-located.

        Returns:
            list[str]: a list of information keys that the agent can provide for those agents who are co-located.

        Note:
            Usually called by observation providers registered in econsimulacra.environment.base._build_observation4co_located_agents_registry.
            Currently, the following built-in observation provider is supported: econsimulacra.environment.base._obs_others_inventory
            If another agent who is co-located with the agent is requesting "others_inventory" informaation (i.e., "others_inventory" is in the self.request_obses of the another agent),
            the agent can provide its inventory information to them by adding "inventory" in self.info4co_located_agents.
        """
        return self.info4co_located_agents

    def provide_info4allowed_agents(self) -> list[str]:
        """provide information for those agents who are allowed.

        Returns:
            list[str]: a list of information keys that the agent can provide for those agents who are is_rich_info_allowed.

        Note:
            Usually called by observation providers registered in econsimulacra.environment.base._build_observation4allowed_agents_registry.
            Currently, built-in observation provider is not supported, but users can implement their own observation provider
            and register it in econsimulacra.environment.base._build_observation4allowed_agents_registry to provide rich information for those agents who are is_rich_info_allowed.
        """
        return self.info4allowed_agents

    def request_obs(self) -> list[str]:
        return self.request_obses

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.agent_name}, inventory={self.inventory_dic})"
