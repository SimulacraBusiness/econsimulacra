import random
from random import Random
from typing import Any, Generic, Literal, Optional, Type, TypeVar

import numpy as np
from numpy.typing import NDArray

from ..agents import Agent
from ..events import EventManager
from ..items import Item
from ..llm_services import PersonaBuilder
from ..logs import (
    AgentGenerationLog,
    ChangePriceLog,
    ConsumptionLog,
    FollowLog,
    InnerThoughtLog,
    ItemGenerationLog,
    Log,
    Logger,
    MoveLog,
    ObsLog,
    OrderExpirationLog,
    OrderLog,
    OrderReactionLog,
    ProposalExpirationLog,
    ProposalLog,
    ProposalReactionLog,
    SpaceAssignLog,
    StateEvaluationLog,
    TweetLog,
    UnfollowLog,
)
from ..memory import MemoryHandler
from ..sim_utils import find_class
from ..social_networks import SocialNetwork
from .obs_providers import (
    FollowCapProvider,
    IncomingOrdersProvider,
    IncomingSwapProposalsProvider,
    ItemName2PriceProvider,
    MemoryProvider,
    NumFollowersProvider,
    NumFollowsProvider,
    ObsProvider,
    ObsProviderFromCoLocatedAgents,
    OthersInventoriesProvider,
    OthersPosProvider,
    RecommendedFollowsProvider,
    SelfDestinationProvider,
    SelfIDProvider,
    SelfInitPosProvider,
    SelfInventoryProvider,
    SelfIsHouseholdProvider,
    SelfIsMovingProvider,
    SelfNameProvider,
    SelfPosProvider,
    SelfSalaryProvider,
    SelfTweetProvider,
    TimeDeltaProvider,
    TimeProvider,
    VisibleTLProvider,
)
from .order import Order, SwapProposal
from .space import GridSpace
from .time_translator import TimeTranslator

ObsT = TypeVar("ObsT")


class Environment(Generic[ObsT]):
    """Environment class.

    The environment in EconSimulacra contains a grid space where agents are placed and can move around,
    a social network where agents can follow each other and interact, a set of items that agents can trade with each other,
    and services that support LLM-based agents in performing various actions and making decisions.
    The main implemented methods in this class include:

    - ``.reset(self)``: reset the environment to the initial state.
    - ``.get_observations(self, agent_id)``: get the observation for the given agent id.
    - ``.step(self, all_actions_dic)``: execute one step of the environment with the given actions from agents.
    """

    def __init__(
        self,
        config: dict[str, Any],
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): Environment configuration dictionary. Expected keys include:
                - "simulation": dict, with keys related to the overall simulation settings, such as "numSteps".
                - "environment": dict, with keys related to the environment settings,
                    such as "space", "socialNetwork", "cashName", "agents", "items", and "service".
                and other detailed settings for each component specified in "environment" entry.
            logger (Logger, optional): Logger instance for logging environment events.
                If None, no logging will be performed. Defaults to None.

        Note:
            config example:
            {
                "simulation": {
                    "numSteps": int, # Required, total number of steps for the simulation.
                    "parallelBatchSize": 2,
                    "events": ["Event1", "Event2", ...], # Optional, list of event names to be generated in the environment. Default is [].
                },
                "environment": {
                    "space": "gridSpace", # Required, the key of the space configuration.
                    "socialNetwork": "socialNetwork", # Required, the key of the social network configuration.
                    "cashName": str, # Required, the name of the cash item in the environment, must be included in "items" values.
                    "agents": ["Household", "Retailer", "Restaurant", ...], # Required, the keys of the agent configurations.
                    "items": ["Yen", "Rice", ...], # Required, must include the cashName specified above.
                    "service": ["promptBuilder", "llmClient", "timeTranslator", "personaBuilder", "memoryHandler"],
                    # Optional, the common environment services provided for the agents. Default to [].
                }
                "gridSpace": {
                    "type": "GridSpace", # See also: econsimulacra.envs.space.GridSpace
                    "gridSize": [int, ...],
                },
                "socialNetwork": {
                    "type": "SocialNetwork", # See also: econsimulacra.envs.social_networks.base.SocialNetwork
                    "followCap": int, # Optional, default is no limit
                    "recSys": {
                        "type": "TwoHopRecommenderSystem", # See also: econsimulacra.envs.social_networks.recsys.TwoHopRecommenderSystem
                        ...
                    }
                },
                "Household": {
                    "type": "LLMAgent",
                    # Requires "llmClient", "promptBuilder", and optionally "personaBuilder"
                    # to be included in the environment services.
                    # See also: econsimulacra.envs.agents.llm_agent.LLMAgent
                    "isHousehold": bool,
                    "numAgents": int, # Optional, default 1
                    ...,
                    "personaConfig": dict, # Optional, default {}
                },
                "Retailer": {
                    "type": "LLMAgent",
                    "isHousehold": bool, # Optional, default False
                    ...
                },
                "Restaurant": {
                    "type": "LLMAgent",
                    "isHousehold": bool, # Optional, default False
                    ...
                },
                "Yen": {
                    "type": "Item", # See also: econsimulacra.envs.items.base.Item
                    "initialPrice": float,
                },
                "Rice": {
                    "type": "Item",
                    "initialPrice": float,
                },
                "promptBuilder": {
                    "type": "PromptBuilder", # See also: econsimulacra.envs.prompt_builder.PromptBuilder
                    ...
                },
                "llmClient": {
                    "type": "OpenAIClient", # See also: econsimulacra.envs.llm_client.OpenAIClient
                    "api_key": str, # Optional if OPENAI_API_KEY environment variable is set
                    "maxConcurrentGenerations": 2, # Optional, maximum number of concurrent generations allowed for the LLM client. Default is 1 (no concurrency).
                    "json_schema_path": str, # Optional, path to a custom JSON schema file for structured generation
                    "modify_schema": bool, # Optional, whether to modify the default JSON schema based on config
                    "gridSpace": [int, ...], # Optional, only needed if modify_schema is True. Must be the same as environment.gridSpace
                    "items": [str, ...], # Optional, only needed if modify_schema is True. Must include all item names used in the environment.items
                    "numAgents": int, # Optional, only needed if modify_schema is True. Must be the total number of agents in the environment.
                },
                "timeTranslator": {
                    "type": "TimeTranslator", # See also: econsimulacra.envs.time_translator.TimeTranslator
                    "numSteps": int, # must be the same as simulation.numSteps
                    "startDatetime": str, # "%Y-%m-%d %H:%M:%S"
                    "endDatetime": str, # "%Y-%m-%d %H:%M:%S"
                },
                "personaBuilder": {
                    "type": "Big5PersonaBuilder", # See also: econsimulacra.envs.persona_builder.big5.Big5PersonaBuilder
                    "numSteps": int, # must be the same as simulation.numSteps
                    "startDatetime": str, # "%Y-%m-%d %H:%M:%S"
                    "endDatetime": str, # "%Y-%m-%d %H:%M:%S"
                },
                "memoryHandler": {
                    "type": "MemoryHandler", # See also: econsimulacra.envs.memory.MemoryHandler
                    "memoryLength": int, # the maximum number of logs to be stored in memory for each agent
                },
                "Event1": {
                    "type": "EventType",
                    "trigger": {
                        "at": [int], # optional
                        "every": int, # optional
                        "with": [str], # optional
                        "between": [int, int], # optional
                        "probability": float # optional
                    },
                    "other_parameters": ...
                },
                ... # other event configurations
            }
        """
        env_config: dict[str, Any] = config.get("environment", {})
        if "cashName" not in env_config:
            raise ValueError("Environment config must include 'cashName' key.")
        self.cash_name: str = env_config["cashName"]
        self.config: dict[str, Any] = config
        self.seed: Optional[int] = None
        self.prng: Random = random.Random()
        self.registered_classes: list[Type] = []
        self.logger: Optional[Logger] = logger
        self._time: int = -1
        self.service_dic: dict[str, Any] = {}

    def get_service(self, service_type: Type[Any]) -> Optional[Any]:
        """Return the first service instance matching the given type."""
        for provider in self.service_dic.values():
            if isinstance(provider, service_type):
                return provider
        return None

    def get_time_translator(self) -> Optional[TimeTranslator]:
        """Get the TimeTranslator service provider from the environment's service dictionary, if it exists."""
        return self.get_service(TimeTranslator)

    def get_memory_handler(self) -> Optional[MemoryHandler]:
        """Get the MemoryHandler service provider from the environment's service dictionary, if it exists."""
        return self.get_service(MemoryHandler)

    def get_persona_builder(self) -> Optional[PersonaBuilder]:
        """Get the PersonaBuilder service provider from the environment's service dictionary, if it exists."""
        return self.get_service(PersonaBuilder)

    def get_time(self) -> int | str:
        """Get the current time in the environment.

        Note:
            If a TimeTranslator service provider is available, use it to convert the internal time step
            to a datetime string; otherwise, return the internal time step as an integer.
        """
        time_translator: Optional[TimeTranslator] = self.get_time_translator()
        if time_translator is not None:
            return time_translator.step_to_datetime(self._time)
        return self._time

    def get_time_step(self) -> int:
        """Get the current time step in the environment as an integer."""
        return self._time

    def get_timedelta(self) -> int | str:
        """Get the time delta for each step in the environment.

        Note:
            If a TimeTranslator service provider is available, use it to get the time delta
            in a suitable format (e.g., "1 day", "2 hours", etc.);
            otherwise, return the default time delta of 1 (which can be interpreted as 1 step).
        """
        time_translator: Optional[TimeTranslator] = self.get_time_translator()
        if time_translator is not None:
            return time_translator.get_timedelta()
        return 1

    def register_classes(self, class_list: list[Type]) -> None:
        """Register classes to the environment for dynamic instantiation from config.

        Args:
            class_list (list[Type]): a list of classes to be registered.
                These classes can then be referred to by their class names in the environment config for dynamic instantiation.

        Note:
            Use this method to register your custom classes.
        """
        self.registered_classes.extend(class_list)

    def reset(self, seed: Optional[int]) -> None:
        """Reset the environment to the initial state.

        Args:
            seed (Optional[int]): random seed for environment initialization. If None, no specific seed is set.

        Note:
            This method
            - resets the random seed,
            - generates the grid space and social network according to the config,
            - generates service providers, agents, and items according to the config,
        """
        self._set_invalid_action_dic()
        if self.logger is not None:
            self.logger.clear()
        if seed is not None:
            self.seed = seed
            self.prng.seed(seed)
        event_keys: list[str] = self.config.get("simulation", {}).get("events", [])
        events_dic: dict[str, Any] = {
            event_key: self.config.get(event_key, {}) for event_key in event_keys
        }
        self.event_manager: EventManager = EventManager(
            event_names=event_keys,
            events_dic=events_dic,
            registered_classes=self.registered_classes,
            prng=self.prng,
        )
        assert "environment" in self.config, "Config must include 'environment' key."
        assert isinstance(self.config["environment"], dict), (
            "'environment' key must be a dictionary."
        )
        if "space" not in self.config["environment"]:
            raise ValueError("Environment config must include 'space' key.")
        space_key: str = self.config["environment"]["space"]
        self.grid_space: GridSpace = self._generate_space(space_key=space_key)
        if "socialNetwork" not in self.config["environment"]:
            raise ValueError("Environment config must include 'socialNetwork' key.")
        social_network_key: str = self.config["environment"]["socialNetwork"]
        self.social_network: SocialNetwork = self._generate_social_network(
            social_network_key=social_network_key
        )
        service_provider_keys: list[str] = self.config["environment"].get("service", [])
        self._generate_service_providers(service_provider_keys=service_provider_keys)
        assert "items" in self.config["environment"], (
            "Environment config must include 'items' key."
        )
        item_keys: list[str] = self.config["environment"]["items"]
        self._generate_items(item_keys=item_keys)
        assert "agents" in self.config["environment"], (
            "Environment config must include 'agents' key."
        )
        agent_keys: list[str] = self.config["environment"]["agents"]
        self._generate_agents(agent_keys=agent_keys)
        self.pending_orders: list[Order] = []
        self.pending_swap_proposals: list[SwapProposal] = []
        self.latest_order_id: int = 0
        self.latest_proposal_id: int = 0
        if self.logger is not None:
            self.logger.process_logs()
        self._time = 0
        self.event_manager.trigger_events_after_step(
            time_step=self.get_time_step(), env=self
        )

    def _set_invalid_action_dic(self) -> None:
        """Initialize the dictionary for counting invalid actions.

        Note:
            When the agents take invalid actions (e.g., moving to an invalid destination, placing an order with invalid parameters, etc.),
            the environment will ignore the invalid part of the action and execute the rest valid part (if any),
            and count the invalid action in this dictionary for later analysis.
        """
        self.invalid_action_dic: dict[str, int] = {
            "move": 0,
            "consumptions": 0,
            "orders": 0,
            "proposals": 0,
            "reactions": 0,
            "set_prices": 0,
            "follow": 0,
            "unfollow": 0,
        }

    def _generate_space(self, space_key: str) -> GridSpace:
        """Generate the grid space.

        Args:
            space_key (str): name of the grid space type to be generated.
                It is "gridSpace" in the provided example config.

        Returns:
            GridSpace: the generated grid space instance.
        """
        space_config: dict[str, Any] = self.config[space_key]
        space_type: str = space_config.get("type", space_key)
        space_class: Type[GridSpace] = find_class(
            name=space_type, optional_class_list=self.registered_classes
        )
        grid_space: GridSpace = space_class(space_config)
        return grid_space

    def _generate_social_network(self, social_network_key: str) -> SocialNetwork:
        """Generate the social network.

        Args:
            social_network_key (str): name of the social network type to be generated.
                It is "socialNetwork" in the provided example config.

        Returns:
            SocialNetwork: the generated social network instance.
        """
        social_network_config: dict[str, Any] = self.config[social_network_key]
        social_network_type: str = social_network_config.get("type", social_network_key)
        social_network_class: Type[SocialNetwork] = find_class(
            name=social_network_type, optional_class_list=self.registered_classes
        )
        social_network: SocialNetwork = social_network_class(
            config=social_network_config,
            registered_classes=self.registered_classes,
            prng=self.prng,
        )
        return social_network

    def _generate_service_providers(self, service_provider_keys: list[str]) -> None:
        """Generate service providers.

        Args:
            service_provider_keys (list[str]): name list of service provider types to be generated.
                It is ["promptBuilder", "llmClient", "timeTranslator", "personaBuilder", "memoryHandler"] in the provided example config.

        Note:
            The current econsimulacra.agents.llm_agent.LLMAgent implementation requires:
            "llmClient", promptBuilder", and "personaBuilder" (optional) to be included
            in the service_provider_keys.
            See also: econsimulacra.agents.llm_agent.LLMAgent.__init__
        """
        for service_provider_key in service_provider_keys:
            if service_provider_key not in self.config:
                raise ValueError(
                    f"Service provider config for {service_provider_key} is not found in the environment config."
                )
            service_provider_config: dict[str, Any] = self.config[service_provider_key]
            instance_type: str = service_provider_config.get(
                "type", service_provider_key
            )
            service_provider_class: Type = find_class(
                name=instance_type, optional_class_list=self.registered_classes
            )
            service_provider_instance = service_provider_class(
                config=service_provider_config,
                prng=self.prng,
                registered_classes=self.registered_classes,
            )
            self.service_dic[service_provider_key] = service_provider_instance

    def remember_log(self, log: Log) -> None:
        """Call MemoryHandler.update(log) to update the memory with the given log, if MemoryHandler is available in the environment services.

        Args:
            log (Log): the log to be remembered in memory.

        Note:
            Called when any kind of log (e.g., AgentGenerationLog, MoveLog, ConsumptionLog, OrderLog, ProposalLog, etc.)
            is generated in the environment.
        """
        memory_handler: Optional[MemoryHandler] = self.get_memory_handler()
        if memory_handler is not None:
            memory_handler.update(log)

    def get_persona(self, agent_id: int) -> Optional[dict[str, Any]]:
        """Get the persona details for the given agent ID,
        if PersonaBuilder service provider is available in the environment.

        Args:
            agent_id (int): the ID of the agent to get persona details for.

        Returns:
            persona_dic (dict[str, Any], optional): the persona details dictionary for the given agent ID,
                or None if PersonaBuilder service provider is not available in the environment.
        """
        persona_builder: Optional[PersonaBuilder] = self.get_persona_builder()
        persona_dic: Optional[dict[str, Any]] = None
        if persona_builder is not None:
            persona_dic = persona_builder.get_persona(agent_id=agent_id)
        return persona_dic

    def _generate_agents(self, agent_keys: list[str]) -> None:
        """Generate agents and place them in the grid space.

        Args:
            agent_types (list[str]): name list of agent types to be generated.

        Note:
            agent_config optionally includes:
            - "type": str, the type of the agent, which can be used to find the corresponding agent class for instantiation.
            - "isHousehold": bool, whether the agent belongs to the household group.
                If True, the agent is added to the household_ids list.
            - "initialCoords": tuple[int, ...], the initial coordinates of the agent in the grid space.
                If not provided, the agent will be placed in a random empty cell in the grid space.
            See also:
                econsimulacra.envs.agents.base.Agent.__init__
                econsimulacra.envs.agents.llm_agent.LLMAgent.__init__
        """
        current_agent_id: int = 0
        self.agent_ids: list[int] = []
        self.household_ids: list[int] = []
        self.others_ids: list[int] = []
        self.agent_id2agent: dict[int, Agent] = {}
        self.agent_id2agent_name: dict[int, str] = {}
        self.agent_name2agent_id: dict[str, int] = {}
        self.agent_id2initial_coords: dict[int, tuple[int, ...]] = {}
        self.agent_id2is_moving: dict[int, bool] = {}
        self.agent_id2destination: dict[int, Optional[tuple[int, ...]]] = {}
        self.agent_id2initial_inventory: dict[int, dict[str, int | float]] = {}
        self.agent_id2wealth: dict[int, float | int] = {}
        for agent_key in agent_keys:
            agent_config: dict[str, Any] = self.config.get(agent_key, {})
            instance_type: str = agent_config.get("type", agent_key)
            num_agents: int = agent_config.get("numAgents", 1)
            is_household: bool = agent_config.get("isHousehold", False)
            for _ in range(num_agents):
                agent_class: Type[Agent] = find_class(
                    name=instance_type, optional_class_list=self.registered_classes
                )
                agent_instance: Agent = agent_class(
                    agent_id=current_agent_id,
                    agent_name=agent_key + str(current_agent_id),
                    env_service_dic=self.service_dic,
                    prng=self.prng,
                    config=agent_config,
                )
                agent_name: str = agent_instance.get_self_name()
                inventory_dic: dict[str, int | float] = agent_instance.get_inventory()
                self.agent_id2initial_inventory[current_agent_id] = inventory_dic
                wealth: float | int = self._calculate_wealth(
                    inventory_dic=inventory_dic
                )
                self.agent_id2wealth[current_agent_id] = wealth
                log: AgentGenerationLog = AgentGenerationLog(
                    agent_id=current_agent_id,
                    time=self.get_time(),
                    time_step=self.get_time_step(),
                    agent_type=agent_instance.agent_type,
                    agent_name=agent_name,
                    wealth=wealth,
                    inventory_dic=inventory_dic,
                    persona_dic=self.get_persona(agent_id=current_agent_id),
                )
                self.remember_log(log)
                self.event_manager.trigger_events_after_log(log=log, env=self)
                if self.logger is not None:
                    log.read_and_write(logger=self.logger)
                while True:
                    if agent_name not in self.agent_name2agent_id:
                        break
                    agent_name += "_"
                self.agent_name2agent_id[agent_name] = current_agent_id
                self.agent_id2agent_name[current_agent_id] = agent_name
                self.agent_ids.append(current_agent_id)
                if is_household:
                    self.household_ids.append(current_agent_id)
                else:
                    self.others_ids.append(current_agent_id)
                self.agent_id2agent[current_agent_id] = agent_instance
                self._assign_agent_to_space(
                    agent_id=current_agent_id,
                    coords=agent_config.get("initialCoords", None),
                )
                self.agent_id2is_moving[current_agent_id] = False
                self.agent_id2destination[current_agent_id] = None
                self.social_network.add_agent(
                    agent_id=current_agent_id, agent_name=agent_name
                )
                current_agent_id += 1

    def _generate_items(self, item_keys: list[str]) -> None:
        """Generate items.

        Args:
            item_keys (list[str]): name list of items to be generated.

        Note:
            item_config optionally includes:
            - "type": str, the type of the item, which can be used to find the corresponding item class for instantiation.
            - "initialPrice": float, the initial price of the item. If not provided, the initial price is set to 0.
            See also:
                econsimulacra.items.base.Item.__init__
        """
        self.item_name2item: dict[str, Item] = {}
        for item_key in item_keys:
            item_config: dict[str, Any] = self.config.get(item_key, {})
            item_type: str = item_config.get("type", item_key)
            item_class: Type[Item] = find_class(
                name=item_type, optional_class_list=self.registered_classes
            )
            item_instance: Item = item_class(
                item_id=len(self.item_name2item),
                item_name=item_key,
                config=item_config,
            )
            self.item_name2item[item_key] = item_instance
            log: ItemGenerationLog = ItemGenerationLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                item_name=item_key,
                price=item_instance.get_price(),
            )
            self.remember_log(log)
            self.event_manager.trigger_events_after_log(log=log, env=self)
            if self.logger is not None:
                log.read_and_write(logger=self.logger)

    def get_total_amount(self, item_name: str) -> float | int:
        """Get the total amount of the specified item in the environment.

        Args:
            item_name (str): the name of the item to get the total amount for.
        """
        if item_name not in self.item_name2item:
            raise ValueError(f"Item name {item_name} not found in the environment.")
        total_amount: float | int = 0
        for agent in self.agent_id2agent.values():
            total_amount += agent.get_item_amount(item_name=item_name)
        return total_amount

    def _assign_agent_to_space(
        self, agent_id: int, coords: Optional[tuple[int, ...]] = None
    ) -> None:
        """Assign the agent to the grid space at the specified coordinates, or at random empty coordinates if not specified.

        Args:
            agent_id (int): The ID of the agent to assign.
            coords (Optional[tuple[int, ...]]): The coordinates to assign the agent to.
                If None, a random empty coordinate will be chosen.

        Note:
            Called when generating agents in the reset() method.
            The initial coordinates can be specified in the agent config with the "initialCoords" key.
        """
        space_size: tuple[int, ...] = self.grid_space.get_space_size()
        if coords is None:
            coords = tuple(
                self.prng.randint(0, space_size[dim] - 1)
                for dim in range(len(space_size))
            )
        self.grid_space.place_agent(agent_id=agent_id, pos=coords)
        self.agent_id2initial_coords[agent_id] = coords
        log: SpaceAssignLog = SpaceAssignLog(agent_id=agent_id, pos=coords)
        self.remember_log(log)
        self.event_manager.trigger_events_after_log(log=log, env=self)
        if self.logger is not None:
            log.read_and_write(logger=self.logger)

    def step(self, all_actions_dic: dict[int, dict[str, Any]]) -> None:
        """Execute one step of the environment.

        Args:
            all_actions_dic (dict[int, dict[str, Any]]): a dictionary mapping agent IDs
                to their respective action dictionaries for this step.

        Note:
            See also:
            econsimulacra.envs.base.Environment.apply_action_to_env for the expected format of each agent's action dictionary.
        """
        for agent_id, action_dic in all_actions_dic.items():
            self.apply_action_to_env(
                agent_id=agent_id,
                action_dic=action_dic,
            )
        self._process_orders_and_proposals()
        self._update_time()
        self._remove_expired_orders_and_proposals()
        for agent_id in self.agent_ids:
            self.evaluate_agent_state(agent_id=agent_id)
        if self.logger is not None:
            self.logger.process_logs()
        self.event_manager.trigger_events_after_step(
            time_step=self.get_time_step(), env=self
        )

    def apply_action_to_env(self, agent_id: int, action_dic: dict[str, Any]) -> None:
        """Apply the action of a single agent to the environment.

        Args:
            agent_id (int): the ID of the agent whose action is to be applied.
            action_dic (dict[str, Any]): the action dictionary of the agent for this step.

        Note:
            This method processes the action dictionary of a single agent by

            - checking the validity of each part of the action,
            - executing the valid part of the action

            action_dic example::

                {
                    "move": tuple[int, ...] | str,
                    "consumptions": [
                        {"item_name": str, "item_amount": float | int}, ...
                    ],
                    "orders": [
                        {"counterparty_id": int, "counterparty_name": str,
                         "item_name": str, "item_amount": float | int, "ttl": int}, ...
                    ],
                    "proposals": [
                        {"responder_agent_id": int, "responder_agent_name": str,
                         "give_item_name": str, "give_item_amount": float | int,
                         "get_item_name": str, "get_item_amount": float | int, "ttl": int}, ...
                    ],
                    "reactions": [
                        {"kind": "order", "id": int, "accept_amount": float | int},
                        {"kind": "proposal", "id": int, "accept": bool}, ...
                    ],
                    "set_prices": [{"item_name": str, "price": float}, ...],
                    "inner_thought": str,
                    "tweet": str,
                    "follow": int,
                    "unfollow": int,
                }

            See also: ``econsimulacra.llm_services.constant.DEFAULT_ACTION_JSON_SCHEMA``
            for the expected format of the action dictionary used to validate
            generated actions from LLM-based agents.
        """
        where_to_move: Optional[tuple[int, ...] | str] = action_dic.get("move", None)
        move_allowed: bool = self._check_move(where_to_move=where_to_move)
        if not move_allowed:
            self.invalid_action_dic["move"] += 1
            where_to_move = None
        if agent_id in self.household_ids:
            self._move(
                agent_id=agent_id,
                where_to_move=where_to_move,
            )
        consumptions: list[dict[str, Any]] = action_dic.get("consumptions", [])
        valid_consumptions: list[dict[str, Any]] = self._check_consumptions(
            agent_id=agent_id, consumptions=consumptions
        )
        self.invalid_action_dic["consumptions"] += len(consumptions) - len(
            valid_consumptions
        )
        self._consume_items(
            agent_id=agent_id,
            consumptions=valid_consumptions,
        )
        orders: list[dict[str, Any]] = action_dic.get("orders", [])
        valid_orders: list[dict[str, Any]] = self._check_orders(
            agent_id=agent_id, orders=orders
        )
        self.invalid_action_dic["orders"] += len(orders) - len(valid_orders)
        proposals: list[dict[str, Any]] = action_dic.get("proposals", [])
        valid_proposals: list[dict[str, Any]] = self._check_proposals(
            agent_id=agent_id, proposals=proposals
        )
        self.invalid_action_dic["proposals"] += len(proposals) - len(valid_proposals)
        self._add_new_orders_and_proposals(
            agent_id=agent_id,
            orders=valid_orders,
            proposals=valid_proposals,
        )
        reactions: list[dict[str, Any]] = action_dic.get("reactions", [])
        valid_reactions: list[dict[str, Any]] = self._check_reactions(
            agent_id=agent_id, reactions=reactions
        )
        self.invalid_action_dic["reactions"] += len(reactions) - len(valid_reactions)
        self._process_reactions(
            agent_id=agent_id,
            reactions=valid_reactions,
        )
        set_prices: list[dict[str, Any]] = action_dic.get("set_prices", [])
        valid_set_prices: list[dict[str, Any]] = self._check_set_prices(
            agent_id=agent_id, set_prices=set_prices
        )
        self.invalid_action_dic["set_prices"] += len(set_prices) - len(valid_set_prices)
        self._set_prices(
            agent_id=agent_id,
            set_prices=valid_set_prices,
        )
        inner_thought: str = action_dic.get("inner_thought", "")
        self._process_inner_thought(
            agent_id=agent_id,
            inner_thought=inner_thought,
        )
        tweet: Optional[str] = action_dic.get("tweet", None)
        follow_agent_id: Optional[int] = action_dic.get("follow", None)
        unfollow_agent_id: Optional[int] = action_dic.get("unfollow", None)
        valid_follow_agent_id: Optional[int]
        valid_unfollow_agent_id: Optional[int]
        valid_follow_agent_id, valid_unfollow_agent_id = self._check_follow_unfollow(
            agent_id=agent_id,
            follow_agent_id=follow_agent_id,
            unfollow_agent_id=unfollow_agent_id,
        )
        if follow_agent_id is not None and valid_follow_agent_id is None:
            self.invalid_action_dic["follow"] += 1
        if unfollow_agent_id is not None and valid_unfollow_agent_id is None:
            self.invalid_action_dic["unfollow"] += 1
        self._act_in_social_network(
            agent_id=agent_id,
            tweet=tweet,
            follow_agent_id=valid_follow_agent_id,
            unfollow_agent_id=valid_unfollow_agent_id,
        )

    def _process_inner_thought(self, agent_id: int, inner_thought: str) -> None:
        """Process the inner thought of the agent for this step.

        Args:
            agent_id (int): the ID of the agent.
            inner_thought (str): the inner thought text to be processed.

        Note:
            The inner thought is a text generated by the agent to describe its inner feelings.
            It is not an action that directly affects the environment.
            Just generate a InnerThoughtLog and remember it in memory (if MemoryHandler is available) and write it to logger (if logger is available).
        """
        log: InnerThoughtLog = InnerThoughtLog(
            time=self.get_time(),
            time_step=self.get_time_step(),
            agent_id=agent_id,
            inner_thought=inner_thought,
        )
        self.remember_log(log)
        self.event_manager.trigger_events_after_log(log=log, env=self)
        if self.logger is not None:
            log.read_and_write(logger=self.logger)

    def _check_move(self, where_to_move: Optional[tuple[int, ...] | str]) -> bool:
        """Check whether the move is valid.

        Args:
            where_to_move (Optional[tuple[int, ...] | str]): the target position or agent name to move to.

        Returns:
            bool: whether the move is valid.

        Note:
            Checked conditions:
            - If where_to_move is None, the move is valid (agent stays in the current position).
            - If where_to_move is a string, it must be the name of an existing agent.
            - If where_to_move is a tuple, it must be within the bounds of the environment.
        """
        if where_to_move is None:
            return True
        destination_pos: Optional[tuple[int, ...]] = self._calc_destination_pos(
            where_to_move=where_to_move
        )
        if destination_pos is None:
            return False
        space_size: tuple[int, ...] = self.grid_space.get_space_size()
        if len(destination_pos) != len(space_size):
            return False
        for dim in range(len(destination_pos)):
            if not (0 <= destination_pos[dim] < space_size[dim]):
                return False
        return True

    def _move(
        self, agent_id: int, where_to_move: Optional[tuple[int, ...] | str] = None
    ) -> None:
        """Apply move action to the environment by moving the agent one step towards the destination.

        Args:
            agent_id (int): the ID of the agent to move.
            where_to_move (Optional[tuple[int, ...] | str]): the target position or agent name to move to.
                If None, the agent will stay in the current position.

        Note:
            Move action example:
            {
                "move": tuple[int, ...] | str # <- corresponds to the where_to_move argument
            }
            Currently, the agent can only move one step (i.e., to an adjacent cell)
            towards the destination in one step of the environment.
        """
        current_pos: tuple[int, ...] = self.grid_space.get_pos(agent_id=agent_id)
        if where_to_move is None:
            return
        destination_pos: Optional[tuple[int, ...]] = self._calc_destination_pos(
            where_to_move=where_to_move
        )
        if destination_pos is None:
            raise ValueError(f"Invalid move destination: {where_to_move}")
        next_pos: tuple[int, ...] = self._calc_next_pos(
            current_pos=current_pos, destination_pos=destination_pos
        )
        self.grid_space.move_agent(agent_id=agent_id, new_pos=next_pos)
        log: MoveLog = MoveLog(
            time=self.get_time(),
            time_step=self.get_time_step(),
            agent_id=agent_id,
            old_pos=current_pos,
            new_pos=next_pos,
            init_pos=self.agent_id2initial_coords[agent_id],
        )
        self.remember_log(log)
        self.event_manager.trigger_events_after_log(log=log, env=self)
        if self.logger is not None:
            log.read_and_write(logger=self.logger)
        if next_pos == destination_pos:
            self.agent_id2is_moving[agent_id] = False
            self.agent_id2destination[agent_id] = None
        else:
            self.agent_id2is_moving[agent_id] = True
            self.agent_id2destination[agent_id] = destination_pos

    def _calc_destination_pos(
        self, where_to_move: tuple[int, ...] | str
    ) -> Optional[tuple[int, ...]]:
        """Calculate the destination position based on the move target specified in the action dictionary.

        Args:
            where_to_move (tuple[int, ...] | str): the target position or agent name to move to.

        Returns:
            Optional[tuple[int, ...]]: the calculated destination position, or None if the input is invalid.
        """
        destination_pos: Optional[tuple[int, ...]]
        if isinstance(where_to_move, str):
            destination_name: str = where_to_move
            destination_id: Optional[int] = self.agent_name2agent_id.get(
                destination_name
            )
            if destination_id is None:
                return None
            destination_pos = self.grid_space.get_pos(agent_id=destination_id)
        elif isinstance(where_to_move, tuple):
            destination_pos = where_to_move
        else:
            destination_pos = None
        return destination_pos

    def _calc_next_pos(
        self, current_pos: tuple[int, ...], destination_pos: tuple[int, ...]
    ) -> tuple[int, ...]:
        """Calculate the next position for the agent to move towards the destination.

        Args:
            current_pos (tuple[int, ...]): the current position of the agent.
            destination_pos (tuple[int, ...]): the target position of the agent.

        Returns:
            tuple[int, ...]: the next position for the agent to move towards the destination.

        Note:
            The agent can only move one step (i.e., to an adjacent cell) towards
            the destination in one step of the environment.
            This method calculate the nearest adjacent cell to the destination and return
            its coordinates as the next position.
        """
        next_pos: list[int] = list(current_pos)
        for dim in range(len(current_pos)):
            if current_pos[dim] < destination_pos[dim]:
                next_pos[dim] += 1
            elif current_pos[dim] > destination_pos[dim]:
                next_pos[dim] -= 1
        return tuple(next_pos)

    def _check_consumptions(
        self, agent_id: int, consumptions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check whether the consumptions are valid.

        Args:
            agent_id (int): agent id of the agent who performs the consumptions.
            consumptions (list[dict[str, Any]]): list of consumption dictionaries.

        Returns:
            list[dict[str, Any]]: a list of valid consumption dictionaries.

        Note:
            Checked conditions:
            - item_name in each consumption must be in the environment.
            - item_amount in each consumption must be positive and not exceed the agent's inventory.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        valid_consumptions: list[dict[str, Any]] = []
        for consumption in consumptions:
            item_name: str = consumption.get("item_name", "")
            item_amount: float | int = consumption.get("item_amount", 0)
            if item_name not in self.item_name2item:
                continue
            allowed_amount: float | int = agent.get_item_amount(item_name)
            if item_amount > allowed_amount:
                continue
            if item_amount <= 0:
                continue
            valid_consumptions.append(consumption)
        return valid_consumptions

    def _consume_items(self, agent_id: int, consumptions: list[dict[str, Any]]) -> None:
        """Apply the consumption action to the environment by reducing the agent's inventory of the consumed items.

        Args:
            agent_id (int): agent id of the agent who performs the consumptions.
            consumptions (list[dict[str, Any]]): list of consumption dictionaries.

        Note:
            Consumption action example:
            {
                "consumptions": [
                    {"item_name": str, "item_amount": float | int},
                    ...
                ] # <- corresponds to the consumptions list in the arguments.
            }
        """
        agent: Agent = self.agent_id2agent[agent_id]
        for consumption in consumptions:
            item_name: str = consumption["item_name"]
            if item_name not in self.item_name2item:
                raise ValueError(f"Item {item_name} does not exist in the environment.")
            item_amount: float | int = consumption["item_amount"]
            agent.exchange_goods(
                get_item_name=None,
                get_item_amount=None,
                give_item_name=item_name,
                give_item_amount=item_amount,
            )
            log: ConsumptionLog = ConsumptionLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                item_name=item_name,
                item_amount=item_amount,
            )
            self.remember_log(log)
            self.event_manager.trigger_events_after_log(log=log, env=self)
            if self.logger is not None:
                log.read_and_write(logger=self.logger)

    def _check_orders(
        self, agent_id: int, orders: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check whether the orders are valid.

        Args:
            agent_id (int): agent id of the agent who places the orders.
            orders (list[dict[str, Any]]): list of order dictionaries.

        Returns:
            list[dict[str, Any]]: a list of valid order dictionaries.

        Note:
            Checked conditions:
            - counterparty_id or counterparty_name in each order must be provided.
            - counterparty_id must be an existing agent id in the environment.
            - item_name in each order must be an existing item in the environment.
            - item_name in each order cannot be the cash.
            - item_amount in each order must be positive
            - The agent must have enough cash to buy all of the items.
        """
        valid_orders: list[dict[str, Any]] = []
        agent: Agent = self.agent_id2agent[agent_id]
        total_cost: float | int = 0.0
        cash_amount: float | int = agent.get_item_amount(self.cash_name)
        for order_dic in orders:
            counterparty_id: Optional[int] = order_dic.get("counterparty_id", None)
            counterparty_name: Optional[str] = order_dic.get("counterparty_name", None)
            if counterparty_id is None and counterparty_name is None:
                continue
            elif counterparty_id is None and counterparty_name is not None:
                counterparty_id = self.agent_name2agent_id.get(counterparty_name, None)
            if counterparty_id not in self.agent_ids:
                continue
            if "item_name" not in order_dic:
                continue
            item_name: str = order_dic["item_name"]
            if item_name not in self.item_name2item:
                continue
            if item_name == self.cash_name:
                continue
            item_amount: float | int = order_dic.get("item_amount", 0)
            if item_amount <= 0:
                continue
            item: Item = self.item_name2item[item_name]
            expected_price: Optional[float] = item.get_price()
            if expected_price is not None:
                expected_price *= item_amount
                total_cost += expected_price
            if total_cost > cash_amount:
                if expected_price is not None:
                    total_cost -= expected_price
                continue
            else:
                valid_orders.append(order_dic)
        return valid_orders

    def _check_proposals(
        self, agent_id: int, proposals: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check whether the proposals are valid.

        Args:
            agent_id (int): agent id of the agent who makes the proposals.
            proposals (list[dict[str, Any]]): list of proposal dictionaries.

        Returns:
            list[dict[str, Any]]: a list of valid proposal dictionaries.

        Note:
            Checked conditions:
            - responder_agent_id or responder_agent_name in each proposal must be provided.
            - responder_agent_id must be an existing agent id in the environment.
            - give_item_name and get_item_name in each proposal must be existing items in the environment.
            - give_item_amount and get_item_amount in each proposal must be positive.
            - The agent must have enough inventory of give_item to make the proposals.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        valid_proposals: list[dict[str, Any]] = []
        for proposal_dic in proposals:
            responder_agent_id: Optional[int] = proposal_dic.get(
                "responder_agent_id", None
            )
            responder_agent_name: Optional[str] = proposal_dic.get(
                "responder_agent_name", None
            )
            if responder_agent_id is None and responder_agent_name is None:
                continue
            elif responder_agent_id is None and responder_agent_name is not None:
                responder_agent_id = self.agent_name2agent_id.get(
                    responder_agent_name, None
                )
            if responder_agent_id not in self.agent_ids:
                continue
            if (
                "give_item_name" not in proposal_dic
                or "give_item_amount" not in proposal_dic
                or "get_item_name" not in proposal_dic
                or "get_item_amount" not in proposal_dic
            ):
                continue
            give_item_name: str = proposal_dic["give_item_name"]
            give_item_amount: float | int = proposal_dic["give_item_amount"]
            get_item_name: str = proposal_dic["get_item_name"]
            get_item_amount: float | int = proposal_dic["get_item_amount"]
            if give_item_name not in self.item_name2item:
                continue
            if get_item_name not in self.item_name2item:
                continue
            if give_item_amount <= 0 or get_item_amount <= 0:
                continue
            allowed_give_amount: float | int = agent.get_item_amount(give_item_name)
            if give_item_amount > allowed_give_amount:
                continue
            valid_proposals.append(proposal_dic)
        return valid_proposals

    def _add_new_orders_and_proposals(
        self,
        agent_id: int,
        orders: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
    ) -> None:
        """Add the new orders and proposals to the environment.

        Args:
            agent_id (int): agent id of the agent who places the orders and makes the proposals.
            orders (list[dict[str, Any]]): list of order dictionaries.
            proposals (list[dict[str, Any]]): list of proposal dictionaries.

        Note:
            This method generate new Order and SwapProposal instances based on the input order and proposal dictionaries,
            and add them to the pending_orders list and pending_swap_proposals list in the environment, respectively.
            Orders and proposals action example:
            {
                "orders": [
                    {"counterparty_id": int, "counterparty_name": str, "item_name": str, "item_amount": float | int, "ttl": int},
                    ...
                ], # <- corresponds to the orders list in the arguments.
                "proposals": [
                    {"responder_agent_id": int, "responder_agent_name": str, "give_item_name": str, "give_item_amount": float | int, "get_item_name": str, "get_item_amount": float | int, "ttl": int},
                    ...
                ] # <- corresponds to the proposals list in the arguments.
            }
        """
        for order_dic in orders:
            counterparty_id: Optional[int] = order_dic.get("counterparty_id", None)
            counterparty_name: Optional[str] = order_dic.get("counterparty_name", None)
            if counterparty_id is None and counterparty_name is not None:
                counterparty_id = self.agent_name2agent_id[counterparty_name]
            if "item_name" not in order_dic:
                raise ValueError("item_name must be provided in order_dic.")
            assert counterparty_id is not None
            item_name: str = order_dic["item_name"]
            if item_name not in self.item_name2item:
                raise ValueError(
                    f"item_name {item_name} in order_dic is not found in the environment."
                )
            item_amount: float | int = order_dic["item_amount"]
            if "item_amount" not in order_dic:
                raise ValueError("item_amount must be provided in order_dic.")
            ttl: Optional[int] = order_dic.get("ttl", None)
            price: Optional[float] = order_dic.get("price", None)
            new_order: Order = Order(
                agent_id=agent_id,
                counterparty_id=counterparty_id,
                item_name=item_name,
                item_amount=item_amount,
                price=price,
                order_id=self.latest_order_id,
                ttl=ttl,
            )
            order_log: OrderLog = OrderLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                counterparty_id=counterparty_id,
                item_name=item_name,
                item_amount=item_amount,
                price=price,
                order_id=self.latest_order_id,
            )
            self.remember_log(order_log)
            self.event_manager.trigger_events_after_log(log=order_log, env=self)
            if self.logger is not None:
                order_log.read_and_write(logger=self.logger)
            self.pending_orders.append(new_order)
            self.latest_order_id += 1
        for proposal_dic in proposals:
            responder_agent_id: Optional[int] = proposal_dic.get(
                "responder_agent_id", None
            )
            responder_agent_name: Optional[str] = proposal_dic.get(
                "responder_agent_name", None
            )
            if responder_agent_id is None and responder_agent_name is not None:
                responder_agent_id = self.agent_name2agent_id[responder_agent_name]
            elif responder_agent_id is None and responder_agent_name is None:
                raise ValueError(
                    "Either responder_agent_id or responder_agent_name must be provided in proposal_dic."
                )
            assert responder_agent_id is not None
            if "give_item_name" not in proposal_dic:
                raise ValueError("give_item_name must be provided in proposal_dic.")
            if "give_item_amount" not in proposal_dic:
                raise ValueError("give_item_amount must be provided in proposal_dic.")
            if "get_item_name" not in proposal_dic:
                raise ValueError("get_item_name must be provided in proposal_dic.")
            if "get_item_amount" not in proposal_dic:
                raise ValueError("get_item_amount must be provided in proposal_dic.")
            give_item_name: str = proposal_dic["give_item_name"]
            give_item_amount: float | int = proposal_dic["give_item_amount"]
            get_item_name: str = proposal_dic["get_item_name"]
            get_item_amount: float | int = proposal_dic["get_item_amount"]
            ttl = proposal_dic.get("ttl", None)
            new_proposal: SwapProposal = SwapProposal(
                proposer_agent_id=agent_id,
                responder_agent_id=responder_agent_id,
                give_item_name=give_item_name,
                give_item_amount=give_item_amount,
                get_item_name=get_item_name,
                get_item_amount=get_item_amount,
                proposal_id=self.latest_proposal_id,
                ttl=ttl,
            )
            proposal_log: ProposalLog = ProposalLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                proposal_id=self.latest_proposal_id,
                proposer_agent_id=agent_id,
                responder_agent_id=responder_agent_id,
                give_item_name=give_item_name,
                give_item_amount=give_item_amount,
                get_item_name=get_item_name,
                get_item_amount=get_item_amount,
            )
            self.remember_log(proposal_log)
            self.event_manager.trigger_events_after_log(log=proposal_log, env=self)
            if self.logger is not None:
                proposal_log.read_and_write(logger=self.logger)
            self.pending_swap_proposals.append(new_proposal)
            self.latest_proposal_id += 1

    def _check_follow_unfollow(
        self,
        agent_id: int,
        follow_agent_id: Optional[int],
        unfollow_agent_id: Optional[int],
    ) -> tuple[Optional[int], Optional[int]]:
        """Check whether the follow and unfollow actions are valid.

        Args:
            agent_id (int): agent id of the agent who performs the follow and unfollow actions.
            follow_agent_id (int, optional): agent id of the target agent to follow. Optional, default None.
            unfollow_agent_id (int, optional): agent id of the target agent to unfollow. Optional, default None.

        Returns:
            tuple[Optional[int], Optional[int]]: A tuple containing the valid follow_agent_id and unfollow_agent_id, or None if invalid.

        Note:
            Checked conditions:
            - follow_agent_id and unfollow_agent_id cannot be the same.
            - follow_agent_id and unfollow_agent_id must be existing agent ids in the environment.
            - If follow_agent_id is already followed by the agent, it cannot be followed again.
            - If unfollow_agent_id is not followed by the agent, it cannot be unfollowed.
            - The number of follows after performing the follow and unfollow actions cannot exceed
                the follow cap. See also: econsimulacra.social_networks.base.SocialNetwork.follow_cap
        """
        valid_follow_agent_id: Optional[int] = follow_agent_id
        valid_unfollow_agent_id: Optional[int] = unfollow_agent_id
        if agent_id == follow_agent_id:
            valid_follow_agent_id = None
        if follow_agent_id is not None:
            if follow_agent_id not in self.agent_ids:
                valid_follow_agent_id = None
            if follow_agent_id in self.social_network.get_follows(agent_id=agent_id):
                valid_follow_agent_id = None
        if agent_id == unfollow_agent_id:
            valid_unfollow_agent_id = None
        if unfollow_agent_id is not None:
            if unfollow_agent_id not in self.agent_ids:
                valid_unfollow_agent_id = None
            if unfollow_agent_id not in self.social_network.get_follows(
                agent_id=agent_id
            ):
                valid_unfollow_agent_id = None
        allowed_num_follows: Optional[int] = (
            self.social_network.get_allowed_num_follows(agent_id=agent_id)
        )
        if allowed_num_follows is not None:
            if valid_unfollow_agent_id is not None:
                allowed_num_follows += 1
            if valid_follow_agent_id is not None:
                allowed_num_follows -= 1
            if allowed_num_follows < 0:
                valid_follow_agent_id = None
        return valid_follow_agent_id, valid_unfollow_agent_id

    def _act_in_social_network(
        self,
        agent_id: int,
        tweet: Optional[str] = None,
        follow_agent_id: Optional[int] = None,
        unfollow_agent_id: Optional[int] = None,
    ) -> None:
        """Apply the actions in the social network to the environment.

        Args:
            agent_id (int): agent id of the agent who performs the actions in the social network.
            tweet (str, optional): the message to tweet. Optional, default None.
            follow_agent_id (int, optional): agent id of the target agent to follow. Optional, default None.
            unfollow_agent_id (int, optional): agent id of the target agent to unfollow. Optional, default None.

        Note:
            Action in social network example:
            {
                "tweet": str, # <- corresponds to the tweet argument
                "follow": int, # <- corresponds to the follow_agent_id argument
                "unfollow": int # <- corresponds to the unfollow_agent_id argument
            }
            See also:
            - econsimulacra.social_networks.base.SocialNetwork.tweet
            - econsimulacra.social_networks.base.SocialNetwork.follow_agent
            - econsimulacra.social_networks.base.SocialNetwork.unfollow_agent
        """
        if tweet is not None and len(tweet) > 0:
            self.social_network.tweet(agent_id=agent_id, message=tweet)
            tweet_log: TweetLog = TweetLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                message=tweet,
                num_follows=self.social_network.get_num_follows(agent_id=agent_id),
                num_followers=self.social_network.get_num_followers(agent_id=agent_id),
            )
            self.remember_log(tweet_log)
            self.event_manager.trigger_events_after_log(log=tweet_log, env=self)
            if self.logger is not None:
                tweet_log.read_and_write(logger=self.logger)
        if unfollow_agent_id is not None:
            self.social_network.unfollow_agent(
                agent_id=agent_id, target_agent_id=unfollow_agent_id
            )
            unfollow_log: UnfollowLog = UnfollowLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                target_agent_id=unfollow_agent_id,
                num_follows=self.social_network.get_num_follows(agent_id=agent_id),
                num_followers=self.social_network.get_num_followers(agent_id=agent_id),
            )
            self.remember_log(unfollow_log)
            self.event_manager.trigger_events_after_log(log=unfollow_log, env=self)
            if self.logger is not None:
                unfollow_log.read_and_write(logger=self.logger)
        if follow_agent_id is not None:
            self.social_network.follow_agent(
                agent_id=agent_id, target_agent_id=follow_agent_id
            )
            follow_log: FollowLog = FollowLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                target_agent_id=follow_agent_id,
                num_follows=self.social_network.get_num_follows(agent_id=agent_id),
                num_followers=self.social_network.get_num_followers(agent_id=agent_id),
            )
            self.remember_log(follow_log)
            self.event_manager.trigger_events_after_log(log=follow_log, env=self)
            if self.logger is not None:
                follow_log.read_and_write(logger=self.logger)

    def _process_orders_and_proposals(self) -> None:
        """Process the pending orders and proposals in the environment by executing the valid ones.

        Note:
            See also:
            - econsimulacra.envs.order.Order
            - econsimulacra.envs.order.SwapProposal
            - econsimulacra.agents.base.Agent.exchange_goods
        """
        for order in self.pending_orders:
            if order.accepted_amount > 0:
                agent_id: int = order.agent_id
                counterparty_id: int = order.counterparty_id
                agent: Agent = self.agent_id2agent[agent_id]
                counterparty: Agent = self.agent_id2agent[counterparty_id]
                accepted_amount: float | int = order.accepted_amount
                item_name: str = order.item_name
                item: Item = self.item_name2item[item_name]
                total_price: float = accepted_amount * max(
                    0 if order.price is None else order.price, item.price
                )
                agent.exchange_goods(
                    get_item_name=item_name,
                    get_item_amount=accepted_amount,
                    give_item_name=self.cash_name,
                    give_item_amount=total_price,
                )
                counterparty.exchange_goods(
                    get_item_name=self.cash_name,
                    get_item_amount=total_price,
                    give_item_name=item_name,
                    give_item_amount=accepted_amount,
                )
                order.execute()
        for proposal in self.pending_swap_proposals:
            if proposal.accept is not None:
                if proposal.accept:
                    proposer_id: int = proposal.proposer_agent_id
                    responder_id: int = proposal.responder_agent_id
                    proposer: Agent = self.agent_id2agent[proposer_id]
                    responder: Agent = self.agent_id2agent[responder_id]
                    proposer.exchange_goods(
                        get_item_name=proposal.get_item_name,
                        get_item_amount=proposal.get_item_amount,
                        give_item_name=proposal.give_item_name,
                        give_item_amount=proposal.give_item_amount,
                    )
                    responder.exchange_goods(
                        get_item_name=proposal.give_item_name,
                        get_item_amount=proposal.give_item_amount,
                        give_item_name=proposal.get_item_name,
                        give_item_amount=proposal.get_item_amount,
                    )

    def _check_reactions(
        self, agent_id: int, reactions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check whether the reactions are valid.

        Args:
            agent_id (int): agent id of the agent who makes the reactions.
            reactions (list[dict[str, Any]]): list of reaction dictionaries.

        Returns:
            list[dict[str, Any]]: list of valid reaction dictionaries.

        Note:
            Checked conditions:
            - Each reaction must include "kind" key, and its value must be either "order" or "proposal".
            - Reacting to the same order or proposal more than once in the same step is not allowed.
            - For "order" reactions:
                - "id" and "accept_amount" keys must be included.
                - "id" must correspond to an existing pending order where the agent is the counterparty.
                - "accept_amount" must be non-negative and not exceed the order's item amount.
                - The agent must have enough of the item to fulfill the accept_amount.
            - For "proposal" reactions:
                - "id" and "accept" keys must be included.
                - "id" must correspond to an existing pending proposal where the agent is the responder.
                - If "accept" is True, the agent must have enough of the item to fulfill the proposal's get_item_amount.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        holding_amount: float | int
        reacted_order_ids: list[int] = []
        reacted_proposal_ids: list[int] = []
        valid_reactions: list[dict[str, Any]] = []
        for reaction in reactions:
            if "kind" not in reaction:
                continue
            kind: str = reaction["kind"]
            if kind == "order":
                if "id" not in reaction or "accept_amount" not in reaction:
                    continue
                order_id: int = reaction["id"]
                accept_amount: float | int = reaction["accept_amount"]
                if order_id in reacted_order_ids:
                    continue
                reacted_order_ids.append(order_id)
                if accept_amount < 0:
                    continue
                find_corresponding_order: bool = False
                order: Order
                for order in self.pending_orders:
                    if order.order_id == order_id and order.counterparty_id == agent_id:
                        if accept_amount > order.item_amount:
                            continue
                        holding_amount = agent.get_item_amount(order.item_name)
                        if accept_amount > holding_amount:
                            continue
                        find_corresponding_order = True
                        break
                if not find_corresponding_order:
                    continue
                valid_reactions.append(reaction)
            elif kind == "proposal":
                if "id" not in reaction or "accept" not in reaction:
                    continue
                proposal_id: int = reaction["id"]
                if proposal_id in reacted_proposal_ids:
                    continue
                reacted_proposal_ids.append(proposal_id)
                accept: bool = reaction["accept"]
                find_corresponding_proposal: bool = False
                proposal: SwapProposal
                for proposal in self.pending_swap_proposals:
                    if (
                        proposal.proposal_id == proposal_id
                        and proposal.responder_agent_id == agent_id
                    ):
                        if accept:
                            holding_amount = agent.get_item_amount(
                                proposal.get_item_name
                            )
                            if proposal.get_item_amount > holding_amount:
                                continue
                        find_corresponding_proposal = True
                        break
                if not find_corresponding_proposal:
                    continue
                valid_reactions.append(reaction)
            else:
                continue
        return valid_reactions

    def _process_reactions(
        self, agent_id: int, reactions: list[dict[str, Any]]
    ) -> None:
        """Apply the reactions to the environment.

        Args:
            agent_id (int): agent id of the agent who makes the reactions.
            reactions (list[dict[str, Any]]): list of reaction dictionaries.

        Note:
            Reaction action example:
            {
                "reactions": [
                    {"kind": "order", "id": int, "accept_amount": float | int},
                    {"kind": "proposal", "id": int, "accept": bool},
                    ...
                ] # <- corresponds to the reactions list in the arguments.
            }
            See also:
            - econsimulacra.envs.order.Order.react
            - econsimulacra.envs.order.SwapProposal.react
        """
        for reaction in reactions:
            if "kind" not in reaction:
                raise ValueError("Each reaction must include 'kind' key.")
            kind: Literal["order", "proposal"] = reaction["kind"]
            if "id" not in reaction:
                raise ValueError("Each reaction must include 'id' key.")
            if kind == "order":
                order_id: int = reaction["id"]
                if "accept_amount" not in reaction:
                    raise ValueError("Order reaction must include 'accept_amount' key.")
                accept_amount: float | int = reaction["accept_amount"]
                for order in self.pending_orders:
                    if order.order_id == order_id:
                        if order.counterparty_id != agent_id:
                            raise ValueError(
                                f"Agent {agent_id} cannot react to order {order_id}."
                            )
                        order.react(amount=accept_amount)
                        order_reaction_log: OrderReactionLog = OrderReactionLog(
                            time=self.get_time(),
                            time_step=self.get_time_step(),
                            agent_id=order.agent_id,
                            counterparty_id=agent_id,
                            item_name=order.item_name,
                            item_amount=order.item_amount,
                            price=max(
                                0 if order.price is None else order.price,
                                self.item_name2item[order.item_name].price,
                            ),
                            order_id=order.order_id,
                            accept_amount=accept_amount,
                        )
                        self.remember_log(order_reaction_log)
                        self.event_manager.trigger_events_after_log(
                            log=order_reaction_log, env=self
                        )
                        if self.logger is not None:
                            order_reaction_log.read_and_write(logger=self.logger)
            elif kind == "proposal":
                proposal_id: int = reaction["id"]
                if "accept" not in reaction:
                    raise ValueError("Proposal reaction must include 'accept' key.")
                accept: bool = reaction["accept"]
                for proposal in self.pending_swap_proposals:
                    if (
                        proposal.proposal_id == proposal_id
                        and proposal.responder_agent_id == agent_id
                    ):
                        proposal.react(accept=accept)
                        proposal_reaction_log: ProposalReactionLog = (
                            ProposalReactionLog(
                                time=self.get_time(),
                                time_step=self.get_time_step(),
                                proposal_id=proposal.proposal_id,
                                proposer_agent_id=proposal.proposer_agent_id,
                                responder_agent_id=proposal.responder_agent_id,
                                give_item_name=proposal.give_item_name,
                                give_item_amount=proposal.give_item_amount,
                                get_item_name=proposal.get_item_name,
                                get_item_amount=proposal.get_item_amount,
                                accept=accept,
                            )
                        )
                        self.remember_log(proposal_reaction_log)
                        self.event_manager.trigger_events_after_log(
                            log=proposal_reaction_log, env=self
                        )
                        if self.logger is not None:
                            proposal_reaction_log.read_and_write(logger=self.logger)
            else:
                raise ValueError(
                    f"Unknown reaction kind: {kind}. Must be either 'order' or 'proposal'."
                )

    def _check_set_prices(
        self, agent_id: int, set_prices: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check whether the set_prices are valid.

        Args:
            agent_id (int): agent id of the agent who sets the prices.
            set_prices (list[dict[str, Any]]): list of set_price dictionaries.

        Returns:
            list[dict[str, Any]]: list of valid set_price dictionaries.

        Note:
            Checked conditions:
            - The agent cannot set prices if it is a household (i.e., in household_ids).
            - item_name in each set_price must be an existing item in the environment.
            - The agent must have non-negative inventory of the item to set its price.
            - price in each set_price must be non-negative.
        """
        valid_set_prices: list[dict[str, Any]] = []
        if agent_id in self.household_ids:
            return []
        agent: Agent = self.agent_id2agent[agent_id]
        for set_price in set_prices:
            if "item_name" not in set_price or "price" not in set_price:
                continue
            item_name: str = set_price["item_name"]
            price: float = set_price["price"]
            if item_name not in self.item_name2item:
                continue
            if agent.get_item_amount(item_name) <= 0:
                continue
            if price < 0:
                continue
            valid_set_prices.append(set_price)
        return valid_set_prices

    def _set_prices(self, agent_id: int, set_prices: list[dict[str, Any]]) -> None:
        """Apply the set_prices action to the environment.

        Args:
            agent_id (int): agent id of the agent who sets the prices.
            set_prices (list[dict[str, Any]]): list of set_price dictionaries.

        Note:
            Set prices action example:
            {
                "set_prices": [
                    {"item_name": str, "price": float},
                    ...
                ] # <- corresponds to the set_prices list in the arguments.
            }
        """
        for set_price in set_prices:
            if "item_name" not in set_price:
                raise ValueError("Each set_price must include 'item_name' key.")
            if "price" not in set_price:
                raise ValueError("Each set_price must include 'price' key.")
            item_name: str = set_price["item_name"]
            price: float = set_price["price"]
            item: Item = self.item_name2item[item_name]
            old_price: float = item.get_price()
            item.set_price(price=price, set_by=agent_id)
            change_price_log: ChangePriceLog = ChangePriceLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                item_name=item_name,
                old_price=old_price,
                new_price=price,
            )
            self.remember_log(change_price_log)
            self.event_manager.trigger_events_after_log(log=change_price_log, env=self)
            if self.logger is not None:
                change_price_log.read_and_write(logger=self.logger)

    def _update_time(self) -> None:
        """Update the time in the environment by 1 step"""
        for order in self.pending_orders:
            order.update_time()
        for proposal in self.pending_swap_proposals:
            proposal.update_time()
        self._time += 1

    def _remove_expired_orders_and_proposals(self) -> None:
        """Remove the expired orders and proposals from the environment.

        Note:
            The Order and SwapProposal instances whose time to live (ttl)
            has reached 0 are considered expired and removed from the environment.
            See also:
            - econsimulacra.envs.order.Order.ttl
            - econsimulacra.envs.order.Order.is_expired
            - econsimulacra.envs.order.Order.is_fulfilled
            - econsimulacra.envs.order.SwapProposal.ttl
            - econsimulacra.envs.order.SwapProposal.is_expired
            - econsimulacra.envs.order.SwapProposal.is_fulfilled
        """
        _pending_orders: list[Order] = []
        for order in self.pending_orders:
            if order.is_expired():
                expired_order_log: OrderExpirationLog = OrderExpirationLog(
                    time=self.get_time(),
                    time_step=self.get_time_step(),
                    order_id=order.order_id,
                )
                self.remember_log(expired_order_log)
                self.event_manager.trigger_events_after_log(
                    log=expired_order_log, env=self
                )
                if self.logger is not None:
                    expired_order_log.read_and_write(logger=self.logger)
            elif order.is_fulfilled():
                continue
            else:
                _pending_orders.append(order)
        self.pending_orders = _pending_orders
        _pending_swap_proposals: list[SwapProposal] = []
        for proposal in self.pending_swap_proposals:
            if proposal.is_expired():
                expired_proposal_log: ProposalExpirationLog = ProposalExpirationLog(
                    time=self.get_time(),
                    time_step=self.get_time_step(),
                    proposal_id=proposal.proposal_id,
                )
                self.remember_log(expired_proposal_log)
                self.event_manager.trigger_events_after_log(
                    log=expired_proposal_log, env=self
                )
                if self.logger is not None:
                    expired_proposal_log.read_and_write(logger=self.logger)
            elif proposal.is_fulfilled():
                continue
            else:
                _pending_swap_proposals.append(proposal)
        self.pending_swap_proposals = _pending_swap_proposals

    def evaluate_agent_state(self, agent_id: int) -> None:
        """Evaluate the state of the agent. Generate agent evaluation log.

        Args:
            agent_id (int): agent id of the agent to evaluate.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        wealth: float = self._calculate_wealth(agent.get_inventory())
        self.agent_id2wealth[agent_id] = wealth
        log: StateEvaluationLog = StateEvaluationLog(
            time=self.get_time(),
            time_step=self.get_time_step(),
            agent_id=agent_id,
            wealth=wealth,
            relative_wealth=self.calculate_relative_wealth(agent_id=agent_id),
            buying_power=self.calculate_buying_power(agent_id=agent_id),
            inventory_dic=agent.get_inventory(),
            persona_dic=self.get_persona(agent_id=agent_id),
        )
        self.remember_log(log)
        self.event_manager.trigger_events_after_log(log=log, env=self)
        if self.logger is not None:
            log.read_and_write(logger=self.logger)

    def _calculate_wealth(self, inventory_dic: dict[str, float | int]) -> float:
        """Calculate the wealth based on the inventory_dic.

        Args:
            inventory_dic (dict[str, float | int]): the inventory dictionary to calculate the wealth.
                The keys are item names, and the values are the corresponding item amounts.

        Returns:
            float: the calculated wealth.
        """
        wealth: float = 0
        for item_name, item_amount in inventory_dic.items():
            if item_name in self.item_name2item:
                item: Item = self.item_name2item[item_name]
                wealth += item_amount * item.get_price()
        return wealth

    def calculate_relative_wealth(self, agent_id: int) -> Optional[float]:
        """Calculate the relative wealth of the agent compared to other agents.

        Args:
            agent_id (int): agent id of the agent to calculate the relative wealth.

        Returns:
            relative_wealth (float, optional): the calculated relative wealth.
                Returns None if the agent is not a household agent.

        Note:
            See also: econsimulacra.logs.StateEvaluationLog
        """
        if agent_id not in self.household_ids:
            return None
        household_wealth_arr: NDArray[np.float64] = np.array(
            [
                self.agent_id2wealth.get(agent_id, 0.0)
                for agent_id in self.agent_ids
                if agent_id in self.household_ids
            ],
            dtype=np.float64,
        )
        if len(household_wealth_arr) == 0:
            raise ValueError(
                "No household agents found in agent_id2wealth, "
                f"even though {agent_id} is in household_ids."
            )
        avg_wealth: float = float(np.mean(household_wealth_arr))
        std_wealth: float = float(np.std(household_wealth_arr))
        agent_wealth: float = self.agent_id2wealth[agent_id]
        return (agent_wealth - avg_wealth) / std_wealth if std_wealth > 0 else 0.0

    def calculate_buying_power(self, agent_id: int) -> float:
        """Calculate the buying power of the agent based on its inventory and the item prices.

        Args:
            agent_id (int): agent id of the agent to calculate the buying power.

        Returns:
            buying_power (float): the calculated buying power.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        inventory_dic: dict[str, float | int] = agent.get_inventory()
        cash_amount: float | int = inventory_dic.get(self.cash_name, 0)
        weighted_price: float = self._calc_weighted_price()
        buying_power: float = cash_amount / weighted_price if weighted_price > 0 else 0
        return buying_power

    def _calc_weighted_price(self) -> float:
        total_weight: float = sum(
            item.weight_in_basket for item in self.item_name2item.values()
        )
        weighted_price: float = 0
        for item in self.item_name2item.values():
            weight: float = (
                item.weight_in_basket / total_weight if total_weight > 0 else 0
            )
            price: float = item.get_price()
            weighted_price += weight * price
        return weighted_price

    def get_observations(self, agent_id: int) -> ObsT:
        """Get the observations for the agent with the given agent_id.

        Args:
            agent_id (int): agent id of the agent to get the observations for.

        Returns:
            ObsT: the observations for the agent.

        Note:
            The observations are provided by the registered observation providers based on the agent's request.
            There are three types of observations:

            1. General observations provided to all agents, such as time and self position.
            2. Additional observations provided only to agents with ``is_rich_info_allowed=True``,
               such as ``item_name2price``.
            3. Additional observations provided only to co-located agents, such as ``others_inventory``.

            The agent can request the observations by specifying the keys of the desired observations.
            If the agent requests "all", all available observations will be provided.
        """
        if not hasattr(self, "_obs_providers"):
            self._obs_providers: dict[str, ObsProvider] = (
                self._build_observation_registry()
            )
        if not hasattr(self, "_obs4allowed_agents_providers"):
            self._obs4allowed_agents_providers: dict[str, ObsProvider] = (
                self._build_observation4allowed_agents_registry()
            )
        co_located_agents: set[int] = self.grid_space.get_colocated_agents(
            agent_id=agent_id
        )
        self._obs4co_located_agents_providers: dict[
            str, ObsProviderFromCoLocatedAgents
        ] = self._build_observation4co_located_agents_registry(
            co_located_agents=co_located_agents
        )
        obs_providers: dict[str, ObsProvider | ObsProviderFromCoLocatedAgents] = {}
        obs_providers.update(self._obs_providers)
        agent: Agent = self.agent_id2agent[agent_id]
        if agent.is_rich_info_allowed:
            obs_providers.update(self._obs4allowed_agents_providers)
        if len(co_located_agents) > 0:
            obs_providers.update(self._obs4co_located_agents_providers)
        keys_to_request: list[str] = agent.request_obs()
        if "all" in keys_to_request:
            keys_to_request = list(obs_providers.keys())
        observation: dict[str, Any] = {}
        for key in keys_to_request:
            if key not in obs_providers:
                raise ValueError(f"Unknown observation key requested: {key}.")
            provider: ObsProvider | ObsProviderFromCoLocatedAgents = obs_providers[key]
            obs = provider.get_obs(agent_id=agent_id)
            observation[key] = obs
            log: ObsLog = ObsLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                obs_type=key,
                obs=obs,
            )
            self.event_manager.trigger_events_after_log(log=log, env=self)
            self.remember_log(log)
            if self.logger is not None:
                log.read_and_write(logger=self.logger)
        return observation  # type: ignore

    def _build_observation_registry(self) -> dict[str, ObsProvider]:
        """Build the registry of general observation providers available to all agents.

        Returns:
            dict[str, ObsProvider]: Dispatch table for general observation providers.

        Note:
            Custom observation provider can be added by creating a new ObsProvider class
            and registering it in this method.
            See also:
            econsimulacra.envs.obs_providers
        """
        return {
            "time": TimeProvider(env=self),
            "timedelta": TimeDeltaProvider(env=self),
            "self_agent_id": SelfIDProvider(env=self),
            "self_name": SelfNameProvider(env=self),
            "self_is_household": SelfIsHouseholdProvider(env=self),
            "memory": MemoryProvider(env=self),
            "self_pos": SelfPosProvider(env=self),
            "self_init_pos": SelfInitPosProvider(env=self),
            "self_is_moving": SelfIsMovingProvider(env=self),
            "self_destination": SelfDestinationProvider(env=self),
            "others_pos": OthersPosProvider(env=self),
            "self_salary": SelfSalaryProvider(env=self),
            "self_inventory": SelfInventoryProvider(env=self),
            "self_tweet": SelfTweetProvider(env=self),
            "follow_cap": FollowCapProvider(env=self),
            "num_followers": NumFollowersProvider(env=self),
            "num_follows": NumFollowsProvider(env=self),
            "visible_tl": VisibleTLProvider(env=self),
            "recommended_follows": RecommendedFollowsProvider(env=self),
            "incoming_orders": IncomingOrdersProvider(env=self),
            "incoming_proposals": IncomingSwapProposalsProvider(env=self),
        }

    def _build_observation4allowed_agents_registry(self) -> dict[str, ObsProvider]:
        """Build the registry of additional observation providers available only to agents who are allowed to have rich information.

        Returns:
            dict[str, ObsProvider]: Dispatch table for additional observation providers for allowed agents.
        """
        return {
            "item_name2price": ItemName2PriceProvider(env=self),
        }

    def _build_observation4co_located_agents_registry(
        self, co_located_agents: set[int]
    ) -> dict[str, ObsProviderFromCoLocatedAgents]:
        """Build the registry of additional observation providers available only to agents who are co-located with other agents in the same grid cell.

        Returns:
            dict[str, ObsProviderFromCoLocatedAgents]: Dispatch table for additional observation providers for co-located agents.
        """
        return {
            "others_inventory": OthersInventoriesProvider(
                env=self, co_located_agents=co_located_agents
            ),
        }
