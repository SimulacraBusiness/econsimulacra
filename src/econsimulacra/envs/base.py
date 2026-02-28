from abc import ABC
from ..agents import Agent
from ..sim_utils import find_class
from ..items import Item
from ..logs import AgentGenerationLog
from ..logs import SpaceAssignLog
from ..logs import MoveLog
from ..logs import ConsumptionLog
from ..logs import OrderLog
from ..logs import ProposalLog
from ..logs import OrderReactionLog
from ..logs import ProposalReactionLog
from ..logs import ChangePriceLog
from ..logs import TweetLog
from ..logs import FollowLog
from ..logs import UnfollowLog
from ..logs import Logger
from .obs_providers import ObsProvider
from .obs_providers import ObsProviderForCoLocatedAgents
from .obs_providers import TimeProvider
from .obs_providers import TimeDeltaProvider
from .obs_providers import SelfIDProvider
from .obs_providers import SelfNameProvider
from .obs_providers import SelfPosProvider
from .obs_providers import SelfInitPosProvider
from .obs_providers import SelfIsMovingProvider
from .obs_providers import SelfDestinationProvider
from .obs_providers import OthersPosProvider
from .obs_providers import SelfInventoryProvider
from .obs_providers import SelfTweetProvider
from .obs_providers import FollowCapProvider
from .obs_providers import NumFollowersProvider
from .obs_providers import NumFollowsProvider
from .obs_providers import VisibleTLProvider
from .obs_providers import RecommendedFollowsProvider
from .obs_providers import IncomingOrdersProvider
from .obs_providers import IncomingSwapProposalsProvider
from .obs_providers import ItemName2PriceProvider
from .obs_providers import OthersInventoriesProvider
from .order import Order
from .order import SwapProposal
import random
from random import Random
from .social_network import SocialNetwork
from .space import GridSpace
from .time_translator import TimeTranslator
from typing import Any
from typing import Generic
from typing import Literal
from typing import Optional
from typing import Type
from typing import TypeVar

ObsT = TypeVar("ObsT")


class Environment(ABC, Generic[ObsT]):
    def __init__(
        self,
        config: dict[str, Any],
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): Environment configuration dictionary.

        Note:
            config example:
            {
                "simulation": {
                    "numSteps": int,
                },
                "environment": {
                    "gridSpace": [int, ...],
                    "followCap": int, # Optional, default is no limit
                    "cashName": str,
                    "agents": ["Household", "Retailer", "Restaurant", ...],
                    "items": ["Yen", "Rice", ...],
                    "service": ["promptBuilder", "llmClient", "timeTranslator", "personaBuilder"], # Optional, default []
                }
                "Household": {
                    "type": "LLMAgent",
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
                    "type": "Item",
                    "initialPrice": float,
                },
                "Rice": {
                    "type": "Item",
                    "initialPrice": float,
                },
                "promptBuilder": {
                    "type": "PromptBuilder", # Optional, default is the same as the service name
                    ...
                },
                "llmClient": {
                    "type": "OpenAIClient", # Optional, default is the same as the service name
                    "api_key": str, # Optional if OPENAI_API_KEY environment variable is set
                    "json_schema_path": str, # Optional, path to a custom JSON schema file for structured generation
                    "modify_schema": bool, # Optional, whether to modify the default JSON schema based on config
                    "gridSpace": [int, ...], # Optional, only needed if modify_schema is True. Must be the same as environment.gridSpace
                    "items": [str, ...], # Optional, only needed if modify_schema is True. Must include all item names used in the environment.items
                    "numAgents": int, # Optional, only needed if modify_schema is True. Must be the total number of agents in the environment.
                },
                "timeTranslator": {
                    "type": "TimeTranslator", # Optional, default is the same as the service name
                    "numSteps": int, # must be the same as simulation.numSteps
                    "startDatetime": str, # "%Y-%m-%d %H:%M:%S"
                    "endDatetime": str, # "%Y-%m-%d %H:%M:%S"
                },
                "personaBuilder": {
                    "type": "Big5PersonaBuilder", # Optional, default is the same as the service name
                    ...
                },
                ...,
            }
        """
        env_config: dict[str, Any] = config.get("environment", {})
        if "gridSpace" not in env_config:
            raise ValueError("Environment config must include 'gridSpace' key.")
        self.space_size: tuple[int, ...] = env_config["gridSpace"]
        if "cashName" not in env_config:
            raise ValueError("Simulation config must include 'cashName' key.")
        self.cash_name: str = env_config["cashName"]
        self.config: dict[str, Any] = config
        self.seed: Optional[int] = None
        self.prng: Random = random.Random()
        self.registered_classes: list[Type] = []
        self.logger: Optional[Logger] = logger
        self._time: int = -1
        self.service_dic: dict[str, Any] = {}

    def get_time_translator(self) -> Optional[TimeTranslator]:
        for provider in self.service_dic.values():
            if isinstance(provider, TimeTranslator):
                return provider
        return None

    def get_time(self) -> int | str:
        time_translator: Optional[TimeTranslator] = self.get_time_translator()
        if time_translator is not None:
            return time_translator.step_to_datetime(self._time)
        return self._time

    def get_time_step(self) -> int:
        return self._time

    def get_timedelta(self) -> int | str:
        time_translator: Optional[TimeTranslator] = self.get_time_translator()
        if time_translator is not None:
            return time_translator.get_timedelta()
        return 1

    def register_classes(self, class_list: list[Type]) -> None:
        self.registered_classes.extend(class_list)

    def reset(self, seed: Optional[int]) -> None:
        self._set_invalid_action_dic()
        if self.logger is not None:
            self.logger.clear()
        if seed is not None:
            self.seed = seed
            self.prng.seed(seed)
        self.grid_space: GridSpace = GridSpace(space_size=self.space_size)
        assert "environment" in self.config, "Config must include 'environment' key."
        assert isinstance(self.config["environment"], dict), (
            "'environment' key must be a dictionary."
        )
        follow_cap: Optional[int] = self.config["environment"].get("followCap", None)
        self.social_network: SocialNetwork = SocialNetwork(follow_cap=follow_cap)
        service_provider_keys: list[str] = self.config["environment"].get("service", [])
        self._generate_service_providers(service_provider_keys=service_provider_keys)
        assert "agents" in self.config["environment"], (
            "Environment config must include 'agents' key."
        )
        agent_keys: list[str] = self.config["environment"]["agents"]
        self._generate_agents(agent_keys=agent_keys)
        assert "items" in self.config["environment"], (
            "Environment config must include 'items' key."
        )
        item_keys: list[str] = self.config["environment"]["items"]
        self._generate_items(item_keys=item_keys)
        self.pending_orders: list[Order] = []
        self.pending_swap_proposals: list[SwapProposal] = []
        self.latest_order_id: int = 0
        self.latest_proposal_id: int = 0
        if self.logger is not None:
            self.logger.process_logs()
        self._time = 0

    def _set_invalid_action_dic(self) -> None:
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

    def _generate_service_providers(self, service_provider_keys: list[str]) -> None:
        """generate service providers.

        Args:
            service_provider_keys (list[str]): name list of service provider types to be generated.
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
                config=service_provider_config, prng=self.prng
            )
            self.service_dic[service_provider_key] = service_provider_instance

    def _generate_agents(self, agent_keys: list[str]) -> None:
        """generate agents and place them in the grid space.

        Args:
            agent_types (list[str]): name list of agent types to be generated.
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
                log: AgentGenerationLog = AgentGenerationLog(
                    agent_id=current_agent_id,
                    time=self.get_time(),
                    time_step=self.get_time_step(),
                    agent_type=agent_instance.agent_type,
                    agent_name=agent_name,
                    inventory_dic=agent_instance.inventory_dic.copy(),
                )
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
                self.social_network.add_agent(current_agent_id)
                current_agent_id += 1

    def _generate_items(self, item_keys: list[str]) -> None:
        """generate items.

        Args:
            item_keys (list[str]): name list of items to be generated.
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

    def get_total_amount(self, item_name: str) -> float | int:
        if item_name not in self.item_name2item:
            raise ValueError(f"Item name {item_name} not found in the environment.")
        total_amount: float | int = 0
        for agent in self.agent_id2agent.values():
            total_amount += agent.get_item_amount(item_name=item_name)
        return total_amount

    def _assign_agent_to_space(
        self, agent_id: int, coords: Optional[tuple[int, ...]] = None
    ) -> None:
        if coords is None:
            coords = tuple(
                self.prng.randint(0, self.space_size[dim] - 1)
                for dim in range(len(self.space_size))
            )
        self.grid_space.place_agent(agent_id=agent_id, pos=coords)
        self.agent_id2initial_coords[agent_id] = coords
        log: SpaceAssignLog = SpaceAssignLog(agent_id=agent_id, pos=coords)
        if self.logger is not None:
            log.read_and_write(logger=self.logger)

    def step(self, all_actions_dic: dict[int, dict[str, Any]]) -> None:
        """execute one step of the environment.

        Args:
            all_actions_dic (dict[int, dict[str, Any]]): _description_
        """
        for agent_id, action_dic in all_actions_dic.items():
            self.apply_action_to_env(
                agent_id=agent_id,
                action_dic=action_dic,
            )
        self._process_orders_and_proposals()
        self._update_time()
        self._remove_expired_orders_and_proposals()
        if self.logger is not None:
            self.logger.process_logs()

    def apply_action_to_env(self, agent_id: int, action_dic: dict[str, Any]) -> None:
        """
        action_dic example:
        {
            "move": list[int, ...] | str, # destination coordinates or destination name. Optional, default None
            "consumptions": [
                {"item_name": str, "item_amount": float | int},
                ...
            ], # Optional, default []
            "orders": [
                {"counterparty_id": int, "counterparty_name": str, "item_name": str, "item_amount": float | int, "ttl": int},
                ...
            ] , # Optional, default []
            "proposals": [
                {"responder_agent_id": int, "responder_agent_name": str, "give_item_name": str, "give_item_amount": float | int, "get_item_name": str, "get_item_amount": float | int, "ttl": int},
                ...
            ] , # Optional, default []
            "reactions": [
                {"kind": "order", "id": int, "accept_amount": float | int},
                {"kind": "proposal", "id": int, "accept": bool},
                ...
            ], # Optional, default []
            "set_prices": [
                {"item_name": str, "price": float},
                ...
            ], # Optional, default []
            "tweet": str, # Optional, default None
            "follow": int, # agent_id to follow. Optional, default None
            "unfollow": int, # agent_id to unfollow. Optional, default None
        }
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
        consumptions_allowed: bool = self._check_consumptions(
            agent_id=agent_id, consumptions=consumptions
        )
        if not consumptions_allowed:
            self.invalid_action_dic["consumptions"] += 1
            consumptions = []
        self._consume_items(
            agent_id=agent_id,
            consumptions=consumptions,
        )
        orders: list[dict[str, Any]] = action_dic.get("orders", [])
        orders_allowed: bool = self._check_orders(agent_id=agent_id, orders=orders)
        if not orders_allowed:
            self.invalid_action_dic["orders"] += 1
            orders = []
        proposals: list[dict[str, Any]] = action_dic.get("proposals", [])
        proposals_allowed: bool = self._check_proposals(
            agent_id=agent_id, proposals=proposals
        )
        if not proposals_allowed:
            self.invalid_action_dic["proposals"] += 1
            proposals = []
        self._add_new_orders_and_proposals(
            agent_id=agent_id,
            orders=orders,
            proposals=proposals,
        )
        reactions: list[dict[str, Any]] = action_dic.get("reactions", [])
        reactions_allowed: bool = self._check_reactions(
            agent_id=agent_id, reactions=reactions
        )
        if not reactions_allowed:
            self.invalid_action_dic["reactions"] += 1
            reactions = []
        self._process_reactions(
            agent_id=agent_id,
            reactions=reactions,
        )
        set_prices: list[dict[str, Any]] = action_dic.get("set_prices", [])
        set_prices_allowed: bool = self._check_set_prices(
            agent_id=agent_id, set_prices=set_prices
        )
        if not set_prices_allowed:
            self.invalid_action_dic["set_prices"] += 1
            set_prices = []
        self._set_prices(
            agent_id=agent_id,
            set_prices=set_prices,
        )
        tweet: Optional[str] = action_dic.get("tweet", None)
        follow_agent_id: Optional[int] = action_dic.get("follow", None)
        unfollow_agent_id: Optional[int] = action_dic.get("unfollow", None)
        follow_unfollow_allowed: bool = self._check_follow_unfollow(
            agent_id=agent_id,
            follow_agent_id=follow_agent_id,
            unfollow_agent_id=unfollow_agent_id,
        )
        if not follow_unfollow_allowed:
            self.invalid_action_dic["follow"] += 1
            self.invalid_action_dic["unfollow"] += 1
            follow_agent_id = None
            unfollow_agent_id = None
        self._act_in_social_network(
            agent_id=agent_id,
            tweet=tweet,
            follow_agent_id=follow_agent_id,
            unfollow_agent_id=unfollow_agent_id,
        )

    def _check_move(self, where_to_move: Optional[tuple[int, ...] | str]) -> bool:
        """check whether the move is valid.

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
        if len(destination_pos) != len(self.space_size):
            return False
        for dim in range(len(destination_pos)):
            if not (0 <= destination_pos[dim] < self.space_size[dim]):
                return False
        return True

    def _move(
        self, agent_id: int, where_to_move: Optional[tuple[int, ...] | str] = None
    ) -> None:
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
        )
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
        next_pos: list[int] = list(current_pos)
        for dim in range(len(current_pos)):
            if current_pos[dim] < destination_pos[dim]:
                next_pos[dim] += 1
            elif current_pos[dim] > destination_pos[dim]:
                next_pos[dim] -= 1
        return tuple(next_pos)

    def _check_consumptions(
        self, agent_id: int, consumptions: list[dict[str, Any]]
    ) -> bool:
        """check whether the consumptions are valid.

        Args:
            agent_id (int): agent id of the agent who performs the consumptions.
            consumptions (list[dict[str, Any]]): list of consumption dictionaries.

        Returns:
            bool: whether the consumptions are valid.

        Note:
            Checked conditions:
            - item_name in each consumption must be in the environment.
            - item_amount in each consumption must be positive and not exceed the agent's inventory.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        for consumption in consumptions:
            item_name: str = consumption.get("item_name", "")
            item_amount: float | int = consumption.get("item_amount", 0)
            if item_name not in self.item_name2item:
                return False
            allowed_amount: float | int = agent.get_item_amount(item_name)
            if item_amount > allowed_amount:
                return False
            if item_amount <= 0:
                return False
        return True

    def _consume_items(self, agent_id: int, consumptions: list[dict[str, Any]]) -> None:
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
            if self.logger is not None:
                log.read_and_write(logger=self.logger)

    def _check_orders(self, agent_id: int, orders: list[dict[str, Any]]) -> bool:
        """check whether the orders are valid.

        Args:
            agent_id (int): agent id of the agent who places the orders.
            orders (list[dict[str, Any]]): list of order dictionaries.

        Returns:
            bool: whether the orders are valid.

        Note:
            Checked conditions:
            - counterparty_id or counterparty_name in each order must be provided.
            - counterparty_id must be an existing agent id in the environment.
            - item_name in each order must be an existing item in the environment.
            - item_amount in each order must be positive
            - The agent must have enough cash to buy all of the items.
        """
        total_cost: float | int = 0.0
        for order_dic in orders:
            counterparty_id: Optional[int] = order_dic.get("counterparty_id", None)
            counterparty_name: Optional[str] = order_dic.get("counterparty_name", None)
            if counterparty_id is None and counterparty_name is None:
                return False
            elif counterparty_id is None and counterparty_name is not None:
                counterparty_id = self.agent_name2agent_id.get(counterparty_name, None)
            if counterparty_id not in self.agent_ids:
                return False
            if "item_name" not in order_dic:
                return False
            item_name: str = order_dic["item_name"]
            if item_name not in self.item_name2item:
                return False
            item_amount: float | int = order_dic.get("item_amount", 0)
            if item_amount <= 0:
                return False
            item: Item = self.item_name2item[item_name]
            expected_price: Optional[float] = item.get_price()
            if expected_price is not None:
                expected_price *= item_amount
                total_cost += expected_price
        agent: Agent = self.agent_id2agent[agent_id]
        cash_amount: float | int = agent.get_item_amount(self.cash_name)
        if total_cost > cash_amount:
            return False
        return True

    def _check_proposals(self, agent_id: int, proposals: list[dict[str, Any]]) -> bool:
        """check whether the proposals are valid.

        Args:
            agent_id (int): agent id of the agent who makes the proposals.
            proposals (list[dict[str, Any]]): list of proposal dictionaries.

        Returns:
            bool: whether the proposals are valid.

        Note:
            Checked conditions:
            - responder_agent_id or responder_agent_name in each proposal must be provided.
            - responder_agent_id must be an existing agent id in the environment.
            - give_item_name and get_item_name in each proposal must be existing items in the environment.
            - give_item_amount and get_item_amount in each proposal must be positive.
            - The agent must have enough inventory of give_item to make the proposals.
        """
        agent: Agent = self.agent_id2agent[agent_id]
        for proposal_dic in proposals:
            responder_agent_id: Optional[int] = proposal_dic.get(
                "responder_agent_id", None
            )
            responder_agent_name: Optional[str] = proposal_dic.get(
                "responder_agent_name", None
            )
            if responder_agent_id is None and responder_agent_name is None:
                return False
            elif responder_agent_id is None and responder_agent_name is not None:
                responder_agent_id = self.agent_name2agent_id.get(
                    responder_agent_name, None
                )
            if responder_agent_id not in self.agent_ids:
                return False
            if (
                "give_item_name" not in proposal_dic
                or "give_item_amount" not in proposal_dic
                or "get_item_name" not in proposal_dic
                or "get_item_amount" not in proposal_dic
            ):
                return False
            give_item_name: str = proposal_dic["give_item_name"]
            give_item_amount: float | int = proposal_dic["give_item_amount"]
            get_item_name: str = proposal_dic["get_item_name"]
            get_item_amount: float | int = proposal_dic["get_item_amount"]
            if give_item_name not in self.item_name2item:
                return False
            if get_item_name not in self.item_name2item:
                return False
            if give_item_amount <= 0 or get_item_amount <= 0:
                return False
            allowed_give_amount: float | int = agent.get_item_amount(give_item_name)
            if give_item_amount > allowed_give_amount:
                return False
        return True

    def _add_new_orders_and_proposals(
        self,
        agent_id: int,
        orders: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
    ) -> None:
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
            if self.logger is not None:
                proposal_log.read_and_write(logger=self.logger)
            self.pending_swap_proposals.append(new_proposal)
            self.latest_proposal_id += 1

    def _check_follow_unfollow(
        self,
        agent_id: int,
        follow_agent_id: Optional[int],
        unfollow_agent_id: Optional[int],
    ) -> bool:
        if agent_id == follow_agent_id:
            return False
        if follow_agent_id is not None:
            if follow_agent_id not in self.agent_ids:
                return False
            if follow_agent_id in self.social_network.get_follows(agent_id=agent_id):
                return False
        if agent_id == unfollow_agent_id:
            return False
        if unfollow_agent_id is not None:
            if unfollow_agent_id not in self.agent_ids:
                return False
            if unfollow_agent_id not in self.social_network.get_follows(agent_id=agent_id):
                return False
        num_follows: int = self.social_network.get_num_follows(agent_id=agent_id)
        if unfollow_agent_id is not None:
            num_follows -= 1
        if (
            self.social_network.follow_cap is not None
            and num_follows >= self.social_network.follow_cap
        ):
            return False
        return True

    def _act_in_social_network(
        self,
        agent_id: int,
        tweet: Optional[str] = None,
        follow_agent_id: Optional[int] = None,
        unfollow_agent_id: Optional[int] = None,
    ) -> None:
        if tweet is not None:
            self.social_network.tweet(agent_id=agent_id, message=tweet)
            tweet_log: TweetLog = TweetLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                message=tweet,
            )
            if self.logger is not None:
                tweet_log.read_and_write(logger=self.logger)
        if follow_agent_id is not None:
            self.social_network.follow_agent(
                agent_id=agent_id, target_agent_id=follow_agent_id
            )
            follow_log: FollowLog = FollowLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                target_agent_id=follow_agent_id,
            )
            if self.logger is not None:
                follow_log.read_and_write(logger=self.logger)
        if unfollow_agent_id is not None:
            self.social_network.unfollow_agent(
                agent_id=agent_id, target_agent_id=unfollow_agent_id
            )
            unfollow_log: UnfollowLog = UnfollowLog(
                time=self.get_time(),
                time_step=self.get_time_step(),
                agent_id=agent_id,
                target_agent_id=unfollow_agent_id,
            )
            if self.logger is not None:
                unfollow_log.read_and_write(logger=self.logger)

    def _process_orders_and_proposals(self) -> None:
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

    def _check_reactions(self, agent_id: int, reactions: list[dict[str, Any]]) -> bool:
        """check whether the reactions are valid.

        Args:
            agent_id (int): agent id of the agent who makes the reactions.
            reactions (list[dict[str, Any]]): list of reaction dictionaries.

        Returns:
            bool: whether the reactions are valid.

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
        for reaction in reactions:
            if "kind" not in reaction:
                return False
            kind: str = reaction["kind"]
            if kind == "order":
                if "id" not in reaction or "accept_amount" not in reaction:
                    return False
                order_id: int = reaction["id"]
                accept_amount: float | int = reaction["accept_amount"]
                if order_id in reacted_order_ids:
                    return False
                reacted_order_ids.append(order_id)
                if accept_amount < 0:
                    return False
                find_corresponding_order: bool = False
                order: Order
                for order in self.pending_orders:
                    if order.order_id == order_id and order.counterparty_id == agent_id:
                        if accept_amount > order.item_amount:
                            return False
                        holding_amount = agent.inventory_dic.get(order.item_name, 0)
                        if accept_amount > holding_amount:
                            return False
                        find_corresponding_order = True
                        break
                if not find_corresponding_order:
                    return False
            elif kind == "proposal":
                if "id" not in reaction or "accept" not in reaction:
                    return False
                proposal_id: int = reaction["id"]
                if proposal_id in reacted_proposal_ids:
                    return False
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
                            holding_amount = agent.inventory_dic.get(
                                proposal.get_item_name, 0
                            )
                            if proposal.get_item_amount > holding_amount:
                                return False
                        find_corresponding_proposal = True
                        break
                if not find_corresponding_proposal:
                    return False
            else:
                return False
        return True

    def _process_reactions(
        self, agent_id: int, reactions: list[dict[str, Any]]
    ) -> None:
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
                            agent_id=agent_id,
                            counterparty_id=order.agent_id,
                            item_name=order.item_name,
                            item_amount=order.item_amount,
                            price=order.price,
                            order_id=order.order_id,
                            accept_amount=accept_amount,
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
                        if self.logger is not None:
                            proposal_reaction_log.read_and_write(logger=self.logger)
            else:
                raise ValueError(
                    f"Unknown reaction kind: {kind}. Must be either 'order' or 'proposal'."
                )

    def _check_set_prices(
        self, agent_id: int, set_prices: list[dict[str, Any]]
    ) -> bool:
        """check whether the set_prices are valid.

        Args:
            agent_id (int): agent id of the agent who sets the prices.
            set_prices (list[dict[str, Any]]): list of set_price dictionaries.

        Returns:
            bool: whether the set_prices are valid.

        Note:
            Checked conditions:
            - The agent cannot set prices if it is a household (i.e., in household_ids).
            - item_name in each set_price must be an existing item in the environment.
            - The agent must have non-negative inventory of the item to set its price.
            - price in each set_price must be non-negative.
        """

        if agent_id in self.household_ids:
            return False
        agent: Agent = self.agent_id2agent[agent_id]
        for set_price in set_prices:
            if "item_name" not in set_price or "price" not in set_price:
                return False
            item_name: str = set_price["item_name"]
            price: float = set_price["price"]
            if item_name not in self.item_name2item:
                return False
            if agent.get_item_amount(item_name) < 0:
                return False
            if price < 0:
                return False
        return True

    def _set_prices(self, agent_id: int, set_prices: list[dict[str, Any]]) -> None:
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
            if self.logger is not None:
                change_price_log.read_and_write(logger=self.logger)

    def _update_time(self) -> None:
        for order in self.pending_orders:
            order.update_time()
        for proposal in self.pending_swap_proposals:
            proposal.update_time()
        self._time += 1

    def _remove_expired_orders_and_proposals(self) -> None:
        self.pending_orders = [
            order for order in self.pending_orders if not order.is_fulfilled()
        ]
        self.pending_swap_proposals = [
            proposal
            for proposal in self.pending_swap_proposals
            if not proposal.is_fulfilled()
        ]

    def get_observations(self, agent_id: int) -> ObsT:
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
            str, ObsProviderForCoLocatedAgents
        ] = self._build_observation4co_located_agents_registry(
            co_located_agents=co_located_agents
        )
        obs_providers: dict[str, ObsProvider] = {}
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
            provider: ObsProvider | ObsProviderForCoLocatedAgents = obs_providers[key]
            print(provider)
            observation[key] = provider.get_obs(agent_id=agent_id)
        return observation  # type: ignore

    def _build_observation_registry(self) -> dict[str, ObsProvider]:
        return {
            "time": TimeProvider(env=self),
            "timedelta": TimeDeltaProvider(env=self),
            "self_agent_id": SelfIDProvider(env=self),
            "self_name": SelfNameProvider(env=self),
            "self_pos": SelfPosProvider(env=self),
            "self_init_pos": SelfInitPosProvider(env=self),
            "self_is_moving": SelfIsMovingProvider(env=self),
            "self_destination": SelfDestinationProvider(env=self),
            "others_pos": OthersPosProvider(env=self),
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
        return {
            "item_name2price": ItemName2PriceProvider(env=self),
        }

    def _build_observation4co_located_agents_registry(
        self, co_located_agents: set[int]
    ) -> dict[str, ObsProviderForCoLocatedAgents]:
        return {
            "others_inventory": OthersInventoriesProvider(
                env=self, co_located_agents=co_located_agents
            ),
        }
