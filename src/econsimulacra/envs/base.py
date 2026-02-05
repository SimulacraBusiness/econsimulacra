from abc import ABC, abstractmethod
from ..agents import Agent
from .env_utils import find_class
from ..items import Item
from ..logs import AgentGenerationLog
from ..logs import SpaceAssignLog
from ..logs import MoveLog
from ..logs import ConsumptionLog
from ..logs import OrderLog
from ..logs import ProposalLog
from ..logs import OrderReactionLog
from ..logs import ProposalReactionLog
from ..logs import TweetLog
from ..logs import FollowLog
from ..logs import UnfollowLog
from ..logs import Logger
import random
from random import Random
from .order import Order
from .order import SwapProposal
from .social_network import SocialNetwork
from .space import GridSpace
from typing import Any
from typing import Literal
from typing import Optional
from typing import Type
from typing import Generic, TypeVar

ObsT = TypeVar("ObsT")


class Environment(ABC, Generic[ObsT]):
    def __init__(
        self,
        config: dict[str, Any],
        logger: Optional[Logger] = None,
    ) -> None:
        """Initialization.

        config example:
        {
            "simulation": {
                "numSteps": int,
            },
            "environment": {
                "gridSpace": [int, ...],
                "cashName": str,
                "agents": ["Household", "Retailer", "Restaurant", ...],
                "items": ["Yen", "Rice", ...]
            }
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
        env_config: dict[str, Any] = config.get("environment", {})
        if "gridSpace" not in env_config:
            raise ValueError("Environment config must include 'gridSpace' key.")
        self.space_size: tuple[int, ...] = env_config["gridSpace"]
        if "cashName" not in env_config:
            raise ValueError("Simulation config must include 'cashName' key.")
        self.cash_name: str = env_config["cashName"]
        self.config: dict[str, Any] = config
        self.prng: Random = random.Random()
        self.registered_classes: list[Type] = []
        self.logger: Optional[Logger] = logger
        self._time: int = -1

    def get_time(self) -> int:
        return self._time

    def register_classes(self, class_list: list[Type]) -> None:
        self.registered_classes.extend(class_list)

    def reset(self, seed: int) -> None:
        if self.logger is not None:
            self.logger.clear()
        self.prng.seed(seed)
        self.grid_space: GridSpace = GridSpace(space_size=self.space_size)
        self.social_network: SocialNetwork = SocialNetwork()
        assert "environment" in self.config, "Config must include 'environment' key."
        assert "agents" in self.config["environment"], (
            "Environment config must include 'agents' key."
        )
        agent_types: list[str] = self.config["environment"]["agents"]
        self._generate_agents(agent_types=agent_types)
        assert "items" in self.config["environment"], (
            "Environment config must include 'items' key."
        )
        item_names: list[str] = self.config["environment"]["items"]
        self._generate_items(item_names=item_names)
        self.pending_orders: list[Order] = []
        self.pending_swap_proposals: list[SwapProposal] = []
        self.latest_order_id: int = 0
        self.latest_proposal_id: int = 0
        if self.logger is not None:
            self.logger.process_logs()
        self._time = 0

    def _generate_agents(self, agent_types: list[str]) -> None:
        """generate agents and place them in the grid space.

        Args:
            agent_types (list[str]): name list of agent types to be generated.
        """
        current_agent_id: int = 0
        self.agent_ids: list[int] = []
        self.household_ids: list[int] = []
        self.others_ids: list[int] = []
        self.agent_id2agent: dict[int, Agent] = {}
        self.agent_name2agent_id: dict[str, int] = {}
        self.agent_id2initial_coords: dict[int, tuple[int, ...]] = {}
        self.agent_id2is_moving: dict[int, bool] = {}
        self.agent_id2destination: dict[int, Optional[tuple[int, ...]]] = {}
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
                log: AgentGenerationLog = AgentGenerationLog(
                    agent_id=current_agent_id,
                    time=self.get_time(),
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

    def _generate_items(self, item_names: list[str]) -> None:
        """generate items.

        Args:
            item_names (list[str]): name list of items to be generated.
        """
        self.item_name2item: dict[str, Any] = {}
        for item_name in item_names:
            item_class: Type[Item] = find_class(
                name=item_name, optional_class_list=self.registered_classes
            )
            item_instance: Item = item_class(
                item_id=len(self.item_name2item), item_name=item_name
            )
            self.item_name2item[item_name] = item_instance

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
            "tweet": str, # Optional, default None
            "follow": int, # agent_id to follow. Optional, default None
            "unfollow": int, # agent_id to unfollow. Optional, default None
        }
        """
        where_to_move: Optional[tuple[int, ...] | str] = action_dic.get("move", None)
        self._move(
            agent_id=agent_id,
            where_to_move=where_to_move,
        )
        consumptions: list[dict[str, Any]] = action_dic.get("consumptions", [])
        self._consume_items(
            agent_id=agent_id,
            consumptions=consumptions,
        )
        orders: list[dict[str, Any]] = action_dic.get("orders", [])
        proposals: list[dict[str, Any]] = action_dic.get("proposals", [])
        self._add_new_orders_and_proposals(
            agent_id=agent_id,
            orders=orders,
            proposals=proposals,
        )
        reactions: list[dict[str, Any]] = action_dic.get("reactions", [])
        self._process_reactions(
            agent_id=agent_id,
            reactions=reactions,
        )
        tweet: Optional[str] = action_dic.get("tweet", None)
        follow_agent_id: Optional[int] = action_dic.get("follow", None)
        unfollow_agent_id: Optional[int] = action_dic.get("unfollow", None)
        self._act_in_social_network(
            agent_id=agent_id,
            tweet=tweet,
            follow_agent_id=follow_agent_id,
            unfollow_agent_id=unfollow_agent_id,
        )

    def _move(
        self, agent_id: int, where_to_move: Optional[tuple[int, ...] | str] = None
    ) -> None:
        current_pos: tuple[int, ...] = self.grid_space.get_pos(agent_id=agent_id)
        if where_to_move is None:
            return
        elif isinstance(where_to_move, str):
            destination_name: str = where_to_move
            destination_id: int = self.agent_name2agent_id[destination_name]
            destination_pos: tuple[int, ...] = self.grid_space.get_pos(
                agent_id=destination_id
            )
        elif isinstance(where_to_move, tuple):
            destination_pos = where_to_move
        else:
            raise ValueError(
                f"where_to_move must be either tuple[int, ...] or str, but got {type(where_to_move)}."
            )

        def calc_next_pos(
            current_pos: tuple[int, ...], destination_pos: tuple[int, ...]
        ) -> tuple[int, ...]:
            next_pos: list[int] = list(current_pos)
            for dim in range(len(current_pos)):
                if current_pos[dim] < destination_pos[dim]:
                    next_pos[dim] += 1
                elif current_pos[dim] > destination_pos[dim]:
                    next_pos[dim] -= 1
            return tuple(next_pos)

        next_pos: tuple[int, ...] = calc_next_pos(
            current_pos=current_pos, destination_pos=destination_pos
        )
        self.grid_space.move_agent(agent_id=agent_id, new_pos=next_pos)
        log: MoveLog = MoveLog(
            time=self.get_time(),
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

    def _consume_items(self, agent_id: int, consumptions: list[dict[str, Any]]) -> None:
        agent: Agent = self.agent_id2agent[agent_id]
        for consumption in consumptions:
            item_name: str = consumption["item_name"]
            item_amount: float | int = consumption["item_amount"]
            agent.exchange_goods(
                get_item_name=None,
                get_item_amount=None,
                give_item_name=item_name,
                give_item_amount=item_amount,
            )
            log: ConsumptionLog = ConsumptionLog(
                time=self.get_time(),
                agent_id=agent_id,
                item_name=item_name,
                item_amount=item_amount,
            )
            if self.logger is not None:
                log.read_and_write(logger=self.logger)

    def _add_new_orders_and_proposals(
        self,
        agent_id,
        orders: list[dict[str, Any]],
        proposals: list[dict[str, Any]],
    ) -> None:
        for order_dic in orders:
            counterparty_id: Optional[int] = order_dic.get("counterparty_id", None)
            counterparty_name: Optional[str] = order_dic.get("counterparty_name", None)
            if counterparty_id is None and counterparty_name is not None:
                counterparty_id = self.agent_name2agent_id[counterparty_name]
            elif counterparty_id is None and counterparty_name is None:
                raise ValueError(
                    "Either counterparty_id or counterparty_name must be provided in order_dic."
                )
            if "item_name" not in order_dic:
                raise ValueError("item_name must be provided in order_dic.")
            assert counterparty_id is not None
            item_name: str = order_dic["item_name"]
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

    def _act_in_social_network(
        self,
        agent_id: int,
        tweet: Optional[str] = None,
        follow_agent_id: Optional[int] = None,
        unfollow_agent_id: Optional[int] = None,
    ) -> None:
        if tweet is not None:
            self.social_network.tweet(agent_id=agent_id, message=tweet)
            tweet_log: TweetLog = TweetLog(time=self.get_time(), agent_id=agent_id, message=tweet)
            if self.logger is not None:
                tweet_log.read_and_write(logger=self.logger)
        if follow_agent_id is not None:
            self.social_network.follow_agent(
                agent_id=agent_id, target_agent_id=follow_agent_id
            )
            follow_log: FollowLog = FollowLog(
                time=self.get_time(), agent_id=agent_id, target_agent_id=follow_agent_id
            )
            if self.logger is not None:
                follow_log.read_and_write(logger=self.logger)
        if unfollow_agent_id is not None:
            self.social_network.unfollow_agent(
                agent_id=agent_id, target_agent_id=unfollow_agent_id
            )
            unfollow_log: UnfollowLog = UnfollowLog(
                time=self.get_time(),
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
                                f"Agent {agent_id} cannot react to order {order_id} as it is not the counterparty."
                            )
                        order.react(amount=accept_amount)
                        order_reaction_log: OrderReactionLog = OrderReactionLog(
                            time=self.get_time(),
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

    @abstractmethod
    def get_observations(self, agent_id: int) -> ObsT:
        pass
