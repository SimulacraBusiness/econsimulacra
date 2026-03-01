from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from ..logs import Log
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
import random
from typing import Any
from typing import Callable
from typing import Deque
from typing import Literal
from typing import Optional


@dataclass
class ConsumptionHistoryItem:
    item_name: str
    quantity: int | float
    time: int | str


@dataclass
class MoveHistoryItem:
    pos: tuple[int, ...]
    time: Optional[int | str]


@dataclass
class PurchaseHistoryItem:
    item_name: str
    quantity: int | float
    price: int | float
    time: int | str
    from_agent_id: int


@dataclass
class SaleHistoryItem:
    item_name: str
    quantity: int | float
    price: int | float
    time: int | str
    to_agent_id: int


@dataclass
class ExchangeHistoryItem:
    give_item_name: str
    give_item_quantity: int | float
    get_item_name: str
    get_item_quantity: int | float
    time: int | str
    counterparty_id: int


@dataclass
class SetPriceHistoryItem:
    item_name: str
    old_price: int | float
    new_price: int | float
    time: int | str


@dataclass
class SocialHistoryItem:
    action: Literal["follow", "unfollow"]
    target_agent_id: int
    time: int | str
    num_followers: int
    num_follows: int


@dataclass
class AgentMemory:
    consumption_history: Deque[ConsumptionHistoryItem]
    move_history: Deque[MoveHistoryItem]
    purchase_history: Deque[PurchaseHistoryItem]
    sale_history: Deque[SaleHistoryItem]
    exchange_history: Deque[ExchangeHistoryItem]
    set_price_history: Deque[SetPriceHistoryItem]
    social_history: Deque[SocialHistoryItem]


class MemoryHandler:
    def __init__(
        self, config: dict[str, Any], prng: Optional[random.Random] = None
    ) -> None:
        self.config: dict[str, Any] = config
        if "memoryLength" in self.config:
            self.memory_length: int = self.config["memoryLength"]
        else:
            raise ValueError(
                "memoryLength must be specified in the config for MemoryHandler."
            )
        assert 1 <= self.memory_length, "memoryLength must be at least 1."
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.agent_id2memory: dict[int, AgentMemory] = {}
        self.memory_updaters: dict[type[Log], Callable[[Any], None]] = (
            self._build_memory_registry()
        )

    def get_memory(self, agent_id: int) -> dict[str, Any]:
        """summarize and return the memory of the agent with the given agent_id.

        Args:
            agent_id (int): the id of the agent whose memory is to be retrieved.

        Note:
            The structure of the summarized memory is defined as the following dictionary:
            {
                "memory_length": int, # the maximum number of logs to be stored in memory for the agent.
                "move_history": "(x0,y0) -> (x1,y1) -> (x2,y2)", # the history of the agent's movement.
                "consumption_history": "item_name1 x quantity1 at time1, item_name2 x quantity2 at time2, ...", # the history of the agent's consumption.
                "purchase_history": "item_name1 x quantity1 at price1 from agent_id1 at time1, item_name2 x quantity2 at price2 from agent_id2 at time2, ...", # the history of the agent's purchase.
                "sale_history": "item_name1 x quantity1 at price1 to agent_id1 at time1, item_name2 x quantity2 at price2 to agent_id2 at time2, ...", # the history of the agent's sale.
                "exchange_history": "give item_name1 x quantity1, get item_name2 x quantity2 with agent_id1 at time1; give item_name3 x quantity3, get item_name4 x quantity4 with agent_id2 at time2; ...", # the history of the agent's exchange.
                "set_price_history": "item_name1: old_price1 -> new_price1 at time1, item_name2: old_price2 -> new_price2 at time2, ...", # the history of the agent's price change.
                "social_history": "follow target_agent_id1 at time1 (num_followers: num_followers1, num_follows: num_follows1); unfollow target_agent_id2 at time2 (num_followers: num_followers2, num_follows: num_follows2); ...", # the history of the agent's social actions.
            }
        """
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        summarized_memory: dict[str, Any] = {
            "memory_length": f"Max memory length is {self.memory_length}.",
            "move_history": "",
            "consumption_history": "",
            "purchase_history": "",
            "sale_history": "",
            "exchange_history": "",
            "set_price_history": "",
            "social_history": "",
        }
        if len(agent_memory.move_history) > 0:
            summarized_memory["move_history"] = "You have moved to " + " -> ".join(
                f"{item.pos}" for item in agent_memory.move_history
            )
        if len(agent_memory.consumption_history) > 0:
            summarized_memory["consumption_history"] = "You have consumed " + ", ".join(
                f"{item.item_name} x {item.quantity} at {item.time}"
                for item in agent_memory.consumption_history
            )
        if len(agent_memory.purchase_history) > 0:
            summarized_memory["purchase_history"] = "You have purchased " + ", ".join(
                f"{item.item_name} x {item.quantity} at {item.price} from agent_id {item.from_agent_id} at {item.time}"
                for item in agent_memory.purchase_history
            )
        if len(agent_memory.sale_history) > 0:
            summarized_memory["sale_history"] = "You have sold " + ", ".join(
                f"{item.item_name} x {item.quantity} at {item.price} to agent_id {item.to_agent_id} at {item.time}"
                for item in agent_memory.sale_history
            )
        if len(agent_memory.exchange_history) > 0:
            summarized_memory["exchange_history"] = "You have exchanged " + "; ".join(
                f"give {item.give_item_name} x {item.give_item_quantity}, get {item.get_item_name} x {item.get_item_quantity} with agent_id {item.counterparty_id} at {item.time}"
                for item in agent_memory.exchange_history
            )
        if len(agent_memory.set_price_history) > 0:
            summarized_memory["set_price_history"] = (
                "You have changed price "
                + ", ".join(
                    f"{item.item_name}: {item.old_price} -> {item.new_price} at {item.time}"
                    for item in agent_memory.set_price_history
                )
            )
        if len(agent_memory.social_history) > 0:
            summarized_memory["social_history"] = (
                "Your social actions are "
                + "; ".join(
                    f"{item.action} target_agent_id {item.target_agent_id} at {item.time} (num_followers: {item.num_followers}, num_follows: {item.num_follows})"
                    for item in agent_memory.social_history
                )
            )
        return summarized_memory

    def _build_memory_registry(self) -> dict[type[Log], Callable[[Any], None]]:
        return {
            AgentGenerationLog: self._process_agent_generation_log,
            SpaceAssignLog: self._process_space_assign_log,
            MoveLog: self._process_move_log,
            ConsumptionLog: self._process_consumption_log,
            OrderLog: self._process_order_log,
            ProposalLog: self._process_proposal_log,
            OrderReactionLog: self._process_order_reaction_log,
            ProposalReactionLog: self._process_proposal_reaction_log,
            ChangePriceLog: self._process_change_price_log,
            TweetLog: self._process_tweet_log,
            FollowLog: self._process_follow_log,
            UnfollowLog: self._process_unfollow_log,
        }

    def update(self, log: Log) -> None:
        """update memory based on the log. This method is called in Environment.remember_log."""
        handler: Optional[Callable[[Any], None]] = self.memory_updaters.get(type(log))
        if handler is not None:
            handler(log)

    def _process_agent_generation_log(self, log: AgentGenerationLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            self.agent_id2memory[agent_id] = AgentMemory(
                consumption_history=deque(maxlen=self.memory_length),
                move_history=deque(maxlen=self.memory_length),
                purchase_history=deque(maxlen=self.memory_length),
                sale_history=deque(maxlen=self.memory_length),
                exchange_history=deque(maxlen=self.memory_length),
                set_price_history=deque(maxlen=self.memory_length),
                social_history=deque(maxlen=self.memory_length),
            )
        else:
            raise ValueError(f"Agent with id {agent_id} already exists in memory.")

    def _process_space_assign_log(self, log: SpaceAssignLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        move_history: Deque[MoveHistoryItem] = agent_memory.move_history
        if len(move_history) > 0:
            raise ValueError(
                f"Agent with id {agent_id} already has a position assigned in memory."
            )
        move_history.append(MoveHistoryItem(pos=log.pos, time=None))

    def _process_move_log(self, log: MoveLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        move_history: Deque[MoveHistoryItem] = agent_memory.move_history
        old_pos: Optional[tuple[int, ...]] = (
            move_history[-1].pos if move_history else None
        )
        old_pos_in_log: tuple[int, ...] = log.old_pos
        if old_pos != old_pos_in_log:
            raise ValueError(
                f"Agent with id {agent_id} has a different position in memory ({old_pos}) and in log ({old_pos_in_log})."
            )
        move_history.append(MoveHistoryItem(pos=log.new_pos, time=log.time))

    def _process_consumption_log(self, log: ConsumptionLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        consumption_history: Deque[ConsumptionHistoryItem] = (
            agent_memory.consumption_history
        )
        last_consumption: Optional[ConsumptionHistoryItem] = (
            consumption_history[-1] if consumption_history else None
        )
        if last_consumption is not None:
            last_consumption_time: int | str = last_consumption.time
            last_consumption_item_name: str = last_consumption.item_name
            if (
                last_consumption_time == log.time
                and last_consumption_item_name == log.item_name
            ):
                consumption_history[-1].quantity += log.item_amount
                return
        consumption_history.append(
            ConsumptionHistoryItem(
                item_name=log.item_name,
                quantity=log.item_amount,
                time=log.time,
            )
        )

    def _process_order_log(self, log: OrderLog) -> None:
        pass

    def _process_proposal_log(self, log: ProposalLog) -> None:
        pass

    def _process_order_reaction_log(self, log: OrderReactionLog) -> None:
        purchase_agent_id: int = log.agent_id
        sale_agent_id: int = log.counterparty_id
        if purchase_agent_id not in self.agent_id2memory:
            raise ValueError(
                f"Agent with id {purchase_agent_id} does not exist in memory."
            )
        if sale_agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {sale_agent_id} does not exist in memory.")
        self._process_order_reaction_log_purchase_agent(log)
        self._process_order_reaction_log_sale_agent(log)

    def _process_order_reaction_log_purchase_agent(self, log: OrderReactionLog) -> None:
        purchase_agent_id: int = log.agent_id
        sale_agent_id: int = log.counterparty_id
        purchase_agent_memory: AgentMemory = self.agent_id2memory[purchase_agent_id]
        purchase_history: Deque[PurchaseHistoryItem] = (
            purchase_agent_memory.purchase_history
        )
        last_purchase: Optional[PurchaseHistoryItem] = (
            purchase_history[-1] if purchase_history else None
        )
        if last_purchase is not None:
            last_purchase_time: int | str = last_purchase.time
            last_purchase_item_name: str = last_purchase.item_name
            last_purchase_from: int = last_purchase.from_agent_id
            if (
                last_purchase_time == log.time
                and last_purchase_item_name == log.item_name
                and last_purchase_from == sale_agent_id
            ):
                purchase_history[-1].quantity += log.accept_amount
                return
        purchase_history.append(
            PurchaseHistoryItem(
                item_name=log.item_name,
                quantity=log.accept_amount,
                price=log.price,
                time=log.time,
                from_agent_id=sale_agent_id,
            )
        )

    def _process_order_reaction_log_sale_agent(self, log: OrderReactionLog) -> None:
        purchase_agent_id: int = log.agent_id
        sale_agent_id: int = log.counterparty_id
        sale_agent_memory: AgentMemory = self.agent_id2memory[sale_agent_id]
        sale_history: Deque[SaleHistoryItem] = sale_agent_memory.sale_history
        last_sale: Optional[SaleHistoryItem] = (
            sale_history[-1] if sale_history else None
        )
        if last_sale is not None:
            last_sale_time: int | str = last_sale.time
            last_sale_item_name: str = last_sale.item_name
            last_sale_to: int = last_sale.to_agent_id
            if (
                last_sale_time == log.time
                and last_sale_item_name == log.item_name
                and last_sale_to == purchase_agent_id
            ):
                sale_history[-1].quantity += log.accept_amount
                return
        sale_history.append(
            SaleHistoryItem(
                item_name=log.item_name,
                quantity=log.accept_amount,
                price=log.price,
                time=log.time,
                to_agent_id=purchase_agent_id,
            )
        )

    def _process_proposal_reaction_log(self, log: ProposalReactionLog) -> None:
        proposer_agent_id: int = log.proposer_agent_id
        responder_agent_id: int = log.responder_agent_id
        if proposer_agent_id not in self.agent_id2memory:
            raise ValueError(
                f"Agent with id {proposer_agent_id} does not exist in memory."
            )
        if responder_agent_id not in self.agent_id2memory:
            raise ValueError(
                f"Agent with id {responder_agent_id} does not exist in memory."
            )
        if log.accept:
            self._process_proposal_reaction_log_proposer_agent(log)
            self._process_proposal_reaction_log_responder_agent(log)

    def _process_proposal_reaction_log_proposer_agent(
        self, log: ProposalReactionLog
    ) -> None:
        proposer_agent_id: int = log.proposer_agent_id
        responder_agent_id: int = log.responder_agent_id
        proposer_agent_memory: AgentMemory = self.agent_id2memory[proposer_agent_id]
        exchange_history: Deque[ExchangeHistoryItem] = (
            proposer_agent_memory.exchange_history
        )
        exchange_history.append(
            ExchangeHistoryItem(
                give_item_name=log.give_item_name,
                give_item_quantity=log.give_item_amount,
                get_item_name=log.get_item_name,
                get_item_quantity=log.get_item_amount,
                time=log.time,
                counterparty_id=responder_agent_id,
            )
        )

    def _process_proposal_reaction_log_responder_agent(
        self, log: ProposalReactionLog
    ) -> None:
        proposer_agent_id: int = log.proposer_agent_id
        responder_agent_id: int = log.responder_agent_id
        responder_agent_memory: AgentMemory = self.agent_id2memory[responder_agent_id]
        exchange_history: Deque[ExchangeHistoryItem] = (
            responder_agent_memory.exchange_history
        )
        exchange_history.append(
            ExchangeHistoryItem(
                give_item_name=log.get_item_name,
                give_item_quantity=log.get_item_amount,
                get_item_name=log.give_item_name,
                get_item_quantity=log.give_item_amount,
                time=log.time,
                counterparty_id=proposer_agent_id,
            )
        )

    def _process_change_price_log(self, log: ChangePriceLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        set_price_history: Deque[SetPriceHistoryItem] = agent_memory.set_price_history
        set_price_history.append(
            SetPriceHistoryItem(
                item_name=log.item_name,
                old_price=log.old_price,
                new_price=log.new_price,
                time=log.time,
            )
        )

    def _process_tweet_log(self, log: TweetLog) -> None:
        pass

    def _process_follow_log(self, log: FollowLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        social_history: Deque[SocialHistoryItem] = agent_memory.social_history
        social_history.append(
            SocialHistoryItem(
                action="follow",
                target_agent_id=log.target_agent_id,
                time=log.time,
                num_followers=log.num_followers,
                num_follows=log.num_follows,
            )
        )

    def _process_unfollow_log(self, log: UnfollowLog) -> None:
        agent_id: int = log.agent_id
        if agent_id not in self.agent_id2memory:
            raise ValueError(f"Agent with id {agent_id} does not exist in memory.")
        agent_memory: AgentMemory = self.agent_id2memory[agent_id]
        social_history: Deque[SocialHistoryItem] = agent_memory.social_history
        social_history.append(
            SocialHistoryItem(
                action="unfollow",
                target_agent_id=log.target_agent_id,
                time=log.time,
                num_followers=log.num_followers,
                num_follows=log.num_follows,
            )
        )
