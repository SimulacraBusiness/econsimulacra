from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Deque, Literal, Optional


@dataclass
class ConsumptionHistoryItem:
    """A class representing a consumption history item in the agent's memory.

    Attributes:
        item_name (str): the name of the consumed item.
        quantity (int | float): the quantity of the consumed item.
        time (int | str): the time of the consumption.
        time_step (int): the time step of the consumption.

    Note:
        This history item is generated based on the ConsumptionLog.
        See also: econsimulacra.logs.base.ConsumptionLog,
        econsimulacra.envs.base.Environment._consume_items(agent_id, consumptions)
    """

    item_name: str
    quantity: int | float
    time: int | str
    time_step: int


@dataclass
class MoveHistoryItem:
    """A class representing a movement history item in the agent's memory.

    Attributes:
        pos (tuple[int, ...]): the position of the agent after the movement.
        init_pos (tuple[int, ...]): the initial position of the agent assigned by the environment.
        time (int | str, optional): the time of the movement.
            It can be None for the initial position assigned by the environment, which is based on the SpaceAssignLog.

    Note:
        This history item is generated based on the MoveLog and SpaceAssignLog.
        See also: econsimulacra.logs.base.MoveLog, econsimulacra.logs.base.SpaceAssignLog,
        econsimulacra.envs.base.Environment._move(agent_id, new_pos),
        econsimulacra.envs.base.Environment._assign_agent_to_space(agent_id, coords)
    """

    pos: tuple[int, ...]
    init_pos: tuple[int, ...]
    time: Optional[int | str]
    time_step: int


@dataclass
class PurchaseHistoryItem:
    """A class representing a purchase history item in the agent's memory.

    Attributes:
        item_name (str): the name of the purchased item.
        quantity (int | float): the quantity of the purchased item.
        price (int | float): the price of the purchased item.
        time (int | str): the time of the purchase.
        from_agent_id (int): the id of the agent from whom the item is purchased.

    Note:
        This history item is generated based on the OrderReactionLog where the agent is the purchase agent
        and the reaction is accept.
        See also: econsimulacra.logs.base.OrderReactionLog,
        econsimulacra.envs.base.Environment._process_reactions(agent_id, reactions)
    """

    item_name: str
    quantity: int | float
    price: int | float
    time: int | str
    time_step: int
    from_agent_id: int


@dataclass
class InnerThoughtHistoryItem:
    """A class representing an inner thought history item in the agent's memory.

    Attributes:
        inner_thought (str): the content of the inner thought.
        time (int | str): the time of the inner thought.
        time_step (int): the time step of the inner thought.
        agent_id (int): the id of the agent generating the inner thought.

    Note:
        This history item is generated based on the InnerThoughtLog.
        See also: econsimulacra.logs.base.InnerThoughtLog,
        econsimulacra.envs.base.Environment._process_inner_thought(agent_id, inner_thought)
    """

    inner_thought: str
    time: int | str
    time_step: int


@dataclass
class SaleHistoryItem:
    """A class representing a sale history item in the agent's memory.

    Attributes:
        item_name (str): the name of the sold item.
        quantity (int | float): the quantity of the sold item.
        price (int | float): the price of the sold item.
        time (int | str): the time of the sale.
        to_agent_id (int): the id of the agent to whom the item is sold.

    Note:
        This history item is generated based on the OrderReactionLog where the agent is the sale agent
        and the reaction is accept.
        See also: econsimulacra.logs.base.OrderReactionLog,
        econsimulacra.envs.base.Environment._process_reactions(agent_id, reactions)
    """

    item_name: str
    quantity: int | float
    price: int | float
    time: int | str
    time_step: int
    to_agent_id: int


@dataclass
class ExchangeHistoryItem:
    """A class representing an exchange history item in the agent's memory.

    Attributes:
        give_item_name (str): the name of the item given in the exchange.
        give_item_quantity (int | float): the quantity of the item given in the exchange.
        get_item_name (str): the name of the item received in the exchange.
        get_item_quantity (int | float): the quantity of the item received in the exchange.
        time (int | str): the time of the exchange.
        counterparty_id (int): the id of the agent with whom the exchange is made.

    Note:
        This history item is generated based on the ProposalReactionLog where the reaction is accept.
        See also: econsimulacra.logs.base.ProposalReactionLog,
        econsimulacra.envs.base.Environment._process_reactions(agent_id, reactions)
    """

    give_item_name: str
    give_item_quantity: int | float
    get_item_name: str
    get_item_quantity: int | float
    time: int | str
    time_step: int
    counterparty_id: int


@dataclass
class SleepHistoryItem:
    """A class representing a sleep history item in the agent's memory.

    Attributes:
        start_time (int | str): the time of the sleep action.
        end_time (int | str): the time when the sleep action ends.

    Note:
        This history item is generated based on the SleepStartLog and SleepEndLog.
        See also: econsimulacra.logs.base.SleepStartLog, econsimulacra.logs.base.SleepEndLog,
        econsimulacra.envs.base.Environment.(agent_id)
    """

    start_time: int | str
    end_time: Optional[int | str]


@dataclass
class SetPriceHistoryItem:
    """A class representing a price change history item in the agent's memory.

    Attributes:
        item_name (str): the name of the item whose price is changed.
        old_price (int | float): the old price of the item.
        new_price (int | float): the new price of the item.
        time (int | str): the time of the price change.

    Note:
        This history item is generated based on the ChangePriceLog.
        See also: econsimulacra.logs.base.ChangePriceLog,
        econsimulacra.envs.base.Environment._set_price(agent_id, set_prices)
    """

    item_name: str
    old_price: int | float
    new_price: int | float
    time: int | str
    time_step: int


@dataclass
class SocialHistoryItem:
    """A class representing a social action history item in the agent's memory.

    Attributes:
        action (Literal["follow", "unfollow"]): the type of the social action.
        target_agent_id (int): the id of the target agent whom the agent follows or unfollows.
        time (int | str): the time of the social action.
        num_followers (int): the number of followers of the agent after the social action.
        num_follows (int): the number of agents that the agent follows after the social action.

    Note:
        This history item is generated based on the FollowLog and UnfollowLog.
        See also: econsimulacra.logs.base.FollowLog, econsimulacra.logs.base.UnfollowLog,
        econsimulacra.envs.base.Environment._act_in_social_network(agent_id, tweet, follow_agent_id, unfollow_agent_id)
    """

    action: Literal["follow", "unfollow"]
    target_agent_id: int
    time: int | str
    time_step: int
    num_followers: int
    num_follows: int


@dataclass
class StateEvaluationHistoryItem:
    """A class representing a state evaluation item in the agent's memory.

    Attributes:
        wealth (float): the wealth of the agent at the time of evaluation.
        relative_wealth (float, optional): The relative wealth of the agent at the time of evaluation.
            Only household agents have this value; for other agent types, it is None.
        buying_power (float, optional): The buying power of the agent at the time of evaluation.
            Only household agents have this value; for other agent types, it is None.
        inventory_dic (dict[str, int | float]): the inventory of the agent at the time of evaluation.
        persona_dic (dict[str, Any], optional): the persona of the agent at the time of evaluation.
        time (int | str): the time of the state evaluation.

    Note:
        This history item is generated based on the StateEvaluationLog.
        See also: econsimulacra.logs.base.StateEvaluationLog,
        econsimulacra.envs.base.Environment.evaluate_agent_state(agent_id)
    """

    wealth: float
    relative_wealth: Optional[float]
    buying_power: Optional[float]
    inventory_dic: dict[str, int | float]
    persona_dic: Optional[dict[str, Any]]
    time: int | str
    time_step: int


@dataclass
class ObsHistoryItem:
    """A class representing an observation item in the agent's memory.

    Attributes:
        obs_type (str): The type of the observation.
        time (int | str): The time of the observation.
        time_step (int): The time step of the observation.
        agent_id (int): The unique id of the agent making the observation.
        obs (Any): The observation details.
    """

    obs_type: str
    time: int | str
    time_step: int
    obs: Any


@dataclass
class AgentMemory:
    """Agent Memory class.

    Store the history of the agent's actions and observations in a summarized form.
    The memory is updated based on the logs generated by the environment.

    Attributes:
        consumption_history (Deque[ConsumptionHistoryItem]): the history of the agent's consumption.
        move_history (Deque[MoveHistoryItem]): the history of the agent's movement.
        purchase_history (Deque[PurchaseHistoryItem]): the history of the agent's purchase.
        sale_history (Deque[SaleHistoryItem]): the history of the agent's sale.
        exchange_history (Deque[ExchangeHistoryItem]): the history of the agent's exchange.
        set_price_history (Deque[SetPriceHistoryItem]): the history of the agent's price change.
        inner_thought_history (Deque[InnerThoughtHistoryItem]): the history of the agent's inner thoughts.
        social_history (Deque[SocialHistoryItem]): the history of the agent's social actions.
        state_evaluation_history (Deque[StateEvaluationHistoryItem]): the history of the agent's state evaluations.
        obs_history (Deque[ObsHistoryItem]): the history of the agent's observations.

    Note:
        The history is stored in a deque with a maximum length of memory_length, which is defined in the MemoryHandler.
        When the history exceeds the maximum length, the oldest history will be removed.
    """

    consumption_history: Deque[ConsumptionHistoryItem]
    sleep_history: Deque[SleepHistoryItem]
    move_history: Deque[MoveHistoryItem]
    purchase_history: Deque[PurchaseHistoryItem]
    sale_history: Deque[SaleHistoryItem]
    exchange_history: Deque[ExchangeHistoryItem]
    set_price_history: Deque[SetPriceHistoryItem]
    inner_thought_history: Deque[InnerThoughtHistoryItem]
    social_history: Deque[SocialHistoryItem]
    state_evaluation_history: Deque[StateEvaluationHistoryItem]
    obs_history: Deque[ObsHistoryItem]
