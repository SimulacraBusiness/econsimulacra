from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class BaseRecord:
    """Base record for log analyses.

    A record instance represents a single log entry.

    In typical usage, a record is created by parsing a log line,
    and the type field is set to the log entry's type.
    A list of BaseRecord instances is then passed to
    a RecordStore (log_analyses.store) for efficient querying.

    Attributes:
        type (str): The type of the log entry, e.g. "agent_action", "space_assign", etc.
            Corresponds to the "type" field in the econsimulacra.logs.Log.
    """

    type: str


@dataclass(frozen=True)
class TimedRecord(BaseRecord):
    """A record with a timestamp and time step.

    Attributes:
        time (int | datetime): The timestamp of the log entry, as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
    """

    time: int | datetime
    time_step: int


@dataclass(frozen=True)
class AgentGenerationRecord(BaseRecord):
    """A record for agent generation events.

    Corresponds to econsimulacra.logs.AgentGenerationLog.

    Attributes:
        type (str): must be "agent_generation".
        time (int | datetime): The timestamp of the log entry, as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the generated agent.
        agent_type (str): The type of the generated agent.
        agent_name (str): The name of the generated agent.
        wealth (float): The initial wealth of the generated agent.
        inventory (dict[str, float]): The initial inventory of the generated agent,
            as a dict mapping item names to amounts.
        persona (Optional[dict[str, Any]]): The persona of the generated agent,
            as a dict. May be None if the agent has no persona.
    """

    time: int | datetime
    time_step: int
    agent_id: int
    agent_type: str
    agent_name: str
    wealth: float
    inventory: dict[str, float]
    persona: Optional[dict[str, Any]]


@dataclass(frozen=True)
class SpaceAssignRecord(BaseRecord):
    """A record for space assignment events.

    Corresponds to econsimulacra.logs.SpaceAssignLog.

    Attributes:
        type (str): must be "space_assign".
        agent_id (int): The ID of the agent being assigned a space.
        pos (tuple[int, ...]): The position to which the agent is assigned.
    """

    agent_id: int
    pos: tuple[int, ...]


@dataclass(frozen=True)
class MoveRecord(TimedRecord):
    """A record for agent movement events.

    Corresponds to econsimulacra.logs.MoveLog.

    Attributes:
        type (str): must be "move".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that moved.
        old_pos (tuple[int, ...]): The previous position of the agent.
        new_pos (tuple[int, ...]): The new position of the agent.
    """

    agent_id: int
    old_pos: tuple[int, ...]
    new_pos: tuple[int, ...]


@dataclass(frozen=True)
class ConsumptionRecord(TimedRecord):
    """A record for agent consumption events.

    Corresponds to econsimulacra.logs.ConsumptionLog.

    Attributes:
        type (str): must be "consumption".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that consumed an item.
        item_name (str): The name of the consumed item.
        item_amount (float): The amount of the consumed item.
    """

    agent_id: int
    item_name: str
    item_amount: float | int


@dataclass(frozen=True)
class OrderRecord(TimedRecord):
    """A record for order events.

    Corresponds to econsimulacra.logs.OrderLog.

    Attributes:
        type (str): must be "order".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that placed the order.
        counterparty_id (int): The ID of the counterparty agent.
        item_name (str): The name of the item being bought or sold.
        item_amount (float | int): The amount of the item being bought or sold.
        price (float, optional): The price of the item, if applicable.
        order_id (int): The ID of the order.
    """

    agent_id: int
    counterparty_id: int
    item_name: str
    item_amount: float | int
    price: Optional[float]
    order_id: int


@dataclass(frozen=True)
class ProposalRecord(TimedRecord):
    """A record for proposal events.

    Corresponds to econsimulacra.logs.ProposalLog.

    Attributes:
        type (str): must be "proposal".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        proposal_id (int): The ID of the proposal.
        proposer_agent_id (int): The ID of the proposing agent.
        responder_agent_id (int): The ID of the responding agent.
        give_item_name (str): The name of the item being offered by the proposer.
        give_item_amount (float | int): The amount of the item being offered by the proposer.
        get_item_name (str): The name of the item being requested by the proposer.
        get_item_amount (float | int): The amount of the item being requested by the proposer.
    """

    proposal_id: int
    proposer_agent_id: int
    responder_agent_id: int
    give_item_name: str
    give_item_amount: float | int
    get_item_name: str
    get_item_amount: float | int


@dataclass(frozen=True)
class OrderReactionRecord(TimedRecord):
    """A record for order reaction events.

    Corresponds to econsimulacra.logs.OrderReactionLog.

    Attributes:
        type (str): must be "order_reaction".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that reacted to the order.
        counterparty_id (int): The ID of the counterparty agent.
        item_name (str): The name of the item being bought or sold.
        item_amount (float | int): The amount of the item being bought or sold.
        price (float): The price of the item.
        order_id (int): The ID of the order.
        accept_amount (float | int): The amount of the item
            that was accepted in the order reaction.
    """

    agent_id: int
    counterparty_id: int
    item_name: str
    item_amount: float | int
    price: float
    order_id: int
    accept_amount: float | int


@dataclass(frozen=True)
class OrderExpirationRecord(TimedRecord):
    """A record for order expiration events.

    Corresponds to econsimulacra.logs.OrderExpirationLog.

    Attributes:
        type (str): must be "order_expiration".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        order_id (int): The ID of the expired order.
    """

    order_id: int


@dataclass(frozen=True)
class ProposalReactionRecord(TimedRecord):
    """A record for proposal reaction events.

    Corresponds to econsimulacra.logs.ProposalReactionLog.

    Attributes:
        type (str): must be "proposal_reaction".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        proposal_id (int): The ID of the proposal.
        proposer_agent_id (int): The ID of the proposing agent.
        responder_agent_id (int): The ID of the responding agent.
        give_item_name (str): The name of the item being offered by the proposer.
        give_item_amount (float | int): The amount of the item being offered by the proposer.
        get_item_name (str): The name of the item being requested by the proposer.
        get_item_amount (float | int): The amount of the item being requested by the proposer.
        accept (bool): Whether the proposal was accepted or rejected.
    """

    proposal_id: int
    proposer_agent_id: int
    responder_agent_id: int
    give_item_name: str
    give_item_amount: float | int
    get_item_name: str
    get_item_amount: float | int
    accept: bool


@dataclass(frozen=True)
class ProposalExpirationRecord(TimedRecord):
    """A record for proposal expiration events.

    Corresponds to econsimulacra.logs.ProposalExpirationLog.

    Attributes:
        type (str): must be "proposal_expiration".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        proposal_id (int): The ID of the expired proposal.
    """

    proposal_id: int


@dataclass(frozen=True)
class ChangePriceRecord(TimedRecord):
    """A record for price change events.

    Corresponds to econsimulacra.logs.ChangePriceLog.

    Attributes:
        type (str): must be "change_price".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that changed the price.
        item_name (str): The name of the item whose price was changed.
        old_price (float): The old price of the item.
        new_price (float): The new price of the item.
    """

    agent_id: int
    item_name: str
    old_price: float
    new_price: float


@dataclass(frozen=True)
class InnerThoughtRecord(TimedRecord):
    """A record for inner thought events.

    Corresponds to econsimulacra.logs.InnerThoughtLog.

    Attributes:
        type (str): must be "inner_thought".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that had the inner thought.
        inner_thought (str): The content of the inner thought.
    """

    agent_id: int
    inner_thought: str


@dataclass(frozen=True)
class TweetRecord(TimedRecord):
    """A record for tweet events.

    Corresponds to econsimulacra.logs.TweetLog.

    Attributes:
        type (str): must be "tweet".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that posted the tweet.
        message (str): The content of the tweet.
        num_follows (int): The number of accounts the agent follows.
        num_followers (int): The number of followers the agent has.
    """

    agent_id: int
    message: str
    num_follows: int
    num_followers: int


@dataclass(frozen=True)
class FollowRecord(TimedRecord):
    """A record for follow events.

    Corresponds to econsimulacra.logs.FollowLog.

    Attributes:
        type (str): must be "follow".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that followed another agent.
        target_agent_id (int): The ID of the agent being followed.
    """

    agent_id: int
    target_agent_id: int
    num_follows: int
    num_followers: int


@dataclass(frozen=True)
class UnfollowRecord(TimedRecord):
    """A record for unfollow events.

    Corresponds to econsimulacra.logs.UnfollowLog.

    Attributes:
        type (str): must be "unfollow".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that unfollowed another agent.
        target_agent_id (int): The ID of the agent being unfollowed.
    """

    agent_id: int
    target_agent_id: int
    num_follows: int
    num_followers: int


@dataclass(frozen=True)
class StateEvaluationRecord(TimedRecord):
    """A record for state evaluation events.

    Corresponds to econsimulacra.logs.StateEvaluationLog.

    Attributes:
        type (str): must be "state_evaluation".
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent whose state is being evaluated.
        wealth (float): The wealth of the agent.
        inventory (dict[str, float]): The inventory of the agent.
        persona (Optional[dict[str, Any]]): The persona of the agent.
    """

    agent_id: int
    wealth: float
    inventory: dict[str, float]
    persona: Optional[dict[str, Any]]


@dataclass(frozen=True)
class ObsRecord(TimedRecord):
    """A record for observation events.

    Corresponds to econsimulacra.logs.ObsLog.

    Attributes:
        type (str): must be "observation".
        obs_type (str): The type of the observation. See also: econsimulacra.logs.ObsLog.
        time (int | datetime): The timestamp of the log entry,
            as a datetime object or integer.
        time_step (int): The time step of the log entry, as an integer.
        agent_id (int): The ID of the agent that made the observation.
        obs (Any): The content of the observation.
    """

    obs_type: str
    agent_id: int
    obs: Any
