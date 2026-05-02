from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .records import (
    BaseRecord,
    AgentGenerationRecord,
    SpaceAssignRecord,
    MoveRecord,
    ConsumptionRecord,
    OrderRecord,
    ProposalRecord,
    OrderReactionRecord,
    OrderExpirationRecord,
    ProposalReactionRecord,
    ProposalExpirationRecord,
    ChangePriceRecord,
    TweetRecord,
    FollowRecord,
    UnfollowRecord,
    StateEvaluationRecord,
)


def parse_time(value: str | int) -> datetime | int:
    """Parse a time value from a string or integer."""
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError(f"Invalid time value: {value}")


def to_pos(value: list[int]) -> tuple[int, ...]:
    """Convert a list of integers to a tuple of integers (position format)."""
    return tuple(int(value[i]) for i in range(len(value)))


def extract_prefixed_values(value: dict[str, Any], prefix: str) -> dict[str, float]:
    """Extract values with a given prefix and convert them to floats."""
    result: dict[str, float] = {}
    prefix_with_sep = f"{prefix}_"

    for key, val in value.items():
        if key.startswith(prefix_with_sep):
            key_name = key[len(prefix_with_sep) :]
            result[key_name] = float(val)

    return result


def to_inventory(value: dict[str, Any]) -> dict[str, float]:
    return extract_prefixed_values(value, "inventory")


def to_persona(value: dict[str, Any]) -> dict[str, float]:
    return extract_prefixed_values(value, "persona")


def parse_record(record: dict[str, Any]) -> BaseRecord:
    record_type = record["type"]

    if record_type == "agent_generation":
        return AgentGenerationRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            agent_type=str(record["agent_type"]),
            agent_name=str(record["agent_name"]),
            wealth=float(record["wealth"]),
            inventory=to_inventory(record),
            persona=to_persona(record),
        )

    if record_type == "space_assign":
        return SpaceAssignRecord(
            type=record_type,
            agent_id=int(record["agent_id"]),
            pos=to_pos(record["pos"]),
        )

    if record_type == "move":
        return MoveRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            old_pos=to_pos(record["old_pos"]),
            new_pos=to_pos(record["new_pos"]),
        )

    if record_type == "consumption":
        return ConsumptionRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            item_name=str(record["item_name"]),
            item_amount=float(record["item_amount"]),
        )

    if record_type == "order":
        return OrderRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            counterparty_id=int(record["counterparty_id"]),
            item_name=str(record["item_name"]),
            item_amount=float(record["item_amount"]),
            price=float(record["price"]) if record["price"] is not None else None,
            order_id=int(record["order_id"]),
        )

    if record_type == "proposal":
        return ProposalRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            proposal_id=int(record["proposal_id"]),
            proposer_agent_id=int(record["proposer_agent_id"]),
            responder_agent_id=int(record["responder_agent_id"]),
            give_item_name=str(record["give_item_name"]),
            give_item_amount=float(record["give_item_amount"]),
            get_item_name=str(record["get_item_name"]),
            get_item_amount=float(record["get_item_amount"]),
        )

    if record_type == "order_reaction":
        return OrderReactionRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            counterparty_id=int(record["counterparty_id"]),
            item_name=str(record["item_name"]),
            item_amount=float(record["item_amount"]),
            price=float(record["price"]),
            order_id=int(record["order_id"]),
            accept_amount=float(record["accept_amount"]),
        )

    if record_type == "order_expiration":
        return OrderExpirationRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            order_id=int(record["order_id"]),
        )

    if record_type == "proposal_reaction":
        return ProposalReactionRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            proposal_id=int(record["proposal_id"]),
            proposer_agent_id=int(record["proposer_agent_id"]),
            responder_agent_id=int(record["responder_agent_id"]),
            give_item_name=str(record["give_item_name"]),
            give_item_amount=float(record["give_item_amount"]),
            get_item_name=str(record["get_item_name"]),
            get_item_amount=float(record["get_item_amount"]),
            accept=bool(record["accept"]),
        )

    if record_type == "proposal_expiration":
        return ProposalExpirationRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            proposal_id=int(record["proposal_id"]),
        )

    if record_type == "change_price":
        return ChangePriceRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            item_name=str(record["item_name"]),
            old_price=float(record["old_price"]),
            new_price=float(record["new_price"]),
        )

    if record_type == "tweet":
        return TweetRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            message=str(record["message"]),
            num_follows=int(record["num_follows"]),
            num_followers=int(record["num_followers"]),
        )

    if record_type == "follow":
        return FollowRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            target_agent_id=int(record["target_agent_id"]),
            num_follows=int(record["num_follows"]),
            num_followers=int(record["num_followers"]),
        )

    if record_type == "unfollow":
        return UnfollowRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            target_agent_id=int(record["target_agent_id"]),
            num_follows=int(record["num_follows"]),
            num_followers=int(record["num_followers"]),
        )

    if record_type == "state_evaluation":
        return StateEvaluationRecord(
            type=record_type,
            time=parse_time(record["time"]),
            time_step=int(record["time_step"]),
            agent_id=int(record["agent_id"]),
            wealth=float(record["wealth"]),
            inventory=to_inventory(record),
            persona=to_persona(record),
        )

    raise ValueError(f"Unknown Record type: {record_type}")


def load_from_file(file_path: str) -> list[BaseRecord]:
    """Load records from a log file.

    Args:
        file_path (str): The path to the log file in JSON Lines format.

    Returns:
        list[BaseRecord]: A list of parsed records.

    Notes:
        econsimulacra.logs.DictLogger.save() produces log files in the expected format.
    """
    records: list[BaseRecord] = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            record: dict[str, Any] = json.loads(line)
            records.append(parse_record(record))
    return records
