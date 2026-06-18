from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from openai.types.chat import ChatCompletion

if TYPE_CHECKING:
    from .base import LLMRecordConfig


def modify_schema(
    json_schema: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Modify the default JSON schema based on the provided config.

    Args:
        json_schema: The default JSON schema.
        config: The configuration dictionary.

    Returns:
        The modified JSON schema.

    Note:
        This method restricts the action space defined in the JSON schema
        based on the environment configuration.
    """
    assert "properties" in json_schema
    if "gridSpace" in config:
        dim: int = len(config["gridSpace"])
        max_coordinate: int = max(config["gridSpace"])
        if "move" in json_schema["properties"]:
            json_schema["properties"]["move"]["anyOf"][0]["items"]["minimum"] = 0
            json_schema["properties"]["move"]["anyOf"][0]["items"]["maximum"] = (
                max_coordinate
            )
            json_schema["properties"]["move"]["anyOf"][0]["minItems"] = dim
            json_schema["properties"]["move"]["anyOf"][0]["maxItems"] = dim
    if "items" in config:
        item_names: list[str] = list(config["items"])
        if "consumptions" in json_schema["properties"]:
            json_schema["properties"]["consumptions"]["items"]["properties"][
                "item_name"
            ]["enum"] = item_names
        if "orders" in json_schema["properties"]:
            json_schema["properties"]["orders"]["items"]["properties"]["item_name"][
                "enum"
            ] = item_names
        if "proposals" in json_schema["properties"]:
            json_schema["properties"]["proposals"]["items"]["properties"][
                "give_item_name"
            ]["enum"] = item_names
            json_schema["properties"]["proposals"]["items"]["properties"][
                "get_item_name"
            ]["enum"] = item_names
        if "set_prices" in json_schema["properties"]:
            json_schema["properties"]["set_prices"]["items"]["properties"]["item_name"][
                "enum"
            ] = item_names
    if "numAgents" in config:
        num_agents: int = config["numAgents"]
        if "orders" in json_schema["properties"]:
            json_schema["properties"]["orders"]["items"]["properties"][
                "counterparty_id"
            ]["minimum"] = 0
            json_schema["properties"]["orders"]["items"]["properties"][
                "counterparty_id"
            ]["maximum"] = num_agents - 1
        if "proposals" in json_schema["properties"]:
            json_schema["properties"]["proposals"]["items"]["properties"][
                "responder_agent_id"
            ]["minimum"] = 0
            json_schema["properties"]["proposals"]["items"]["properties"][
                "responder_agent_id"
            ]["maximum"] = num_agents - 1
        if "follow" in json_schema["properties"]:
            json_schema["properties"]["follow"]["anyOf"][0]["maximum"] = num_agents - 1
        if "unfollow" in json_schema["properties"]:
            json_schema["properties"]["unfollow"]["anyOf"][0]["maximum"] = (
                num_agents - 1
            )
    return json_schema


def save_response_record_from_chat_completion(
    response: ChatCompletion,
    prompt: str,
    record_config: LLMRecordConfig,
) -> None:
    """Save the response record from a ChatCompletion response.

    Args:
        response: The ChatCompletion response object.
        prompt: The prompt that was sent to the LLM.
        record_config: The configuration for saving the record.
    """
    if record_config.save_path is not None:
        record: dict[str, Optional[str | int]] = {
            "wall_clock_time": datetime.fromtimestamp(
                response.created
            ).isoformat()
        }
        if record_config.save_num_tokens:
            if response.usage is not None and response.usage.prompt_tokens is not None:
                record["num_tokens"] = int(response.usage.prompt_tokens)
            else:
                record["num_tokens"] = None
        if record_config.save_prompt_response_pair:
            record["prompt"] = prompt
            content: Optional[str] = response.choices[0].message.content
            record["response"] = content
        with open(record_config.save_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
