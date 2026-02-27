from abc import ABC, abstractmethod
import copy
from ..constant import DEFAULT_ACTION_JSON_SCHEMA
import json
import pathlib
from pathlib import Path
from typing import Any
from typing import Optional


class LLMClient(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config

    @abstractmethod
    async def generate_response(self, prompt: str) -> dict[str, Any]:
        pass

    def _get_json_schema(self, config: dict[str, Any]) -> str:
        """get JSON schema for structured generation from config or use default schema."""
        schema_path_str: Optional[str] = config.get("json_schema_path")
        if schema_path_str is not None:
            schema_path: Path = pathlib.Path(schema_path_str).resolve()
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found at: {schema_path}")
            json_schema: dict[str, Any] = json.loads(
                schema_path.read_text(encoding="utf-8")
            )
        else:
            json_schema = copy.deepcopy(DEFAULT_ACTION_JSON_SCHEMA)
        modify_schema: bool = config.get("modify_schema", False)
        if modify_schema:
            json_schema = self._modify_json_schema(json_schema, config)
        json_schema_str: str = json.dumps(json_schema, ensure_ascii=False)
        return json_schema_str

    def _modify_json_schema(
        self,
        json_schema: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        assert "properties" in json_schema
        if "gridSpace" in config:
            dim: int = len(config["gridSpace"])
            max_coordinate: int = max(config["gridSpace"])
            if "move" in json_schema["properties"]:
                json_schema["properties"]["move"]["items"]["minimum"] = 0
                json_schema["properties"]["move"]["items"]["maximum"] = max_coordinate
                json_schema["properties"]["move"]["minItems"] = dim
                json_schema["properties"]["move"]["maxItems"] = dim
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
            if "set_price" in json_schema["properties"]:
                json_schema["properties"]["set_price"]["items"]["properties"][
                    "item_name"
                ]["enum"] = item_names
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
        return json_schema
