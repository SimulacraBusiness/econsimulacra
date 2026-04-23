import asyncio
import copy
import json
import pathlib
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Type

from ..constant import DEFAULT_ACTION_JSON_SCHEMA


class LLMClient(ABC):
    """LLM Client class (abstract class).

    You can implement your own LLM client by inheriting this class and implementing the generate_response method.
    Currently, OpenAIClient and TransformersClient are implemented as built-in options.

    See also:
        - econsimulacra.llm_services.clients.OpenAIClient: LLM client implementation for OpenAI's API.
        - econsimulacra.llm_services.clients.TransformersClient: LLM client implementation using the Transformers library and Outlines for structured generation.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the LLM client. This must include:
                - modelName: model name or path to the model
                    (e.g., for OpenAI, "gpt-4"; for Transformers, a model name or path compatible with the Transformers library).
                and may include:
                - "jsonSchemaPath": path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - "modifySchema": whether to modify the default JSON schema based on config (optional, default is False).
                    See also: ._get_json_schema() and ._modify_json_schema()
                - "gridSpace": a list of two integers representing the dimensions of the grid space (optional, may be provided if modifySchema is True).
                - "items": a list of item names available in the environment (optional, may be provided if modifySchema is True).
                - "numAgents": the number of agents in the environment (optional, may be provided if modifySchema is True).
                - other model-specific parameters (e.g., for TransformersClient, "device", "dtype", "maxConcurrentGenerations", etc.).
            prng (random.Random, optional): An optional instance of random.Random for reproducible randomness.
                If not provided, a new instance will be created.

        Note:
            The LLMClient class is used as an environment service, and used by the LLMAgent.
            See also:
                econsimulacra.envs.base.Environment._generate_service_providers(service_provider_keys: list[str])
                econsimulacra.agents.llm_agent.LLMAgent
        """
        self.config: dict[str, Any] = config
        if "modelName" not in config:
            raise ValueError("'modelName' must be specified in the LLMClient config.")
        self.model_name: str = config["modelName"]
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: Optional[list[Type]] = registered_classes
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(config.get("maxConcurrentGenerations", 4))

    @abstractmethod
    async def generate_response(self, prompt: str) -> dict[str, Any]:
        pass

    def _get_json_schema(self) -> str:
        """Get JSON schema for structured generation from config or use default schema.

        Returns:
            A JSON schema string for structured generation.
        """
        schema_path_str: Optional[str] = self.config.get("jsonSchemaPath")
        if schema_path_str is not None:
            schema_path: Path = pathlib.Path(schema_path_str).resolve()
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found at: {schema_path}")
            json_schema: dict[str, Any] = json.loads(
                schema_path.read_text(encoding="utf-8")
            )
        else:
            json_schema = copy.deepcopy(DEFAULT_ACTION_JSON_SCHEMA)
        modify_schema: bool = self.config.get("modifySchema", False)
        if modify_schema:
            json_schema = self._modify_json_schema(json_schema, self.config)
        json_schema_str: str = json.dumps(json_schema, ensure_ascii=False)
        return json_schema_str

    def _modify_json_schema(
        self,
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
                json_schema["properties"]["set_prices"]["items"]["properties"][
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
            if "follow" in json_schema["properties"]:
                json_schema["properties"]["follow"]["anyOf"][0]["maximum"] = (
                    num_agents - 1
                )
            if "unfollow" in json_schema["properties"]:
                json_schema["properties"]["unfollow"]["anyOf"][0]["maximum"] = (
                    num_agents - 1
                )
        return json_schema
