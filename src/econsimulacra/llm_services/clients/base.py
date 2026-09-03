import asyncio
import copy
import json
import pathlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

from ...mobility import build_action_schema_with_mobility
from ..constant import DEFAULT_ACTION_JSON_SCHEMA
from .llm_client_utils import modify_schema


@dataclass
class LLMRecordConfig:
    """Configuration for recording LLM prompts and responses."""

    save_path: Optional[str] = None
    save_num_tokens: bool = False
    save_prompt_response_pair: bool = False


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
                - "disabledActions": a list of action types to disable. If not provided, no actions will be disabled.
                - "jsonSchemaPath": path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - "modifySchema": whether to modify the default JSON schema based on config (optional, default is False).
                    See also: ._get_json_schema() and ._modify_json_schema()
                - "gridSpace": a list of two integers representing the dimensions of the grid space (optional, may be provided if modifySchema is True).
                - "items": a list of item names available in the environment (optional, may be provided if modifySchema is True).
                - "numAgents": the number of agents in the environment (optional, may be provided if modifySchema is True).
                - "llmRecordSavePath": path to save the generated prompts (optional, for debugging purposes).
                - "saveNumTokens": whether to save the number of tokens in the generated response (optional, default is False).
                - "savePromptResponsePair": whether to save the prompt-response pair (optional, default is False).
                - other model-specific parameters (e.g., for TransformersClient, "device", "dtype", "maxConcurrentGenerations", etc.).
            prng (random.Random, optional): An optional instance of random.Random for reproducible randomness.
                If not provided, a new instance will be created.

        Note:
            The LLMClient class is used as an environment service, and used by the LLMAgent.
            disabledActions in the LLMClient config must be aligned with that in the PromptBuilder config if the PromptBuilder is used in conjunction with the LLMClient to ensure consistency between the prompt and the expected response format.
            See also: econsimulacra.envs.base.Environment._generate_service_providers,
            econsimulacra.agents.llm_agent.LLMAgent
        """
        self.config: dict[str, Any] = config
        if "modelName" not in config:
            raise ValueError("'modelName' must be specified in the LLMClient config.")
        self.model_name: str = config["modelName"]
        self.disabled_actions: list[str] = config.get("disabledActions", [])
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = registered_classes
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(config.get("maxConcurrentGenerations", 4))

    @abstractmethod
    async def generate_response(self, prompt: str) -> dict[str, Any]:
        pass

    async def generate_response_with_schema(
        self, prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a response using a request-specific action schema when supported.

        Args:
            prompt (str): Input prompt sent to the model.
            json_schema (dict[str, Any]): Schema for this individual request.

        Returns:
            dict[str, Any]: Parsed model response.

        Note:
            The compatibility implementation delegates to ``generate_response``.
            Built-in structured clients override this method.
        """
        return await self.generate_response(prompt)

    def get_action_schema(self, mobility_names: list[str]) -> dict[str, Any]:
        """Build an action schema for one agent's available mobility modes.

        Args:
            mobility_names (list[str]): Mobility names available to the agent.

        Returns:
            dict[str, Any]: Isolated, request-specific action schema.

        Note:
            The stored or configured base schema is not mutated.
        """
        base_schema = json.loads(self._get_json_schema())
        return build_action_schema_with_mobility(base_schema, mobility_names)

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
        for action in self.disabled_actions:
            if action in json_schema["properties"]:
                del json_schema["properties"][action]
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
        return modify_schema(json_schema, config)

    def _get_llm_record_config(self) -> LLMRecordConfig:
        """Extract LLM record configuration from the main config.

        Returns:
            An instance of LLMRecordConfig containing the recording configuration.
        """
        save_path_str: Optional[str] = self.config.get("llmRecordSavePath")
        if save_path_str is not None:
            save_path: Path = pathlib.Path(save_path_str).resolve()
            if not save_path.parent.exists():
                save_path.parent.mkdir(parents=True, exist_ok=True)
        save_num_tokens: bool = self.config.get("saveNumTokens", False)
        if save_num_tokens and save_path_str is None:
            raise ValueError(
                "LLMClient: 'saveNumTokens' is set to True, "
                "but 'llmRecordSavePath' is not provided."
            )
        save_prompt_response_pair: bool = self.config.get(
            "savePromptResponsePair", False
        )
        if save_prompt_response_pair and save_path_str is None:
            raise ValueError(
                "LLMClient: 'savePromptResponsePair' is set to True, "
                "but 'llmRecordSavePath' is not provided."
            )
        if save_path_str is not None:
            if not save_num_tokens and not save_prompt_response_pair:
                raise ValueError(
                    "LLMClient: 'llmRecordSavePath' is provided, "
                    "but neither 'saveNumTokens' nor 'savePromptResponsePair' is set to True."
                )
        return LLMRecordConfig(
            save_path=save_path_str,
            save_num_tokens=save_num_tokens,
            save_prompt_response_pair=save_prompt_response_pair,
        )
