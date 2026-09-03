from __future__ import annotations

import asyncio
import copy
import json
import os
import random
from typing import Any, Optional, Type, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from .base import LLMClient, LLMRecordConfig
from .llm_client_utils import save_response_record_from_chat_completion


class OpenAIClient(LLMClient):
    """OpenAI client for interacting with OpenAI's language models."""

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the OpenAI client. This may include parameters such as:
                - "modelName": model name to use for generation (e.g., "gpt-4-0613").
                - "apiKey": OpenAI API key (optional, can also be set via OPENAI_API_KEY environment variable).
                - "temperature": sampling temperature for generation (optional, default is 1.0).
                - "jsonSchemaPath": path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - "modifySchema": whether to modify the default JSON schema based on config (optional, default is False).
                - "gridSpace": a list of two integers representing the dimensions of the grid space (optional, may be provided if modifySchema is True).
                - "items": a list of item names available in the environment (optional, may be provided if modifySchema is True).
                - "numAgents": the number of agents in the environment (optional, may be provided if modifySchema is True).
                - "timeOut": timeout for API calls in seconds (optional, default is 30).
                - "maxRetries": max retries for transient failures (optional, default is 3).
                - "llmRecordSavePath": path to save the generated prompts (optional, for debugging purposes).
                - "saveNumTokens": whether to save the number of tokens in the generated response (optional, default is False).
                - "savePromptResponsePair": whether to save the prompt-response pair (optional, default is False).

        Note:
            config example::

                {
                    "type": "OpenAIClient",
                    "modelName": "gpt-4o-mini",
                    "apiKey": "your_openai_api_key"
                }
        """
        super().__init__(config, prng, registered_classes)
        api_key: Optional[str] = config.get("apiKey", os.getenv("OPENAI_API_KEY"))
        if api_key is None:
            raise ValueError(
                "OpenAIClient: API key must be provided in the config or set in the OPENAI_API_KEY environment variable."
            )
        time_out: float = config.get("timeOut", 30.0)
        self.max_retries: int = config.get("maxRetries", 3)
        self.temperature: float = config.get("temperature", 1.0)
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key, timeout=time_out, max_retries=self.max_retries
        )
        self.json_schema: dict[str, Any] = json.loads(self._get_json_schema())
        self.record_config: LLMRecordConfig = self._get_llm_record_config()

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        """Generate a response from the OpenAI API based on the given prompt.

        Args:
            prompt (str): The input prompt to send to the OpenAI API.

        Returns:
            dict[str, Any]: The parsed JSON response from the OpenAI API.

        Note:
            Existing callers use the client-level schema through the dynamic path.
        """
        return await self.generate_response_with_schema(
            prompt=prompt, json_schema=self.json_schema
        )

    async def generate_response_with_schema(
        self, prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a response using a request-specific JSON schema.

        Args:
            prompt (str): Input prompt sent to the OpenAI API.
            json_schema (dict[str, Any]): Schema for this individual request.

        Returns:
            dict[str, Any]: Parsed JSON response.

        Note:
            The schema is copied to keep concurrent agent requests isolated.
        """
        schema: dict[str, Any] = copy.deepcopy(json_schema)
        use_temperature: bool = True
        for _ in range(self.max_retries):
            try:
                # reasoning models (e.g. o-series, gpt-5.x) only accept the
                # default temperature, so drop it after a temperature rejection
                request: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "agent_action",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                }
                if use_temperature:
                    request["temperature"] = self.temperature
                async with self._sem:
                    response: ChatCompletion = (
                        await self.client.chat.completions.create(**request)
                    )
                break
            except BadRequestError as e:
                if use_temperature:
                    use_temperature = False
                    continue
                print(f"[BadRequestError] {e}")
                return {}
            except RateLimitError as e:
                print(f"[RateLimitError] {e}")
                await asyncio.sleep(1)
            except APITimeoutError as e:
                print(f"[APITimeoutError] {e}")
                await asyncio.sleep(1)
            except APIConnectionError as e:
                print(f"[APIConnectionError] {e}")
                await asyncio.sleep(1)
        content: Optional[str] = response.choices[0].message.content
        if content is None:
            raise ValueError("OpenAIClient: Received empty response from OpenAI API.")
        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"OpenAIClient: Failed to parse JSON response: {content}"
            ) from e
        if not isinstance(parsed, dict):
            raise ValueError(
                f"OpenAIClient: Expected JSON object in response, got: {parsed}"
            )
        save_response_record_from_chat_completion(
            response=response,
            prompt=prompt,
            record_config=self.record_config,
        )
        return cast(dict[str, Any], parsed)
