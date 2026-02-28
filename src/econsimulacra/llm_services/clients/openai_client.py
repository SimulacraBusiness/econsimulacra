import asyncio
from .base import LLMClient
from outlines import generate
from outlines import models
import os
import random
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional


class OpenAIClient(LLMClient):
    """OpenAI client for interacting with OpenAI's language models."""

    def __init__(
        self, config: dict[str, Any], prng: Optional[random.Random] = None
    ) -> None:
        """initialization.

        Args:
            config (dict): Configuration dictionary for the OpenAI client. This may include parameters such as:
                - model_name: model name to use for generation (e.g., "gpt-4-0613").
                - api_key: OpenAI API key (optional, can also be set via OPENAI_API_KEY environment variable).
                - json_schema_path: path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - modify_schema: whether to modify the default JSON schema based on config (optional, default is False).

        Note: config example:
            {
                "model_name": "gpt-4o-mini",
                "api_key": "your_openai_api_key" # Optional if OPENAI_API_KEY environment variable is set
            }
        """
        super().__init__(config, prng)
        if "model_name" not in config:
            raise ValueError(
                "OpenAIClient: 'model_name' must be specified in the config."
            )
        model_name: str = config["model_name"]
        api_key: Optional[str] = config.get("api_key", os.getenv("OPENAI_API_KEY"))
        if api_key is None:
            raise ValueError(
                "OpenAIClient: API key must be provided in the config or set in the OPENAI_API_KEY environment variable."
            )
        model = models.openai(
            model_name,
            api_key=api_key,
        )
        json_schema_str: str = self._get_json_schema(config)
        self.json_generator: Callable[[str], dict[str, Any]] = generate.json(
            model, schema_object=json_schema_str
        )
        self._lock = asyncio.Lock()

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        llm_response = await asyncio.to_thread(self.json_generator, prompt)
        return cast(dict[str, Any], llm_response)
