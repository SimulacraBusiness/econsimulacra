from __future__ import annotations
from .base import LLMClient
import json
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai import RateLimitError
import os
import random
import time
from typing import Any
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
                - modelName: model name to use for generation (e.g., "gpt-4-0613").
                - apiKey: OpenAI API key (optional, can also be set via OPENAI_API_KEY environment variable).
                - jsonSchemaPath: path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - modifySchema: whether to modify the default JSON schema based on config (optional, default is False).
                - timeOut: timeout for API calls in seconds (optional, default is 30).
                - maxRetries: max retries for transient failures (optional, default is 3).

        Note: config example:
            {
                "modelName": "gpt-4o-mini",
                "apiKey": "your_openai_api_key" # Optional if OPENAI_API_KEY environment variable is set
            }
        """
        super().__init__(config, prng)
        api_key: Optional[str] = config.get("apiKey", os.getenv("OPENAI_API_KEY"))
        if api_key is None:
            raise ValueError(
                "OpenAIClient: API key must be provided in the config or set in the OPENAI_API_KEY environment variable."
            )
        time_out: float = config.get("timeOut", 30.0)
        max_retries: int = config.get("maxRetries", 3)
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=api_key, timeout=time_out, max_retries=max_retries
        )
        self.json_schema: dict[str, Any] = json.loads(self._get_json_schema())

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        while True:
            try:
                async with self._sem:
                    response: ChatCompletion = (
                        await self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "agent_action",
                                    "strict": True,
                                    "schema": self.json_schema,
                                },
                            },
                        )
                    )
                break
            except RateLimitError as e:
                time.sleep(1)
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
        return cast(dict[str, Any], parsed)
