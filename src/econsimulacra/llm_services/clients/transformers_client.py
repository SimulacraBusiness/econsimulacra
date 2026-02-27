import asyncio
from .base import LLMClient
from outlines import generate
from outlines import models
import torch
from typing import Any
from typing import Callable
from typing import cast


class TransformersClient(LLMClient):
    """Transformers client using Outlines for structured generation."""

    def __init__(self, config: dict[str, Any]) -> None:
        """initialization.

        Args:
            config (dict): Configuration dictionary for the Transformers client. This may include parameters such as:
                - model_name: model name or path to the model in transformers.
                - device: device to run the model on (e.g., "cpu", "cuda").
                - dtype: data type for model weights (e.g., "float16", "int8").
                - trust_remote_code: whether to trust remote code when loading the model.
                - json_schema_path: path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - modify_schema: whether to modify the default JSON schema based on config (optional, default is False).

        Note: config example:
            {
                "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
                "device": "cuda",
                "dtype": "float16",
                "max_new_tokens": 256,
                "json_schema_path": "path/to/schema.json"
            }
        """
        super().__init__(config)
        if "model_name" not in config:
            raise ValueError(
                "TransformersClient: 'model_name' must be specified in the config."
            )
        model_name: str = config["model_name"]
        device_str: str = config.get("device", "cuda")
        dtype_str: str = config.get("dtype", "float16")
        dtype: torch.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_str, torch.bfloat16)
        trust_remote_code: bool = config.get("trust_remote_code", False)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if device_str == "cuda":
            model_kwargs["device_map"] = "auto"
        model = models.transformers(model_name, model_kwargs=model_kwargs)
        json_schema_str: str = self._get_json_schema(config)
        self.json_generator: Callable[[str], dict[str, Any]] = generate.json(
            model, schema_object=json_schema_str
        )
        self._lock = asyncio.Lock()

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        llm_response = await asyncio.to_thread(self.json_generator, prompt)
        return cast(dict[str, Any], llm_response)
