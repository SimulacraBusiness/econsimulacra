import asyncio
from .base import LLMClient
from outlines import generate
from outlines import models
import random
import torch
from typing import Any
from typing import Callable
from typing import cast
from typing import Optional


class TransformersClient(LLMClient):
    """Transformers client using Outlines for structured generation."""

    def __init__(
        self, config: dict[str, Any], prng: Optional[random.Random] = None
    ) -> None:
        """initialization.

        Args:
            config (dict): Configuration dictionary for the Transformers client. This may include parameters such as:
                - modelName: model name or path to the model in transformers.
                - device: device to run the model on (e.g., "cpu", "cuda").
                - dtype: data type for model weights (e.g., "float16", "int8").
                - trust_remote_code: whether to trust remote code when loading the model.
                - jsonSchemaPath: path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - modifySchema: whether to modify the default JSON schema based on config (optional, default is False).

        Note: config example:
            {
                "modelName": "meta-llama/Meta-Llama-3-8B-Instruct",
                "device": "cuda",
                "dtype": "float16",
                "jsonSchemaPath": "path/to/schema.json"
            }
        """
        super().__init__(config, prng)
        device_str: str = config.get("device", "cuda")
        self.max_concurrent_generations: int = config.get("maxConcurrentGenerations", 2)
        if device_str == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA device specified but not available.")
            num_gpus: int = torch.cuda.device_count()
            gpu_ids: list[int] = config.get("gpuIds", list(range(num_gpus)))
            self.json_generators: list[Callable[[str], dict[str, Any]]] = [
                self._make_json_generator(gpu_id) for gpu_id in gpu_ids
            ]
            self.sems = [
                asyncio.Semaphore(self.max_concurrent_generations) for _ in gpu_ids
            ]
        else:
            raise ValueError(f"Unsupported device specified: {device_str}")
        self._rr: int = 0

    def _make_json_generator(self, gpu_id: int) -> Callable[[str], dict[str, Any]]:
        """create a JSON generator for a specific GPU."""
        device_str: str = f"cuda:{gpu_id}"
        dtype_str: str = self.config.get("dtype", "float16")
        dtype: torch.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_str, torch.bfloat16)
        trust_remote_code: bool = self.config.get("trustRemoteCode", False)
        model_kwargs: dict[str, Any] = {
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "device_map": {"": device_str},
        }
        model = models.transformers(self.model_name, model_kwargs=model_kwargs)
        json_schema_str: str = self._get_json_schema()
        return generate.json(model, schema_object=json_schema_str)

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        async with self._lock:
            i = self._rr
            self._rr = (self._rr + 1) % len(self.json_generators)
        async with self.sems[i]:
            llm_response = await asyncio.to_thread(self.json_generators[i], prompt)
        return cast(dict[str, Any], llm_response)
