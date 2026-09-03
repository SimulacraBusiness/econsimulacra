import asyncio
import json
import random
from typing import Any, Callable, Optional, Type, cast

import torch
from outlines import generate, models

from .base import LLMClient


class TransformersClient(LLMClient):
    """Transformers client using Outlines for structured generation."""

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the Transformers client. This may include parameters such as:
                - "modelName": model name or path to the model in transformers.
                - "device": device to run the model on (e.g., "cpu", "cuda").
                - "dtype": data type for model weights (e.g., "float16", "int8").
                - "trust_remote_code": whether to trust remote code when loading the model.
                - "jsonSchemaPath": path to a custom JSON schema file for structured generation (optional, if not provided, a default schema will be used).
                - "modifySchema": whether to modify the default JSON schema based on config (optional, default is False).
                    See also: ._get_json_schema() and ._modify_json_schema()
                - "gridSpace": a list of two integers representing the dimensions of the grid space (optional, may be provided if modifySchema is True).
                - "items": a list of item names available in the environment (optional, may be provided if modifySchema is True).
                - "numAgents": the number of agents in the environment (optional, may be provided if modifySchema is True).

        Note:
            config example::

                {
                    "modelName": "meta-llama/Meta-Llama-3-8B-Instruct",
                    "device": "cuda",
                    "dtype": "float16",
                    "jsonSchemaPath": "path/to/schema.json"
                }
        """
        super().__init__(config, prng, registered_classes)
        device_str: str = config.get("device", "cuda")
        self.max_concurrent_generations: int = config.get("maxConcurrentGenerations", 2)
        if device_str == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA device specified but not available.")
            num_gpus: int = torch.cuda.device_count()
            gpu_ids: list[int] = config.get("gpuIds", list(range(num_gpus)))
            self.json_models: list[Any] = [
                self._make_json_model(gpu_id) for gpu_id in gpu_ids
            ]
            base_schema = self._get_json_schema()
            self.json_generators = [
                self._make_json_generator(model, base_schema)
                for model in self.json_models
            ]
            self._schema2generators: dict[
                str, list[Callable[[str], dict[str, Any]]]
            ] = {base_schema: self.json_generators}
            self.sems = [
                asyncio.Semaphore(self.max_concurrent_generations) for _ in gpu_ids
            ]
        else:
            raise ValueError(f"Unsupported device specified: {device_str}")
        self._rr: int = 0

    def _make_json_model(self, gpu_id: int) -> Any:
        """Load the model assigned to one GPU.

        Args:
            gpu_id (int): The ID of the GPU to use for this generator.

        Returns:
            Any: Outlines-compatible Transformers model.

        Note:
            Models are loaded once and reused by generators for multiple schemas.
        """
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
        return models.transformers(self.model_name, model_kwargs=model_kwargs)  # pyright: ignore

    def _make_json_generator(
        self, model: Any, json_schema: str
    ) -> Callable[[str], dict[str, Any]]:
        """Create a structured generator for a model and schema.

        Args:
            model (Any): Outlines-compatible model.
            json_schema (str): Serialized JSON action schema.

        Returns:
            Callable[[str], dict[str, Any]]: Structured response generator.

        Note:
            Generator creation is cached by schema in ``generate_response_with_schema``.
        """
        return generate.json(model, schema_object=json_schema)  # pyright: ignore

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        """Generate a response from the model based on the given prompt.

        Args:
            prompt (str): The input prompt to send to the model.

        Returns:
            dict[str, Any]: The parsed JSON response from the model.

        Note:
            Existing callers continue to use generators built from the base schema.
        """
        return await self._generate_with_generators(prompt, self.json_generators)

    async def generate_response_with_schema(
        self, prompt: str, json_schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Generate a response using a request-specific action schema.

        Args:
            prompt (str): Input prompt sent to the model.
            json_schema (dict[str, Any]): Schema for this individual request.

        Returns:
            dict[str, Any]: Parsed model response.

        Note:
            Generators are cached by serialized schema while loaded models are reused.
        """
        schema_key = json.dumps(json_schema, ensure_ascii=False, sort_keys=True)
        async with self._lock:
            generators = self._schema2generators.get(schema_key)
            if generators is None:
                generators = [
                    self._make_json_generator(model, schema_key)
                    for model in self.json_models
                ]
                self._schema2generators[schema_key] = generators
        return await self._generate_with_generators(prompt, generators)

    async def _generate_with_generators(
        self,
        prompt: str,
        generators: list[Callable[[str], dict[str, Any]]],
    ) -> dict[str, Any]:
        """Dispatch one prompt across a collection of GPU generators.

        Args:
            prompt (str): Input prompt sent to the model.
            generators (list[Callable[[str], dict[str, Any]]]): Per-GPU generators.

        Returns:
            dict[str, Any]: Parsed model response.

        Note:
            Round-robin selection and per-GPU concurrency limits are preserved.
        """
        async with self._lock:
            i = self._rr
            self._rr = (self._rr + 1) % len(generators)
        async with self.sems[i]:
            llm_response = await asyncio.to_thread(generators[i], prompt)
        return cast(dict[str, Any], llm_response)
