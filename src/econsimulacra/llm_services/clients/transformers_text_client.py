from __future__ import annotations

import asyncio
import random
from typing import Any, Optional, Type

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .text_base import TextGenerationClient


class TransformersTextClient(TextGenerationClient):
    """Generate short plain text with a local Transformers causal LM.

    Args:
        config: Service configuration. It must contain ``modelName`` and may
            contain ``device``, ``dtype``, ``maxModelParameters``,
            ``maxNewTokens``, ``temperature``, ``topP``,
            ``repetitionPenalty``, ``maxPromptTokens``,
            ``maxConcurrentGenerations``, ``numThreads``, ``trustRemoteCode``, and
            ``ignoreGenerationErrors``.
        prng: Optional simulation pseudo-random generator.
        registered_classes: Registered classes accepted for compatibility with
            the environment service factory; unused by this implementation.

    Unlike :class:`TransformersClient`, this service does not use an action
    JSON schema. It is intended for bounded surface realization after another
    component has already selected the semantic content.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Load tokenizer and model and validate the configured size limit.

        Args:
            config: Text-generation service configuration.
            prng: Seeded pseudo-random generator.
            registered_classes: Environment-registered classes; unused.

        Raises:
            ValueError: If configuration is invalid or the loaded model exceeds
                ``maxModelParameters``.
        """
        del registered_classes
        if "modelName" not in config:
            raise ValueError("'modelName' must be specified in text client config.")
        self.config = config
        self.model_name = str(config["modelName"])
        self.prng = prng if prng is not None else random.Random()
        self.device = str(config.get("device", "cpu"))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but not available.")
        self.max_new_tokens = int(config.get("maxNewTokens", 64))
        self.max_prompt_tokens = int(config.get("maxPromptTokens", 512))
        self.temperature = float(config.get("temperature", 0.8))
        self.top_p = float(config.get("topP", 0.9))
        self.repetition_penalty = float(config.get("repetitionPenalty", 1.1))
        self.is_generation_error_ignored = bool(
            config.get("ignoreGenerationErrors", False)
        )
        if self.max_new_tokens <= 0 or self.max_prompt_tokens <= 0:
            raise ValueError("Token limits must be positive.")
        if self.temperature < 0:
            raise ValueError("temperature must be nonnegative.")

        dtype_name = str(config.get("dtype", "float32"))
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        trust_remote_code = bool(config.get("trustRemoteCode", False))
        num_threads = config.get("numThreads")
        if num_threads is not None:
            if int(num_threads) <= 0:
                raise ValueError("numThreads must be positive.")
            torch.set_num_threads(int(num_threads))
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        parameter_count = int(self.model.num_parameters())
        max_model_parameters = int(config.get("maxModelParameters", 1_000_000_000))
        if parameter_count > max_model_parameters:
            raise ValueError(
                f"Model has {parameter_count} parameters, exceeding the configured "
                f"maximum of {max_model_parameters}."
            )
        getattr(self.model, "to")(torch.device(self.device))
        self.model.eval()
        self._seed_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(
            int(config.get("maxConcurrentGenerations", 1))
        )

    async def generate_text(self, prompt: str) -> str:
        """Generate plain text without returning the input prompt tokens.

        Args:
            prompt: Prompt to format with the tokenizer chat template.

        Returns:
            Decoded generated continuation. If ``ignoreGenerationErrors`` is
            enabled, failures produce an empty string.
        """
        async with self._seed_lock:
            generation_seed = self.prng.randrange(0, 2**63 - 1)
        try:
            async with self._semaphore:
                if self.device == "cpu":
                    return self._generate_text_sync(prompt, generation_seed)
                return await asyncio.to_thread(
                    self._generate_text_sync,
                    prompt,
                    generation_seed,
                )
        except Exception:
            if self.is_generation_error_ignored:
                return ""
            raise

    def _generate_text_sync(self, prompt: str, generation_seed: int) -> str:
        """Run one blocking Transformers generation.

        Args:
            prompt: User prompt.
            generation_seed: Per-request seed drawn serially from the simulation
                pseudo-random generator.

        Returns:
            Decoded continuation only.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            formatted_prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt
        encoded = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_prompt_tokens,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        torch.manual_seed(generation_seed)
        if self.device.startswith("cuda"):
            torch.cuda.manual_seed_all(generation_seed)
        is_sampling_enabled = self.temperature > 0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": is_sampling_enabled,
            "repetition_penalty": self.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if is_sampling_enabled:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p
        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generation_kwargs)
        continuation_ids = output_ids[0, input_length:]
        return str(
            self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        ).strip()
