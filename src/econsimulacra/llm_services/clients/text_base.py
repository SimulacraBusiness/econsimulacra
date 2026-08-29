from __future__ import annotations

from abc import ABC, abstractmethod


class TextGenerationClient(ABC):
    """Abstract service for asynchronous plain-text generation.

    This interface is intentionally separate from :class:`LLMClient`, whose
    response contract is a structured action dictionary. Text-only consumers,
    such as the household tweet renderer, should not generate or parse the
    complete simulation action schema.
    """

    @abstractmethod
    async def generate_text(self, prompt: str) -> str:
        """Generate plain text for one prompt.

        Args:
            prompt: Text-generation prompt.

        Returns:
            Generated text without a structured action wrapper.
        """
        raise NotImplementedError
