from __future__ import annotations

import re
from typing import Optional, Protocol

from .social import TweetIntent


class TweetTextGenerator(Protocol):
    """Protocol implemented by an asynchronous tweet text generator."""

    async def generate_text(self, prompt: str) -> str:
        """Generate plain text for one supplied prompt.

        Args:
            prompt: Fully constructed generation prompt.

        Returns:
            Generated plain text.
        """
        raise NotImplementedError


class TweetRenderer:
    """Render a rule-selected tweet intent with a small language model.

    Args:
        text_generator: Service that generates plain text asynchronously.
        language: Output language named in the prompt.
        max_characters: Maximum emitted tweet length.

    The renderer controls wording only. Follow decisions, tweet timing, topic,
    sentiment, and style have already been fixed by rules before this class is
    called.
    """

    def __init__(
        self,
        text_generator: TweetTextGenerator,
        language: str = "English",
        max_characters: int = 140,
    ) -> None:
        """Initialize the renderer.

        Args:
            text_generator: Asynchronous plain-text generation service.
            language: Requested output language.
            max_characters: Positive maximum length of the rendered message.

        Raises:
            ValueError: If ``max_characters`` is not positive.
        """
        if max_characters <= 0:
            raise ValueError("maxCharacters must be positive.")
        self.text_generator = text_generator
        self.language = language
        self.max_characters = max_characters

    async def generate_tweet(
        self,
        intent: TweetIntent,
        previous_tweet: Optional[str] = None,
    ) -> Optional[str]:
        """Generate and sanitize one tweet from a structured intent.

        Args:
            intent: Rule-selected tweet specification.
            previous_tweet: Most recent tweet used for exact duplicate removal.

        Returns:
            Sanitized tweet, or ``None`` after empty or duplicate generation.
        """
        prompt = self.generate_prompt(intent)
        generated_text = await self.text_generator.generate_text(prompt)
        tweet = self._sanitize_tweet(generated_text)
        if not tweet or tweet == (previous_tweet or "").strip():
            return None
        return tweet

    def generate_prompt(self, intent: TweetIntent) -> str:
        """Generate a compact grounding prompt for a Tiny LM.

        Args:
            intent: Rule-selected content specification.

        Returns:
            Prompt that asks only for surface realization of the intent.
        """
        return (
            "Rewrite SOURCE as one complete first-person social-media post.\n"
            "Keep every fact exactly. Add no facts, causes, advice, or explanation.\n"
            "Change you/your to I/my. Use at most 12 words. Return only the post.\n"
            f"Topic: {intent.topic}\n"
            f"Sentiment: {intent.sentiment}\n"
            f"Style: {intent.style}\n"
            f"Language: {self.language}\n"
            f"Maximum characters: {self.max_characters}\n"
            f"SOURCE: {intent.memory_excerpt}"
        )

    def _sanitize_tweet(self, generated_text: str) -> str:
        """Normalize generated text into one bounded social-media message.

        Args:
            generated_text: Raw Tiny LM output.

        Returns:
            Single-line text no longer than ``max_characters``.
        """
        tweet = re.sub(r"\s+", " ", generated_text).strip()
        tweet = re.sub(r"^(tweet|post)\s*:\s*", "", tweet, flags=re.IGNORECASE)
        tweet = tweet.strip(" \"'“”‘’")
        if len(tweet) <= self.max_characters:
            return tweet

        bounded_tweet = tweet[: self.max_characters - 1].rstrip()
        if " " in bounded_tweet:
            bounded_tweet = bounded_tweet.rsplit(" ", maxsplit=1)[0].rstrip()
        return f"{bounded_tweet}…"
