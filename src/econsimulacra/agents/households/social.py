from __future__ import annotations

import math
import re
from dataclasses import dataclass
from random import Random
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeAlias

from .states import SocialState

if TYPE_CHECKING:
    from .policy import ActionCapabilities
    from .states import DecisionContext, HouseholdState

TweetTopic: TypeAlias = Literal[
    "daily_life",
    "consumption",
    "shopping",
    "finances",
    "health",
    "mobility",
    "social",
]
TweetSentiment: TypeAlias = Literal["positive", "neutral", "negative"]
TweetStyle: TypeAlias = Literal[
    "casual",
    "informative",
    "reflective",
    "enthusiastic",
    "terse",
]


@dataclass(frozen=True)
class TweetIntent:
    """Describe tweet content before a language model realizes its wording.

    Args:
        topic: Rule-selected subject of the tweet.
        sentiment: Rule-selected emotional polarity.
        style: Rule-selected writing style.
        memory_excerpt: Relevant summarized memory supplied as grounding.

    The intent is deliberately model-independent. A text generator may express
    it in natural language, but it must not choose whether to tweet or change
    the topic, sentiment, or style.
    """

    topic: TweetTopic
    sentiment: TweetSentiment
    style: TweetStyle
    memory_excerpt: str


@dataclass(frozen=True)
class SocialDecision:
    """Represent one rule-based social-network decision.

    Args:
        follow_agent_id: Recommended agent to follow, if any.
        unfollow_agent_id: Currently followed agent to unfollow, if any.
        tweet_intent: Structured tweet specification, if a tweet event occurs.
    """

    follow_agent_id: Optional[int] = None
    unfollow_agent_id: Optional[int] = None
    tweet_intent: Optional[TweetIntent] = None


class _SocialMediaPolicyRules:
    r"""Generate household SNS decisions from explicit behavioral rules.

    Tweet event times follow a discretized exponential-kernel Hawkes process.
    Before sampling at step :math:`t`, excitation is updated as

    .. math::

       h_t=h_{t^-}e^{-\beta\Delta t}+\sum_k w_k I_{k,t},\qquad
       \lambda_t=\mu+h_t,

    where :math:`I_{k,t}` indicates a changed memory category. The probability
    of at least one event in a unit simulation step is

    .. math::

       P(N_t-N_{t-1}>0)=1-e^{-\lambda_t}.

    A successfully realized tweet adds the self-excitation impulse :math:`\alpha`.
    This permits bursts without imposing a minimum inter-event interval.

    Args:
        config: ``socialRule`` household configuration.
        prng: Shared seeded pseudo-random generator.
    """

    _MEMORY_TOPIC: dict[str, TweetTopic] = {
        "consumption_history": "consumption",
        "purchase_history": "shopping",
        "sale_history": "finances",
        "exchange_history": "finances",
        "set_price_history": "finances",
        "state_evaluation_history": "finances",
        "sleep_history": "health",
        "move_history": "mobility",
        "social_history": "social",
        "tweet_history": "social",
        "inner_thought_history": "daily_life",
        "invalid_action_history": "daily_life",
    }
    _TOPIC_MEMORY: dict[TweetTopic, tuple[str, ...]] = {
        "daily_life": ("inner_thought_history", "invalid_action_history"),
        "consumption": ("consumption_history",),
        "shopping": ("purchase_history", "set_price_history"),
        "finances": (
            "state_evaluation_history",
            "sale_history",
            "exchange_history",
        ),
        "health": ("sleep_history", "consumption_history"),
        "mobility": ("move_history",),
        "social": ("social_history", "tweet_history"),
    }

    def __init__(self, config: dict[str, Any], prng: Random) -> None:
        """Initialize rule parameters and empty social state.

        Args:
            config: ``socialRule`` household configuration.
            prng: Shared seeded pseudo-random generator.

        Raises:
            ValueError: If a Hawkes intensity parameter is negative or the
                exponential decay rate is not positive.
        """
        self.config = config
        self.prng = prng
        self.state = SocialState()

        tweet_rule: dict[str, Any] = config.get("tweet", {})
        self.base_intensity = float(tweet_rule.get("baseIntensity", 0.002))
        self.self_excitation = float(tweet_rule.get("selfExcitation", 0.18))
        self.decay_rate = float(tweet_rule.get("decayRate", 0.7))
        self.memory_excitation = float(tweet_rule.get("memoryExcitation", 0.025))
        self.stress_excitation_scale = float(
            tweet_rule.get("stressExcitationScale", 0.001)
        )
        self.max_memory_excerpt_characters = int(
            tweet_rule.get("maxMemoryExcerptCharacters", 320)
        )
        if self.base_intensity < 0 or self.self_excitation < 0:
            raise ValueError("Hawkes intensities must be nonnegative.")
        if self.memory_excitation < 0 or self.stress_excitation_scale < 0:
            raise ValueError("Memory excitation parameters must be nonnegative.")
        if self.decay_rate <= 0:
            raise ValueError("Hawkes decayRate must be positive.")
        if self.max_memory_excerpt_characters <= 0:
            raise ValueError("maxMemoryExcerptCharacters must be positive.")

        self.topic_priority: tuple[TweetTopic, ...] = tuple(
            config.get(
                "topicPriority",
                (
                    "health",
                    "finances",
                    "shopping",
                    "social",
                    "consumption",
                    "mobility",
                    "daily_life",
                ),
            )
        )
        follow_rule: dict[str, Any] = config.get("follow", {})
        self.follow_probability = float(follow_rule.get("probability", 0.01))
        self.follow_cooldown_steps = int(follow_rule.get("cooldownSteps", 24))
        unfollow_rule: dict[str, Any] = config.get("unfollow", {})
        self.unfollow_probability = float(unfollow_rule.get("probability", 0.005))
        self.unfollow_cooldown_steps = int(unfollow_rule.get("cooldownSteps", 48))
        self.empty_tweet_steps = int(unfollow_rule.get("emptyTweetSteps", 72))
        self.negative_keywords = tuple(
            str(value).lower()
            for value in unfollow_rule.get(
                "negativeKeywords", ("hate", "scam", "spam", "嫌い", "詐欺")
            )
        )

    def generate_social_decision(
        self,
        context: DecisionContext,
        household_state: HouseholdState,
        capabilities: ActionCapabilities,
    ) -> SocialDecision:
        """Generate follow, unfollow, and tweet-intent decisions for one step.

        Args:
            context: Normalized current observation.
            household_state: Current physiological and activity state.
            capabilities: Enabled action keys for the household.

        Returns:
            Rule-based decision. Its tweet intent still requires text rendering.
        """
        if (
            context.obs.get("self_is_sleeping", False)
            or household_state.mode == "SLEEPING"
        ):
            self._update_hawkes_intensity(context)
            return SocialDecision()

        unfollow_agent_id = self.get_unfollow_agent_id(context, capabilities)
        follow_agent_id = self.get_follow_agent_id(
            context,
            capabilities,
            unfollow_agent_id=unfollow_agent_id,
        )
        tweet_intent: Optional[TweetIntent] = None
        self._update_hawkes_intensity(context)
        if capabilities.is_enabled("tweet") and self.should_tweet():
            tweet_intent = self.generate_tweet_intent(context)
        return SocialDecision(
            follow_agent_id=follow_agent_id,
            unfollow_agent_id=unfollow_agent_id,
            tweet_intent=tweet_intent,
        )

    def should_tweet(self) -> bool:
        """Return whether the current Hawkes intensity produces a tweet event.

        Returns:
            ``True`` when a seeded Bernoulli draw succeeds for the probability
            implied by the current Hawkes intensity.
        """
        intensity = self.get_hawkes_intensity()
        event_probability = 1.0 - math.exp(-intensity)
        return self.prng.random() < event_probability

    def get_hawkes_intensity(self) -> float:
        """Get the nonnegative tweet intensity at the current step.

        Returns:
            Baseline intensity plus current decayed excitation.
        """
        return max(0.0, self.base_intensity + self.state.hawkes_excitation)

    def record_generated_tweet(self, intent: TweetIntent) -> None:
        """Record a successful tweet and add its self-excitation impulse.

        Args:
            intent: Intent whose text was successfully generated and emitted.
        """
        self.state.hawkes_excitation += self.self_excitation
        self.state.last_tweet_intent = intent

    def get_unfollow_agent_id(
        self,
        context: DecisionContext,
        capabilities: ActionCapabilities,
    ) -> Optional[int]:
        """Get one currently followed agent satisfying unfollow rules.

        Args:
            context: Current observation containing the visible timeline.
            capabilities: Enabled household actions.

        Returns:
            Target agent identifier, or ``None`` when no rule applies.
        """
        if not capabilities.is_enabled("unfollow"):
            return None
        if not self._has_elapsed_cooldown(
            context.time_step,
            self.state.last_unfollow_step,
            self.unfollow_cooldown_steps,
        ):
            return None

        candidates: list[int] = []
        visible_timeline = context.obs.get("visible_tl", ()) or ()
        visible_ids: set[int] = set()
        for entry in visible_timeline:
            agent_id = int(entry["agent_id"])
            visible_ids.add(agent_id)
            message = str(entry.get("message", "")).strip()
            if message:
                self.state.empty_timeline_steps[agent_id] = 0
            else:
                self.state.empty_timeline_steps[agent_id] = (
                    self.state.empty_timeline_steps.get(agent_id, 0) + 1
                )
            is_inactive = (
                self.state.empty_timeline_steps[agent_id] >= self.empty_tweet_steps
            )
            is_negative = any(
                keyword in message.lower() for keyword in self.negative_keywords
            )
            if is_inactive or is_negative:
                candidates.append(agent_id)
        self.state.empty_timeline_steps = {
            agent_id: steps
            for agent_id, steps in self.state.empty_timeline_steps.items()
            if agent_id in visible_ids
        }
        if (
            not candidates
            and visible_ids
            and self.prng.random() < self.unfollow_probability
        ):
            candidates = sorted(visible_ids)
        if not candidates:
            return None
        target = self.prng.choice(sorted(set(candidates)))
        self.state.last_unfollow_step = context.time_step
        return target

    def get_follow_agent_id(
        self,
        context: DecisionContext,
        capabilities: ActionCapabilities,
        unfollow_agent_id: Optional[int] = None,
    ) -> Optional[int]:
        """Get one valid recommended agent satisfying follow rules.

        Args:
            context: Current observation containing recommendations and counts.
            capabilities: Enabled household actions.
            unfollow_agent_id: Same-step target whose removal may free capacity.

        Returns:
            Recommended target identifier, or ``None`` when no rule applies.
        """
        if not capabilities.is_enabled("follow"):
            return None
        if not self._has_elapsed_cooldown(
            context.time_step,
            self.state.last_follow_step,
            self.follow_cooldown_steps,
        ):
            return None
        follow_cap = context.obs.get("follow_cap")
        num_follows = int(context.obs.get("num_follows", 0))
        if follow_cap is not None:
            remaining_capacity = int(follow_cap) - num_follows
            if unfollow_agent_id is not None:
                remaining_capacity += 1
            if remaining_capacity <= 0:
                return None
        if self.prng.random() >= self.follow_probability:
            return None

        self_agent_id = context.obs.get("self_agent_id")
        followed_ids = {
            int(entry["agent_id"])
            for entry in (context.obs.get("visible_tl", ()) or ())
        }
        recommended_agent_ids = {
            self._get_recommended_agent_id(recommendation)
            for recommendation in (context.obs.get("recommended_follows", ()) or ())
        }
        candidates = sorted(
            agent_id
            for agent_id in recommended_agent_ids
            if agent_id is not None
            and agent_id != self_agent_id
            and agent_id not in followed_ids
            and agent_id != unfollow_agent_id
        )
        if not candidates:
            return None
        target = self.prng.choice(candidates)
        self.state.last_follow_step = context.time_step
        return target

    def generate_tweet_intent(self, context: DecisionContext) -> TweetIntent:
        """Generate rule-selected topic, sentiment, style, and memory excerpt.

        Args:
            context: Observation containing summarized memory.

        Returns:
            Fully specified model-independent tweet intent.
        """
        memory = self._get_memory(context)
        topic = self.get_tweet_topic(memory)
        sentiment = self.get_tweet_sentiment(topic, memory)
        style = self.get_tweet_style(topic, sentiment, memory)
        return TweetIntent(
            topic=topic,
            sentiment=sentiment,
            style=style,
            memory_excerpt=self.get_memory_excerpt(topic, memory),
        )

    def get_tweet_topic(self, memory: dict[str, Any]) -> TweetTopic:
        """Get a topic from changed memories, stress, and configured priority.

        Args:
            memory: Current summarized memory.

        Returns:
            Selected tweet topic.
        """
        scored_topics: dict[TweetTopic, float] = {}
        for key, topic in self._MEMORY_TOPIC.items():
            if key in self.state.changed_memory_keys:
                scored_topics[topic] = scored_topics.get(topic, 0.0) + 1.0
            stress = memory.get(f"{key}_stress")
            if isinstance(stress, (int, float)):
                scored_topics[topic] = (
                    scored_topics.get(topic, 0.0) + float(stress) / 100.0
                )
        if not scored_topics:
            return "daily_life"
        priority_index = {
            topic: len(self.topic_priority) - index
            for index, topic in enumerate(self.topic_priority)
        }
        return max(
            scored_topics,
            key=lambda topic: (scored_topics[topic], priority_index.get(topic, 0)),
        )

    def get_tweet_sentiment(
        self,
        topic: TweetTopic,
        memory: dict[str, Any],
    ) -> TweetSentiment:
        """Get sentiment from stress and event-specific rule signals.

        Args:
            topic: Already selected tweet topic.
            memory: Current summarized memory.

        Returns:
            Positive, neutral, or negative sentiment.
        """
        relevant_keys = self._TOPIC_MEMORY[topic]
        stresses = [
            float(memory[f"{key}_stress"])
            for key in relevant_keys
            if isinstance(memory.get(f"{key}_stress"), (int, float))
        ]
        if stresses and max(stresses) >= 60.0:
            return "negative"
        if topic in {"consumption", "shopping", "social"}:
            summaries = " ".join(str(memory.get(key, "")) for key in relevant_keys)
            no_history_markers = ("no consumption", "no purchase", "no social")
            if summaries and not any(
                marker in summaries.lower() for marker in no_history_markers
            ):
                return "positive"
        return "neutral"

    def get_tweet_style(
        self,
        topic: TweetTopic,
        sentiment: TweetSentiment,
        memory: dict[str, Any],
    ) -> TweetStyle:
        """Get writing style from selected topic and sentiment.

        Args:
            topic: Rule-selected topic.
            sentiment: Rule-selected sentiment.
            memory: Current summarized memory; reserved for custom subclasses.

        Returns:
            Rule-selected writing style.
        """
        del memory
        if sentiment == "negative":
            return "terse" if topic in {"finances", "health"} else "reflective"
        if sentiment == "positive":
            return "enthusiastic"
        if topic in {"shopping", "finances"}:
            return "informative"
        return "casual"

    def get_memory_excerpt(
        self,
        topic: TweetTopic,
        memory: dict[str, Any],
    ) -> str:
        """Get compact topic-relevant grounding text for the Tiny LM.

        Args:
            topic: Selected topic.
            memory: Current summarized memory.

        Returns:
            Newline-delimited summaries and stress reasons. If memory is not
            available, an explicit no-memory marker is returned.
        """
        excerpts: list[str] = []
        for key in self._TOPIC_MEMORY[topic]:
            summary = memory.get(key)
            if summary:
                excerpts.append(f"{key}: {self._get_compact_memory_text(str(summary))}")
            stress_reason = memory.get(f"{key}_stress_reason")
            if stress_reason:
                excerpts.append(
                    f"{key}_stress: {self._get_compact_memory_text(str(stress_reason))}"
                )
        if not excerpts:
            return "No relevant memory is available."
        excerpt = "\n".join(excerpts)
        return excerpt[: self.max_memory_excerpt_characters].rstrip()

    @staticmethod
    def _get_compact_memory_text(memory_text: str) -> str:
        """Get the latest compact component of a summarized memory string.

        Args:
            memory_text: Natural-language summary that may contain a long
                semicolon-delimited history.

        Returns:
            Most recent semicolon-delimited component. Short summaries are
            returned unchanged.

        Memory summaries are optimized for larger action-generating LLMs and
        may contain dozens of events. A Tiny LM should receive the latest event
        rather than a prompt truncated halfway through the history.
        """
        normalized_text = " ".join(memory_text.split())
        if "; " in normalized_text:
            latest_component = normalized_text.rsplit("; ", maxsplit=1)[-1]
        else:
            latest_component = normalized_text
        return re.sub(r"\s+at time .*$", ".", latest_component)

    def _update_hawkes_intensity(self, context: DecisionContext) -> None:
        """Decay excitation and add impulses for changed memory categories.

        Args:
            context: Current normalized observation.
        """
        if self.state.last_hawkes_step is None:
            elapsed_steps = 0
        else:
            elapsed_steps = max(0, context.time_step - self.state.last_hawkes_step)
        self.state.hawkes_excitation *= math.exp(-self.decay_rate * elapsed_steps)

        memory = self._get_memory(context)
        changed_keys: list[str] = []
        if self.state.last_memory_snapshot:
            changed_keys = [
                key
                for key in self._MEMORY_TOPIC
                if memory.get(key) != self.state.last_memory_snapshot.get(key)
            ]
            self.state.hawkes_excitation += self.memory_excitation * len(changed_keys)
            for key in changed_keys:
                stress = memory.get(f"{key}_stress")
                if isinstance(stress, (int, float)):
                    self.state.hawkes_excitation += self.stress_excitation_scale * max(
                        0.0, float(stress)
                    )
        self.state.changed_memory_keys = tuple(changed_keys)
        self.state.last_memory_snapshot = {
            key: value
            for key, value in memory.items()
            if key in self._MEMORY_TOPIC
            or key.endswith("_stress")
            or key.endswith("_stress_reason")
        }
        self.state.last_hawkes_step = context.time_step

    def _get_memory(self, context: DecisionContext) -> dict[str, Any]:
        """Get summarized memory from the current observation.

        Args:
            context: Current normalized observation.

        Returns:
            Memory mapping, or an empty mapping when unavailable.
        """
        memory = context.obs.get("memory")
        return memory if isinstance(memory, dict) else {}

    @staticmethod
    def _get_recommended_agent_id(recommendation: Any) -> Optional[int]:
        """Get an agent identifier from either supported recommendation shape.

        Args:
            recommendation: Integer identifier or a recommendation mapping with
                an ``agent_id`` field.

        Returns:
            Integer identifier, or ``None`` for a malformed recommendation.

        ``TwoHopRecommenderSystem`` currently returns mappings, while older
        providers and user extensions may return bare integers. Supporting both
        preserves compatibility with the documented observation contract.
        """
        if isinstance(recommendation, dict):
            recommendation = recommendation.get("agent_id")
        if isinstance(recommendation, int):
            return recommendation
        return None

    @staticmethod
    def _has_elapsed_cooldown(
        current_step: int,
        last_step: Optional[int],
        cooldown_steps: int,
    ) -> bool:
        """Return whether a follow-management cooldown has elapsed.

        Args:
            current_step: Current simulation step.
            last_step: Previous action step, if any.
            cooldown_steps: Required number of steps between actions.

        Returns:
            ``True`` when an action may be attempted.

        Follow-management cooldowns protect against graph churn. They are not
        used for tweets; tweet timing is governed exclusively by the Hawkes
        process.
        """
        return last_step is None or current_step - last_step >= cooldown_steps
