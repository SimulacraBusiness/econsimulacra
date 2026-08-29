import asyncio
import math
from random import Random
from typing import Any

import pytest

from econsimulacra.agents.households import (
    ActionCapabilities,
    DecisionContext,
    HouseholdState,
    RuleBasedHousehold,
    SocialMediaPolicy,
    TweetIntent,
    TweetRenderer,
)


def _context(step: int = 0, **obs: Any) -> DecisionContext:
    observation = {
        "time": step,
        "self_agent_id": 0,
        "self_is_sleeping": False,
        "self_pos": (0, 0),
        "self_init_pos": (0, 0),
        "self_inventory": {"Yen": 100.0, "Rice": 1.0},
        "follow_cap": 2,
        "num_follows": 0,
        "visible_tl": (),
        "recommended_follows": (),
        "memory": {},
        **obs,
    }
    return DecisionContext(
        obs=observation,
        time_step=step,
        hour=float(step % 24),
        current_pos=(0, 0),
        inventory={"Yen": 100.0, "Rice": 1.0},
    )


def _state() -> HouseholdState:
    return HouseholdState(
        sleep_pressure=0.3,
        hunger=0.2,
        last_meal_elapsed=1.0,
        home=(0, 0),
    )


class FixedTextGenerator:
    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    async def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.text


def test_hawkes_intensity_decays_and_tweet_adds_self_excitation() -> None:
    policy = SocialMediaPolicy(
        {
            "tweet": {
                "baseIntensity": 0.01,
                "selfExcitation": 0.4,
                "decayRate": 0.5,
            }
        },
        Random(42),
    )

    policy.generate_social_decision(_context(step=0), _state(), ActionCapabilities())
    intent = TweetIntent("daily_life", "neutral", "casual", "A memory")
    policy.record_generated_tweet(intent)
    assert policy.get_hawkes_intensity() == pytest.approx(0.41)

    policy.generate_social_decision(_context(step=2), _state(), ActionCapabilities())
    assert policy.get_hawkes_intensity() == pytest.approx(0.01 + 0.4 * math.exp(-1.0))


def test_changed_stressed_memory_selects_topic_sentiment_and_style() -> None:
    policy = SocialMediaPolicy(
        {"tweet": {"baseIntensity": 100.0}},
        Random(3),
    )
    capabilities = ActionCapabilities()
    policy.generate_social_decision(
        _context(step=0, memory={"move_history": "You stayed home."}),
        _state(),
        capabilities,
    )
    decision = policy.generate_social_decision(
        _context(
            step=1,
            memory={
                "move_history": "You moved far from home.",
                "move_history_stress": 80,
                "move_history_stress_reason": "Movement was tiring.",
            },
        ),
        _state(),
        capabilities,
    )

    assert decision.tweet_intent is not None
    assert decision.tweet_intent.topic == "mobility"
    assert decision.tweet_intent.sentiment == "negative"
    assert decision.tweet_intent.style == "reflective"
    assert "Movement was tiring" in decision.tweet_intent.memory_excerpt


def test_memory_excerpt_keeps_latest_event_within_tiny_model_budget() -> None:
    policy = SocialMediaPolicy(
        {"tweet": {"maxMemoryExcerptCharacters": 90}},
        Random(5),
    )
    memory = {
        "state_evaluation_history": (
            "Your state evaluations are Wealth: 100 at time 1; "
            "Wealth: 120 at time 2; Wealth: 130 at time 3."
        )
    }

    excerpt = policy.get_memory_excerpt("finances", memory)

    assert "Wealth: 130." in excerpt
    assert "Wealth: 100" not in excerpt
    assert "at time" not in excerpt
    assert len(excerpt) <= 90


def test_follow_can_replace_an_unfollowed_agent_at_capacity() -> None:
    policy = SocialMediaPolicy(
        {
            "follow": {"probability": 1.0, "cooldownSteps": 0},
            "unfollow": {"emptyTweetSteps": 1, "cooldownSteps": 0},
        },
        Random(7),
    )
    decision = policy.generate_social_decision(
        _context(
            follow_cap=1,
            num_follows=1,
            visible_tl=({"agent_id": 1, "agent_name": "One", "message": ""},),
            recommended_follows=(2,),
        ),
        _state(),
        ActionCapabilities(frozenset({"tweet"})),
    )

    assert decision.unfollow_agent_id == 1
    assert decision.follow_agent_id == 2


def test_disabled_and_sleeping_social_actions_are_not_generated() -> None:
    policy = SocialMediaPolicy(
        {
            "tweet": {"baseIntensity": 100.0},
            "follow": {"probability": 1.0},
        },
        Random(9),
    )
    decision = policy.generate_social_decision(
        _context(
            self_is_sleeping=True,
            recommended_follows=(2,),
        ),
        _state(),
        ActionCapabilities(),
    )
    assert decision.tweet_intent is None
    assert decision.follow_agent_id is None
    assert decision.unfollow_agent_id is None


def test_tweet_renderer_builds_prompt_and_sanitizes_output() -> None:
    generator = FixedTextGenerator('Tweet:  "Shopping went well today!"\n')
    renderer = TweetRenderer(generator, max_characters=12)
    intent = TweetIntent(
        topic="shopping",
        sentiment="positive",
        style="enthusiastic",
        memory_excerpt="Shopping went well today.",
    )

    tweet = asyncio.run(renderer.generate_tweet(intent))

    assert tweet == "Shopping…"
    assert "Topic: shopping" in generator.prompts[0]
    assert "Sentiment: positive" in generator.prompts[0]
    assert "Style: enthusiastic" in generator.prompts[0]
    assert "Language: English" in generator.prompts[0]
    assert "first-person" in generator.prompts[0]
    assert "Add no facts, causes" in generator.prompts[0]
    assert "you/your to I/my" in generator.prompts[0]
    assert "Shopping went well today." in generator.prompts[0]


def test_tweet_renderer_keeps_tiny_model_wording() -> None:
    generator = FixedTextGenerator(
        "I lost income because of unforeseen expenses and should save more."
    )
    renderer = TweetRenderer(generator, max_characters=140)
    intent = TweetIntent(
        topic="financial situation",
        sentiment="concerned",
        style="reflective",
        memory_excerpt="state_evaluation_history: Wealth: 505007.",
    )

    tweet = asyncio.run(renderer.generate_tweet(intent))

    assert tweet == (
        "I lost income because of unforeseen expenses and should save more."
    )


def test_tweet_renderer_allows_messages_longer_than_140_characters() -> None:
    generated_text = "I remember " + "a meaningful household experience " * 5
    generator = FixedTextGenerator(generated_text)
    renderer = TweetRenderer(generator, max_characters=280)
    intent = TweetIntent(
        topic="daily life",
        sentiment="neutral",
        style="reflective",
        memory_excerpt="A meaningful household experience.",
    )

    tweet = asyncio.run(renderer.generate_tweet(intent))

    assert tweet is not None
    assert len(tweet) > 140
    assert tweet == generated_text.strip()


def test_rule_based_household_composes_generated_social_actions() -> None:
    generator = FixedTextGenerator("My wealth is 100.")
    household = RuleBasedHousehold(
        agent_id=0,
        agent_name="Household0",
        env_service_dic={"tweetTextClient": generator},
        prng=Random(11),
        config={
            "foodItems": ("Rice",),
            "sleepRule": {"onsetThreshold": 2.0},
            "mealRule": {"mealThreshold": 2.0},
            "socialRule": {
                "enabled": True,
                "tweet": {"baseIntensity": 100.0},
                "follow": {"probability": 1.0, "cooldownSteps": 0},
            },
        },
    )
    obs = _context(
        step=0,
        recommended_follows=(2,),
        memory={"state_evaluation_history": "Wealth: 100."},
    ).obs

    action = asyncio.run(household.act(obs))

    assert action["tweet"] == "My wealth is 100."
    assert action["follow"] == 2
    assert household.social_media_policy is not None
    assert household.social_media_policy in household.supplemental_policies
    assert household.social_media_policy.state.last_tweet_intent is not None
