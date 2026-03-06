from __future__ import annotations
from abc import ABC, abstractmethod
from collections import defaultdict
import heapq
from random import Random
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import SocialNetwork


class RecommenderSystem(ABC):
    def __init__(self, config: dict[str, Any], prng: Random) -> None:
        self.config: dict[str, Any] = config
        self.max_recs: int = config.get("maxRecommendations", 1)
        self.is_randomized: bool = config.get("isRandomized", False)
        if self.is_randomized:
            self.temperature: float = config.get("temperature", 1.0)
        self.prng: Random = prng
        self._sn: Optional[SocialNetwork] = None

    def bind(self, social_network: SocialNetwork) -> None:
        """Inject the social network instance into the recommender system."""
        self._sn = social_network

    def _require_bound(self) -> SocialNetwork:
        if self._sn is None:
            raise ValueError(
                "Recommender system is not bound to a social network. Call .bind(social_network) after SocialNetwork initialization."
            )
        return self._sn

    @abstractmethod
    def get_recommendations(self, agent_id: int) -> list[int]:
        pass

    def hook_add_agent(self, agent_id: int) -> None:
        """Event hook called when a new agent is added to the social network."""
        pass

    def hook_follow_agent(self, agent_id: int, target_agent_id: int) -> None:
        """Event hook called when an agent follows another agent in the social network."""
        pass

    def hook_unfollow_agent(self, agent_id: int, target_agent_id: int) -> None:
        """Event hook called when an agent unfollows another agent in the social network."""
        pass

    def hook_tweet(self, agent_id: int, message: str) -> None:
        """Event hook called when an agent tweets in the social network."""
        pass


class TwoHopRecommenderSystem(RecommenderSystem):
    def __init__(self, config: dict[str, Any], prng: Random) -> None:
        super().__init__(config, prng)
        self.agent_id2follows: dict[int, set[int]] = defaultdict(set)
        self.agent_id2followers: dict[int, set[int]] = defaultdict(set)
        self.agent_id2num_followers: dict[int, int] = defaultdict(int)
        self.agent_id2two_hop_follows: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # agent_id2two_hop_follows[i][j] = number of i -> * -> j paths
        self._dirty: set[int] = set()
        self._cache: dict[int, list[int]] = {}

    def hook_add_agent(self, agent_id: int) -> None:
        self.agent_id2follows[agent_id] = set()
        self.agent_id2followers[agent_id] = set()
        self.agent_id2num_followers[agent_id] = 0
        self.agent_id2two_hop_follows[agent_id] = defaultdict(int)
        self._dirty.add(agent_id)

    def hook_follow_agent(self, agent_id: int, target_agent_id: int) -> None:
        self.agent_id2follows[agent_id].add(target_agent_id)
        self.agent_id2followers[target_agent_id].add(agent_id)
        self.agent_id2num_followers[target_agent_id] += 1
        for two_hop_target in self.agent_id2follows[target_agent_id]:
            if two_hop_target == agent_id:
                continue
            self.agent_id2two_hop_follows[agent_id][two_hop_target] += 1
        self._dirty.add(agent_id)
        for two_hop_source in self.agent_id2followers[agent_id]:
            if two_hop_source == target_agent_id:
                continue
            self.agent_id2two_hop_follows[two_hop_source][target_agent_id] += 1
            self._dirty.add(two_hop_source)

    def hook_unfollow_agent(self, agent_id: int, target_agent_id: int) -> None:
        self.agent_id2follows[agent_id].remove(target_agent_id)
        self.agent_id2followers[target_agent_id].remove(agent_id)
        self.agent_id2num_followers[target_agent_id] -= 1
        count: int
        for two_hop_target in self.agent_id2follows[target_agent_id]:
            if two_hop_target == agent_id:
                continue
            count = self.agent_id2two_hop_follows[agent_id][two_hop_target]
            if count <= 1:
                self.agent_id2two_hop_follows[agent_id].pop(two_hop_target, None)
            else:
                self.agent_id2two_hop_follows[agent_id][two_hop_target] = count - 1
        self._dirty.add(agent_id)
        for two_hop_source in self.agent_id2followers[agent_id]:
            if two_hop_source == target_agent_id:
                continue
            count = self.agent_id2two_hop_follows[two_hop_source][target_agent_id]
            if count <= 1:
                self.agent_id2two_hop_follows[two_hop_source].pop(target_agent_id, None)
            else:
                self.agent_id2two_hop_follows[two_hop_source][target_agent_id] = (
                    count - 1
                )
            self._dirty.add(two_hop_source)

    def get_recommendations(self, agent_id: int) -> list[int]:
        if agent_id in self._cache and agent_id not in self._dirty:
            return self._cache[agent_id]
        already_follows: set[int] = self.agent_id2follows[agent_id]
        already_follows.add(agent_id)
        candidates: set[int] = (
            set(
                self.agent_id2two_hop_follows[agent_id].keys()
                | set(self.agent_id2num_followers.keys())
            )
            - already_follows
        )
        sorted_candidates: list[int] = heapq.nlargest(
            self.max_recs,
            candidates,
            key=lambda candidate: (
                self.agent_id2two_hop_follows[agent_id].get(candidate, 0),
                self.agent_id2num_followers.get(candidate, 0)
                + self.prng.random()
                * (self.temperature if self.is_randomized else 0.0),
            ),
        )
        recommendations: list[int] = sorted_candidates[: self.max_recs]
        self._cache[agent_id] = recommendations
        self._dirty.discard(agent_id)
        return recommendations
