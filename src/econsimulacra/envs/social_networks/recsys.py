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
    """Base class for recommender systems in the social network environment (abstract class).

    Recommender systems are responsible for generating personalized recommendations
    for agents in the social network, such as which other agents to follow.
    They can also implement event hooks to update their internal state based on
    changes in the social network, such as when agents follow or unfollow each other, or tweet.
    .get_recommendations(agent_id) is the only method that must be implemented by subclasses,
    while the event hooks are optional to implement based on the specific recommendation algorithm.
    """

    def __init__(self, config: dict[str, Any], prng: Random) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): Configuration dictionary for the recommender system. Expected keys include
                - "maxRecommendations" (int, optional): The maximum number of recommendations to generate for each agent.
                - "isRandomized" (bool, optional): Whether to add randomness to the recommendations.
                - "temperature" (float, optional): The temperature parameter controlling the level of randomness.
                    (only relevant if "isRandomized" is True).
            prng (Random): A pseudo-random number generator instance for any stochastic behavior in the recommender system.
        """
        self.config: dict[str, Any] = config
        self.max_recs: int = config.get("maxRecommendations", 1)
        self.is_randomized: bool = config.get("isRandomized", False)
        if self.is_randomized:
            self.temperature: float = config.get("temperature", 1.0)
        self.prng: Random = prng
        self._sn: Optional[SocialNetwork] = None

    def bind(self, social_network: SocialNetwork) -> None:
        """Inject the social network instance into the recommender system.

        Args:
            social_network (SocialNetwork): The social network instance to bind to the recommender system.

        Note:
            This method is called when the SocialNetwork is initialized,
            allowing the recommender system to access the social network's state
            and subscribe to its events.
            See also: econsimulacra.envs.social_networks.base.SocialNetwork.__init__
        """
        self._sn = social_network

    def _require_bound(self) -> SocialNetwork:
        """Helper method to ensure that the recommender system is bound to a social network before accessing it."""
        if self._sn is None:
            raise ValueError(
                "Recommender system is not bound to a social network. Call .bind(social_network) after SocialNetwork initialization."
            )
        return self._sn

    @abstractmethod
    def get_recommendations(self, agent_id: int) -> list[Any]:
        pass

    def hook_add_agent(self, agent_id: int, agent_name: str) -> None:
        """Event hook called when a new agent is added to the social network.

        Args:
            agent_id (int): The unique identifier of the agent that was added.
            agent_name (str): The name of the agent that was added.

        Note:
            See also:
                econsimulacra.envs.social_networks.base.SocialNetwork.add_agent
        """
        pass

    def hook_follow_agent(self, agent_id: int, target_agent_id: int) -> None:
        """Event hook called when an agent follows another agent in the social network.

        Args:
            agent_id (int): The unique identifier of the agent that initiated the follow action.
            target_agent_id (int): The unique identifier of the agent that is being followed.

        Note:
            See also:
                econsimulacra.envs.social_networks.base.SocialNetwork.follow_agent
        """
        pass

    def hook_unfollow_agent(self, agent_id: int, target_agent_id: int) -> None:
        """Event hook called when an agent unfollows another agent in the social network.

        Args:
            agent_id (int): The unique identifier of the agent that initiated the unfollow action.
            target_agent_id (int): The unique identifier of the agent that is being unfollowed.

        Note:
            See also:
                econsimulacra.envs.social_networks.base.SocialNetwork.unfollow_agent
        """
        pass

    def hook_tweet(self, agent_id: int, message: str) -> None:
        """Event hook called when an agent tweets in the social network.

        Args:
            agent_id (int): The unique identifier of the agent that tweeted.
            message (str): The content of the tweet.

        Note:
            See also:
                econsimulacra.envs.social_networks.base.SocialNetwork.tweet
        """
        pass


class TwoHopRecommenderSystem(RecommenderSystem):
    """A simple recommender system that recommends agents based on two-hop connections in the social network."""

    def __init__(self, config: dict[str, Any], prng: Random) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): Configuration dictionary for the recommender system. Expected keys include
                - "maxRecommendations" (int, optional): The maximum number of recommendations to generate for each agent.
                - "isRandomized" (bool, optional): Whether to add randomness to the recommendations.
                - "temperature" (float, optional): The temperature parameter controlling the level of randomness.
                    (only relevant if "isRandomized" is True).
            prng (Random): A pseudo-random number generator instance for any stochastic behavior in the recommender system.

        Note:
            self.agent_id2two_hop_follows[i][j] = number of i -> * -> j paths, where * is any intermediate agent.
            If agent i does not follow agent j, but there exists an intermediate agent * that agent i follows and that follows agent j,
            then agent i considers agent j as a two-hop connection and is more likely to recommend agent j to agent i.
        """
        super().__init__(config, prng)
        self.agent_id2follows: dict[int, set[int]] = defaultdict(set)
        self.agent_id2followers: dict[int, set[int]] = defaultdict(set)
        self.agent_id2num_followers: dict[int, int] = defaultdict(int)
        self.agent_id2two_hop_follows: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._dirty: set[int] = set()
        self._cache: dict[int, list[dict[str, int | str]]] = {}

    def hook_add_agent(self, agent_id: int, agent_name: str) -> None:
        """Event hook called when a new agent is added to the social network.

        Args:
            agent_id (int): The unique identifier of the agent that was added.
            agent_name (str): The name of the agent that was added.
        """
        self.agent_id2follows[agent_id] = set()
        self.agent_id2followers[agent_id] = set()
        self.agent_id2num_followers[agent_id] = 0
        self.agent_id2two_hop_follows[agent_id] = defaultdict(int)
        self._dirty.add(agent_id)

    def hook_follow_agent(self, agent_id: int, target_agent_id: int) -> None:
        """Event hook called when an agent follows another agent in the social network.

        Args:
            agent_id (int): The unique identifier of the agent that is following.
            target_agent_id (int): The unique identifier of the agent being followed.

        Note:
            When agent_id follows target_agent_id, we need to update the two-hop follow counts for
                agent_id -> target_agent_id -> * and
                * -> agent_id -> target_agent_id
            for all relevant intermediate agents *.
        """
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
        """Event hook called when an agent unfollows another agent in the social network.

        Args:
            agent_id (int): The unique identifier of the agent that is unfollowing.
            target_agent_id (int): The unique identifier of the agent being unfollowed.

        Note:
            When agent_id unfollows target_agent_id, we need to update the two-hop follow counts for
                agent_id -> target_agent_id -> * and
                * -> agent_id -> target_agent_id
            for all relevant intermediate agents *.
        """
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

    def get_recommendations(self, agent_id: int) -> list[dict[str, Any]]:
        """Generate recommendations for the given agent based on two-hop connections.

        Args:
            agent_id (int): The unique identifier of the agent for whom to generate recommendations.


        Note:
            The recommendation score for a candidate agent is based on:
                1) The number of two-hop paths from the given agent to the candidate agent (higher is better).
                2) The number of followers the candidate agent has (higher is better).
                3) A random tie-breaker (only if self.is_randomized is True, and controlled by self.temperature).
            Agents that are already followed by the given agent are excluded from the recommendations.
        """
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
        _sn: SocialNetwork = self._require_bound()
        recommendations: list[dict[str, int | str]] = [
            {
                "agent_id": candidate,
                "agent_name": _sn.agent_id2agent_name[candidate],
                "latest_tweet": _sn.agent_id2tweet[candidate],
            }
            for candidate in sorted_candidates[: self.max_recs]
        ]
        self._cache[agent_id] = recommendations
        self._dirty.discard(agent_id)
        return recommendations
