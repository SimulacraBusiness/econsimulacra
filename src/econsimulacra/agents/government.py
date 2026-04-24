from random import Random
from typing import Any

from econsimulacra.agents import Agent

from ..date_utils import get_corresponding_value


class Government(Agent[dict[str, Any]]):
    """Rule-based government agent.

    This class only posts tweets based on a predefined schedule
    and does not interact with other agents or the environment in any other way.
    The policy must be defined separately by the user as "events".
    """

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        env_service_dic: dict[str, Any],
        prng: Random | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the government agent.

        config should contain:
            - tweets: dict[tuple[int | str, int | str], str]: a dictionary mapping
                time steps to tweets that the government will post at those time steps.
                The keys can be either integers (representing time steps) or strings
                (representing datetime in ISO format).
        """
        super().__init__(agent_id, agent_name, env_service_dic, prng, config)
        self.time_span2tweet: dict[tuple[int | str, int | str], str]
        if config is not None and "tweets" in config:
            self.time_span2tweet = {
                (entry["start"], entry["end"]): entry["tweet"]
                for entry in config["tweets"]
            }
        else:
            self.time_span2tweet = {}

    def self_assign_name(self, config: dict[str, Any]) -> None:
        """Assign the agent's name from the config.

        To easily determine the name of the government agent,
        we require that the config contains a "name" field without adding agent_id.
        """
        if "name" not in config:
            raise ValueError("Government agent requires 'name' in config.")
        self.agent_name: str = config["name"]

    def time_to_tweet(self, current_time: str | int) -> str:
        """Return the scheduled tweet."""
        return get_corresponding_value(
            current_time, self.time_span2tweet, default_value=""
        )

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Post a tweet if there is one scheduled for the current time step."""
        if "time" not in obs:
            raise KeyError("Observation must contain 'time' key.")
        current_time: int | str = obs["time"]
        tweet: str = self.time_to_tweet(current_time)
        return {"tweet": tweet} if tweet else {}
