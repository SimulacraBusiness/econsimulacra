from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

from ..agents import Agent

if TYPE_CHECKING:
    from ..memory import MemoryHandler
    from .base import Environment

# TODO: generate_logを追加
# env.get_observations内部でlogをloggerに追加
# get_others_inventotyにpriceを追加
# >> summarizer側で値段が変わったitemをsummarizeして伝える

class ObsProvider(ABC):
    """Observation provider class (abstract class).

    Each ObsProvider class must implement the get_obs method,
    which returns the observation for a given agent_id.

    ObsProvider is used to build the observation space of the environment,
    and to provide observations to agents at each step.
    You can register your custom ObsProviders to the environment by adding
    them to obs_providers through either _build_observation_registry or
    _build_observation4allowed_agents_registry,
    depending on who can observe the information provided by the ObsProvider.

    Example:
        >>> class MyCustomObsProvider(ObsProvider):
        >>>     def get_obs(self, agent_id: int) -> Any:
        >>>         # Return the custom observation for the given agent_id
        >>>         return ...
        >>>
        >>> class MyCustomEnv(Environment):
        >>>     def _build_observation_registry(self) -> dict[str, ObsProvider]:
        >>>         obs_providers = super()._build_observation_registry()
        >>>         obs_providers["my_custom_obs"] = MyCustomObsProvider(env=self)
        >>>         return obs_providers

    See also:
        econsimulacra.envs.base.Environment.get_observations(agent_id: int)
    """

    def __init__(self, env: Environment) -> None:
        """Initialization.

        Args:
            env (Environment): The environment instance to which the ObsProvider belongs.
        """
        self.env: Environment = env

    @abstractmethod
    def get_obs(self, agent_id: int) -> Any:
        """Get the observation for the given agent_id.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            Any: The observation for the given agent_id.
        """
        raise NotImplementedError


class ObsProviderFromCoLocatedAgents(ABC):
    """Observation provider class from co-located agents (abstract class).

    This class is for fetching the information provided by co-located agents in the same grid cell.
    Each ObsProviderFromCoLocatedAgents class must implement the get_obs method,
    which returns the observation for a given agent_id.
    """

    def __init__(self, env: Environment, co_located_agents: set[int]) -> None:
        """Initialization.

        Args:
            env (Environment): The environment instance to which the ObsProvider belongs.
            co_located_agents (set[int]): The set of agent IDs that are co-located
                with the agent for which to provide observations.

        Note:
            The ObsProviderFromCoLocatedAgents class is initialized at every step for each agent,
            with the set of co-located agents in the current step.
            See also:
                econsimulacra.envs.base.Environment.get_observations(agent_id: int)
        """
        self.env: Environment = env
        self.co_located_agents: set[int] = co_located_agents

    @abstractmethod
    def get_obs(self, agent_id: int) -> Any:
        """Get the observation for the given agent_id.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            Any: The observation for the given agent_id.
        """
        raise NotImplementedError


class TimeProvider(ObsProvider):
    """Time Provider class."""

    def get_obs(self, agent_id: int) -> int | str:
        """Get the current time in the environment.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            int | str: The current time in the environment.
        """
        return self.env.get_time()


class TimeDeltaProvider(ObsProvider):
    """Time Delta Provider class."""

    def get_obs(self, agent_id: int) -> int | str:
        """Get the time delta in the environment.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            int | str: The time delta in the environment.
        """
        return self.env.get_timedelta()


class SelfIDProvider(ObsProvider):
    """Self ID Provider class."""

    def get_obs(self, agent_id: int) -> int:
        """Get the agent's own ID.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            int: The agent's own ID.
        """
        return agent_id


class SelfNameProvider(ObsProvider):
    """Self Name Provider class."""

    def get_obs(self, agent_id: int) -> str:
        """Get the agent's own name.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            str: The agent's own name.
        """
        return self.env.agent_id2agent[agent_id].get_self_name()


class SelfIsHouseholdProvider(ObsProvider):
    """Self Is Household Provider class."""

    def get_obs(self, agent_id: int) -> bool:
        """Get whether the agent is a household.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            bool: Whether the agent is a household.
        """
        return agent_id in self.env.household_ids


class SelfPosProvider(ObsProvider):
    """Self Position Provider class."""

    def get_obs(self, agent_id: int) -> tuple[int, ...]:
        """Get the agent's own position.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            tuple[int, ...]: The position of the agent as a tuple of coordinates.
        """
        return self.env.grid_space.get_pos(agent_id)


class SelfInitPosProvider(ObsProvider):
    """Self Initial Position Provider class."""

    def get_obs(self, agent_id: int) -> tuple[int, ...]:
        """Get the agent's initial position.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            tuple[int, ...]: The initial position of the agent as a tuple of coordinates.
        """
        return self.env.agent_id2initial_coords[agent_id]


class SelfIsMovingProvider(ObsProvider):
    """Self Is Moving Provider class."""

    def get_obs(self, agent_id: int) -> bool:
        """Get the agent's moving status.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            bool: Whether the agent is moving. is_moving is set to True when the agent has intended
                to move toward a destination and has not yet arrived.
        """
        return self.env.agent_id2is_moving[agent_id]


class SelfDestinationProvider(ObsProvider):
    """Self Destination Provider class."""

    def get_obs(self, agent_id: int) -> Optional[tuple[int, ...]]:
        """Get the agent's destination.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            tuple[int, ...], optional: The destination of the agent as a tuple of coordinates,
                or None if no destination is set.

        Note:
            The destination is set when the agent.is_moving.
        """
        return self.env.agent_id2destination[agent_id]


class OthersPosProvider(ObsProvider):
    """Others' Position Provider class."""

    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
        """Get the positions of other agents in the environment.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the positions
                of other agents who are willing to share their positions in the environment.
                Each dictionary has the following keys:
                - "agent_id": The ID of the other agent.
                - "agent_name": The name of the other agent.
                - "pos": The position of the other agent as a tuple of coordinates.
        """
        others_pos_infos: list[dict[str, Any]] = []
        for other_agent_id in self.env.agent_ids:
            if other_agent_id == agent_id:
                continue
            other_agent: Agent = self.env.agent_id2agent[other_agent_id]
            if "self_pos" in other_agent.provide_info4all_agents():
                others_pos_infos.append(
                    {
                        "agent_id": other_agent_id,
                        "agent_name": other_agent.get_self_name(),
                        "pos": self.env.grid_space.get_pos(agent_id=other_agent_id),
                    }
                )
        return others_pos_infos


class SelfInventoryProvider(ObsProvider):
    """Self Inventory Provider class."""

    def get_obs(self, agent_id: int) -> dict[str, float | int]:
        """Get the agent's own inventory.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            dict[str, float | int]: The agent's own inventory as a dictionary
                mapping item names to their amounts.
        """
        inventory_dic: dict[str, float | int] = self.env.agent_id2agent[
            agent_id
        ].get_inventory()
        inventory_dic = {
            item_name: item_amount
            for item_name, item_amount in inventory_dic.items()
            if item_amount > 0
        }
        return inventory_dic


class SelfSalaryProvider(ObsProvider):
    """Salary Provider class."""

    def get_obs(self, agent_id: int) -> float | int:
        """Get the salary of the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            float | int: The salary of the agent.

        Note:
            The initial cash amount is regarded as the salary in this implementation.
            Seealso: econsimulacra.events.ConstantSalary
        """
        return self.env.agent_id2initial_inventory[agent_id].get(
            self.env.cash_name, 0.0
        )


class SelfTweetProvider(ObsProvider):
    """Self Tweet Provider class."""

    def get_obs(self, agent_id: int) -> Optional[str]:
        """Get the agent's latest tweet.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            str, optional: The agent's latest tweet, or None if the agent has not tweeted yet.
        """
        return self.env.social_network.get_tweet(agent_id=agent_id)


class FollowCapProvider(ObsProvider):
    """Follow Cap Provider class."""

    def get_obs(self, agent_id: int) -> Optional[int]:
        """Get the follow cap of the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            int, optional: The follow cap of the social network, or None if not set.
        """
        return self.env.social_network.follow_cap


class NumFollowersProvider(ObsProvider):
    """Number of Followers Provider class."""

    def get_obs(self, agent_id: int) -> int:
        """Get the number of followers of the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            int: The number of followers of the agent.
        """
        return self.env.social_network.get_num_followers(agent_id=agent_id)


class NumFollowsProvider(ObsProvider):
    """Number of Follows Provider class."""

    def get_obs(self, agent_id: int) -> str:
        """Get the number of follows of the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            str: The number of follows of the agent.
        """
        num_follows: int = self.env.social_network.get_num_follows(agent_id=agent_id)
        follow_cap: Optional[int] = self.env.social_network.follow_cap
        if follow_cap is not None:
            warning_message: str = ""
            if follow_cap == num_follows:
                warning_message = (
                    f"You have reached the follow cap of {follow_cap}. "
                    + "To newly follow others, you need to unfollow some of your current follows first."
                )
            return f"{num_follows}/{follow_cap} follows. {warning_message}"
        else:
            return str(num_follows)


class VisibleTLProvider(ObsProvider):
    """Visible Timeline Provider class."""

    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
        """Get the visible timeline of the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            list[dict[str, Any]]: The visible timeline of the agent as a list of dictionaries.
                Each dictionary has the following keys:
                - "agent_id": The ID of the agent who posted the tweet.
                - "agent_name": The name of the agent who posted the tweet.
                - "message": The content of the tweet.
        """
        follow_agent_ids: set[int] = self.env.social_network.get_follows(
            agent_id=agent_id
        )
        visible_tl: list[dict[str, Any]] = []
        for follow_agent_id in follow_agent_ids:
            tweet: str = self.env.social_network.get_tweet(agent_id=follow_agent_id)
            visible_tl.append(
                {
                    "agent_id": follow_agent_id,
                    "agent_name": self.env.agent_id2agent[
                        follow_agent_id
                    ].get_self_name(),
                    "message": tweet,
                }
            )
        return visible_tl


class RecommendedFollowsProvider(ObsProvider):
    """Recommended Follows Provider class."""

    def get_obs(self, agent_id: int) -> list[int]:
        """Get the recommended follows for the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            list[int]: A list of recommended agent IDs to follow.
        """
        recommended_follows: list[int] = (
            self.env.social_network.get_recommended_follows(agent_id=agent_id)
        )
        return recommended_follows


class IncomingOrdersProvider(ObsProvider):
    """Incoming Orders Provider class."""

    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
        """Get the incoming orders for the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            list[dict[str, Any]]: A list of incoming orders as dictionaries.
                Each dictionary has the following keys:
                - "order_id": The ID of the order.
                - "agent_id": The ID of the agent who placed the order.
                - "agent_name": The name of the agent who placed the order.
                - "item_name": The name of the item to be sold.
                - "item_amount": The amount of the item to be sold.
                - "description": A description of the order, including the potential earnings for accepting the order.
        """
        incoming_orders: list[dict[str, Any]] = []
        for order in self.env.pending_orders:
            if order.counterparty_id == agent_id:
                item_price: float | int = self.env.item_name2item[order.item_name].price
                incoming_orders.append(
                    {
                        "order_id": order.order_id,
                        "agent_id": order.agent_id,
                        "agent_name": self.env.agent_id2agent_name[order.agent_id],
                        "item_name": order.item_name,
                        "item_amount": order.item_amount,
                        "description": "You are asked to sell your "
                        + f"{order.item_amount} of {order.item_name}. "
                        + f"You would earn {item_price * order.item_amount:.1f} {self.env.cash_name} "
                        + "if you accept this order.",
                    }
                )
        return incoming_orders


class IncomingSwapProposalsProvider(ObsProvider):
    """Incoming Swap Proposals Provider class."""

    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
        """Get the incoming swap proposals for the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            list[dict[str, Any]]: A list of incoming swap proposals as dictionaries.
                Each dictionary has the following keys:
                - "proposal_id": The ID of the proposal.
                - "agent_id": The ID of the agent who proposed the swap.
                - "agent_name": The name of the agent who proposed the swap.
                - "give_item_name": The name of the item to be given.
                - "give_item_amount": The amount of the item to be given.
                - "get_item_name": The name of the item to be received.
                - "get_item_amount": The amount of the item to be received.
                - "description": A description of the swap proposal, including the potential benefits for accepting the proposal.
        """

        incoming_proposals: list[dict[str, Any]] = []
        for proposal in self.env.pending_swap_proposals:
            if proposal.responder_agent_id == agent_id:
                incoming_proposals.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "agent_id": proposal.proposer_agent_id,
                        "agent_name": self.env.agent_id2agent_name[
                            proposal.proposer_agent_id
                        ],
                        "give_item_name": proposal.give_item_name,
                        "give_item_amount": proposal.give_item_amount,
                        "get_item_name": proposal.get_item_name,
                        "get_item_amount": proposal.get_item_amount,
                        "description": "You are asked to give your "
                        + f"{proposal.get_item_amount} of {proposal.get_item_name} "
                        + f"in exchange for {proposal.give_item_amount} of {proposal.give_item_name}.",
                    }
                )
        return incoming_proposals


class ItemName2PriceProvider(ObsProvider):
    """Item Name to Price Provider class."""

    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
        """Get the mapping from item names to their prices.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            list[dict[str, Any]]: A list of item names and their prices as dictionaries.

        Note:
            Currently, the item_name2prices observation is categorized as the observation
                for allowed agents, for example; retailer or restaurant agents, since they
                need to know the prices of items the competitors are selling.
        """
        item_name2prices: list[dict[str, Any]] = []
        for item_name, item in self.env.item_name2item.items():
            if item_name == self.env.cash_name:
                continue
            item_name2prices.append(
                {
                    "item_name": item_name,
                    "price": item.price,
                    "price_set_by": item.price_set_by,
                }
            )
        return item_name2prices


class OthersInventoriesProvider(ObsProviderFromCoLocatedAgents):
    """Others' Inventories Provider class from co-located agents."""

    def get_obs(self, agent_id: int, mask_amount: bool = False) -> list[dict[str, Any]]:
        """Get the inventories of other co-located agents in the same grid cell.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.
            mask_amount (bool): Whether to mask the amount of items in the inventory.
                If True, the amount of items will be replaced with "Unknown".

        Returns:
            list[dict[str, Any]]: A list of inventories of other co-located agents as dictionaries.
                Each dict contains ``"agent_id"`` (int), ``"agent_name"`` (str), and one key per
                non-cash item in the agent's inventory whose value is
                ``{"price": float, "amount": float | str}``.

        Note:
            The inventories of co-located agents are provided only when the other agents are willing to share
            their inventory information with co-located agents,
            i.e. when ``"inventory"`` is included in the agent's ``provideInfo4CoLocatedAgents`` config::

                "Restaurant": {
                    ...
                    "provideInfo4CoLocatedAgents": ["inventory"],
                }
        """
        inventory_infos: list[dict[str, Any]] = []
        for other_agent_id in self.env.agent_ids:
            if (
                other_agent_id == agent_id
                or other_agent_id not in self.co_located_agents
            ):
                continue
            other_agent: Agent = self.env.agent_id2agent[other_agent_id]
            if "inventory" in other_agent.provide_info4co_located_agents():
                inventory_info_dic: dict[
                    str, str | int | dict[str, str | int | float]
                ] = {
                    "agent_id": other_agent_id,
                    "agent_name": other_agent.get_self_name(),
                }
                other_agent_inventory_dic: dict[str, float | int] = (
                    other_agent.get_inventory()
                )
                for item_name, item_amount in other_agent_inventory_dic.items():
                    if item_name == self.env.cash_name:
                        continue
                    if item_amount <= 0:
                        continue
                    price: float = self.env.item_name2item[item_name].price
                    amount: str | float | int = (
                        "Unknown" if mask_amount else item_amount
                    )
                    inventory_info_dic[item_name] = {"price": price, "amount": amount}
                inventory_infos.append(inventory_info_dic)
        return inventory_infos


class MemoryProvider(ObsProvider):
    """Memory Provider class."""

    def get_obs(self, agent_id: int) -> Optional[dict[str, Any]]:
        """Get the memory of the agent.

        Args:
            agent_id (int): The ID of the agent for which to get the observation.

        Returns:
            dict[str, Any], optional: The memory of the agent as a dictionary,
                or None if the environment does not have a memory handler.

        Note:
            See also:
                econsimulacra.envs.memory.MemoryHandler
        """
        memory_handler: Optional[MemoryHandler] = self.env.get_memory_handler()
        if memory_handler is not None:
            return memory_handler.get_memory(agent_id=agent_id)
        else:
            return None
