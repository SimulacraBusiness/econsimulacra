from __future__ import annotations
from abc import ABC, abstractmethod
from ..agents import Agent
from typing import Any
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Environment
    from .memory import MemoryHandler


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
    
    
    """
    def __init__(self, env: Environment, co_located_agents: set[int]) -> None:
        self.env: Environment = env
        self.co_located_agents: set[int] = co_located_agents

    @abstractmethod
    def get_obs(self, agent_id: int) -> Any:
        raise NotImplementedError


class TimeProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> int | str:
        return self.env.get_time()


class TimeDeltaProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> int | str:
        return self.env.get_timedelta()


class SelfIDProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> int:
        return agent_id


class SelfNameProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> str:
        return self.env.agent_id2agent[agent_id].get_self_name()


class SelfPosProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> tuple[int, ...]:
        return self.env.grid_space.get_pos(agent_id)


class SelfInitPosProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> tuple[int, ...]:
        return self.env.agent_id2initial_coords[agent_id]


class SelfIsMovingProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> bool:
        return self.env.agent_id2is_moving[agent_id]


class SelfDestinationProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> Optional[tuple[int, ...]]:
        return self.env.agent_id2destination[agent_id]


class OthersPosProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
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
    def get_obs(self, agent_id: int) -> dict[str, float | int]:
        return self.env.agent_id2agent[agent_id].inventory_dic.copy()


class SelfTweetProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> Optional[str]:
        return self.env.social_network.get_tweet(agent_id=agent_id)


class FollowCapProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> Optional[int]:
        return self.env.social_network.follow_cap


class NumFollowersProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> int:
        return self.env.social_network.get_num_followers(agent_id=agent_id)


class NumFollowsProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> int:
        return self.env.social_network.get_num_follows(agent_id=agent_id)


class VisibleTLProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
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
    def get_obs(self, agent_id: int) -> list[int]:
        recommended_follows: list[int] = (
            self.env.social_network.get_recommended_follows(agent_id=agent_id)
        )
        return recommended_follows


class IncomingOrdersProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
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
    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
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
    def get_obs(self, agent_id: int) -> list[dict[str, Any]]:
        item_name2prices: list[dict[str, Any]] = []
        for item_name, item in self.env.item_name2item.items():
            item_name2prices.append(
                {
                    "item_name": item_name,
                    "price": item.price,
                    "price_set_by": item.price_set_by,
                }
            )
        return item_name2prices


class OthersInventoriesProvider(ObsProviderFromCoLocatedAgents):
    def get_obs(self, agent_id: int, mask_amount: bool = False) -> list[dict[str, Any]]:
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
                for item_name, item_amount in other_agent.inventory_dic.items():
                    if item_name == self.env.cash_name:
                        continue
                    price: float = self.env.item_name2item[item_name].price
                    amount: str | float | int = (
                        "Unknown" if mask_amount else item_amount
                    )
                    inventory_info_dic[item_name] = {"price": price, "amount": amount}
                inventory_infos.append(inventory_info_dic)
        return inventory_infos


class MemoryProvider(ObsProvider):
    def get_obs(self, agent_id: int) -> Optional[dict[str, Any]]:
        memory_handler: Optional[MemoryHandler] = self.env.get_memory_handler()
        if memory_handler is not None:
            return memory_handler.get_memory(agent_id=agent_id)
        else:
            return None
