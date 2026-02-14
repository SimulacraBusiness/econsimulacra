from abc import ABC, abstractmethod
import random
from random import Random
from typing import Any
from typing import Optional
from typing import Generic, TypeVar


ObsT = TypeVar("ObsT")


class Agent(ABC, Generic[ObsT]):
    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        is_rich_info_allowed: bool = False,
        prng: Optional[Random] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.agent_id: int = agent_id
        self.agent_type: str = self.__class__.__name__
        self.agent_name: str = agent_name
        self.is_rich_info_allowed: bool = is_rich_info_allowed
        self.prng: Random = prng if prng is not None else random.Random()
        self.inventory_dic: dict[str, float | int] = self._initialize_inventory(config)
        self.config: dict[str, Any] = config if config is not None else {}
        self.self_assign_name(self.config)

    def get_self_name(self) -> str:
        return self.agent_name

    def self_assign_name(self, config: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _initialize_inventory(self, config: Optional[dict[str, Any]]) -> dict[str, float | int]:
        pass

    @abstractmethod
    async def act(self, obs: ObsT) -> dict[str, Any]:
        pass

    def exchange_goods(
        self,
        get_item_name: Optional[str] = None,
        get_item_amount: Optional[float | int] = None,
        give_item_name: Optional[str] = None,
        give_item_amount: Optional[float | int] = None,
    ) -> None:
        if get_item_name is not None:
            if get_item_amount is None:
                raise ValueError(
                    "get_item_amount must be provided when get_item_name is provided."
                )
            if get_item_name not in self.inventory_dic:
                raise ValueError(
                    f"Agent {self.agent_name} does not have {get_item_name} in inventory."
                )
            self.inventory_dic[get_item_name] += get_item_amount
        if give_item_name is not None:
            if give_item_amount is None:
                raise ValueError(
                    "give_item_amount must be provided when give_item_name is provided."
                )
            if give_item_name not in self.inventory_dic:
                raise ValueError(
                    f"Agent {self.agent_name} does not have {give_item_name} in inventory."
                )
            self.inventory_dic[give_item_name] -= give_item_amount

    def provide_public_info(self) -> Optional[dict[str, Any]]:
        return None

    def provide_info4co_located_agents(self) -> Optional[dict[str, Any]]:
        return None

    def provide_info4allowed_agents(self) -> Optional[dict[str, Any]]:
        return None

    def request_obs(self) -> list[str]:
        return ["all"]

    def request_global_obs4allowed_agents(self) -> list[str]:
        # called if self.is_rich_info_allowed is True
        return ["all"]

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.agent_name}, inventory={self.inventory_dic})"
