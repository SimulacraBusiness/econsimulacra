from abc import ABC, abstractmethod
from ..sim_utils import JsonRandom
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
        prng: Optional[Random] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.agent_id: int = agent_id
        self.agent_type: str = self.__class__.__name__
        self.agent_name: str = agent_name
        self.prng: Random = prng if prng is not None else random.Random()
        self.config: dict[str, Any] = config if config is not None else {}
        self.inventory_dic: dict[str, float | int] = self._initialize_inventory(
            self.config
        )
        self.is_rich_info_allowed: bool
        if self.config is not None and "isRichInfoAllowed" in self.config:
            self.is_rich_info_allowed = self.config["isRichInfoAllowed"]
        else:
            self.is_rich_info_allowed = False
        self.self_assign_name(self.config)

    def get_self_name(self) -> str:
        return self.agent_name

    def self_assign_name(self, config: dict[str, Any]) -> None:
        pass

    def _initialize_inventory(self, config: dict[str, Any]) -> dict[str, float | int]:
        json_random = JsonRandom(prng=self.prng)
        inventory_config: dict[str, Any] = config.get("inventory", {})
        inventory_dic: dict[str, Any] = {}
        for item_name, json_value in inventory_config.items():
            amount = json_random.random(json_value=json_value)
            inventory_dic[item_name] = amount
        return inventory_dic

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

    def provide_info4all_agents(self) -> list[str]:
        return []  # self_pos

    def provide_info4co_located_agents(self) -> list[str]:
        return []  # inventory

    def provide_info4allowed_agents(self) -> list[str]:
        return []  # item_name2price

    def request_obs(self) -> list[str]:
        return ["all"]

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.agent_name}, inventory={self.inventory_dic})"
