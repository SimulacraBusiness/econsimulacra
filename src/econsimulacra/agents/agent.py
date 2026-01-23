from abc import ABC, abstractmethod
from typing import Optional
from typing import Generic, TypeVar


ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class Agent(ABC, Generic[ObsT, ActT]):
    def __init__(
        self,
        agent_id: int,
        agent_name: str,
    ) -> None:
        self.agent_id: int = agent_id
        self.agent_name: str = agent_name
        self.inventory_dic: dict[str, float | int] = self._initialize_inventory()

    @abstractmethod
    def _initialize_inventory(self) -> dict[str, float | int]:
        pass

    @abstractmethod
    def act(self, obs: ObsT) -> ActT:
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

    def __repr__(self) -> str:
        return f"Agent(id={self.agent_id}, name={self.agent_name}, inventory={self.inventory_dic})"
