from abc import ABC, abstractmethod

class Agent(ABC):
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

    def exchange_goods(
        self,
        get_item_name: str,
        get_item_amount: float | int,
        give_item_name: str,
        give_item_amount: float | int,
    ) -> None:
        if get_item_name not in self.inventory_dic:
            raise ValueError(f"Agent {self.agent_name} does not have {get_item_name} in inventory.")
        if give_item_name not in self.inventory_dic:
            raise ValueError(f"Agent {self.agent_name} does not have {give_item_name} in inventory.")
        self.inventory_dic[get_item_name] += get_item_amount
        self.inventory_dic[give_item_name] -= give_item_amount

