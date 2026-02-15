from typing import Any
from typing import Optional


class Item:
    def __init__(
        self,
        item_id: int,
        item_name: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.item_id: int = item_id
        self.item_name: str = item_name
        self.price: float
        self.config: dict[str, Any] = config if config is not None else {}
        if "initialPrice" in self.config:
            self.price = self.config["initialPrice"]
        else:
            self.price = 0.0
        self.price_set_by: Optional[int] = None

    def get_price(self) -> float:
        return self.price

    def set_price(self, price: float, set_by: Optional[int] = None) -> None:
        self.price = price
        self.price_set_by = set_by

    def __repr__(self):
        return f"Item(id={self.item_id}, name={self.item_name}, price={self.price}, price_set_by={self.price_set_by})"
