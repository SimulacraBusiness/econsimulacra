class Item:
    def __init__(self, item_id: int, item_name: str):
        self.item_id: int = item_id
        self.item_name: str = item_name
        self.price: float = 0.0

    def set_price(self, price: float) -> None:
        """Set the price. Often used by retailers to set the price of items they sell."""
        self.price = price

    def __repr__(self):
        return f"Item(id={self.item_id}, name={self.item_name}, price={self.price})"
