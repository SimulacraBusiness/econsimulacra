class Item:
    def __init__(self, name: str):
        self.name = name
        self.price: float = 0.0

    def __repr__(self):
        return f"Item(name={self.name}, price={self.price})"
