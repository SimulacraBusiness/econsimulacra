from typing import Optional


class Order:
    def __init__(
        self,
        agent_id: int,
        counterparty_id: int,
        item_name: str,
        item_amount: float | int,
        price: Optional[float],
        order_id: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        self.agent_id: int = agent_id
        self.counterparty_id: int = counterparty_id
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: Optional[float] = price
        self.order_id: Optional[int] = order_id
        self.expire_in: int = ttl if ttl is not None else 1
        self.accepted_amount: float | int = 0

    def react(self, amount: float | int) -> None:
        if amount > (self.item_amount - self.accepted_amount):
            raise ValueError(
                f"Cannot execute order for amount {amount} greater than remaining item amount {self.item_amount - self.accepted_amount}."
            )
        self.accepted_amount += amount

    def execute(self) -> None:
        self.item_amount -= self.accepted_amount
        self.accepted_amount = 0

    def update_time(self) -> None:
        self.expire_in -= 1

    def is_fulfilled(self) -> bool:
        return self.item_amount <= 0 or self.expire_in <= 0


class SwapProposal:
    def __init__(
        self,
        proposer_agent_id: int,
        responder_agent_id: int,
        give_item_name: str,
        give_item_amount: float | int,
        get_item_name: str,
        get_item_amount: float | int,
        proposal_id: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        self.proposer_agent_id: int = proposer_agent_id
        self.responder_agent_id: int = responder_agent_id
        self.give_item_name: str = give_item_name
        self.give_item_amount: float | int = give_item_amount
        self.get_item_name: str = get_item_name
        self.get_item_amount: float | int = get_item_amount
        self.proposal_id: Optional[int] = proposal_id
        self.expire_in: int = ttl if ttl is not None else 1
        self.accept: Optional[bool] = None

    def react(self, accept: bool) -> None:
        self.accept = accept

    def update_time(self) -> None:
        self.expire_in -= 1

    def is_fulfilled(self) -> bool:
        return self.accept or self.expire_in <= 0