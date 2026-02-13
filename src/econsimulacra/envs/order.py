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
        self.expire_in: int = ttl if ttl is not None else 2
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

    def __repr__(self) -> str:
        return (
            f"Order(order_id={self.order_id}, agent_id={self.agent_id}, counterparty_id={self.counterparty_id}, "
            f"item_name='{self.item_name}', item_amount={self.item_amount}, accepted_amount={self.accepted_amount}, "
            f"price={self.price}, expire_in={self.expire_in})"
        )


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

    def __repr__(self) -> str:
        return (
            f"SwapProposal(proposal_id={self.proposal_id}, "
            f"proposer_agent_id={self.proposer_agent_id}, responder_agent_id={self.responder_agent_id}, "
            f"give_item_name='{self.give_item_name}', give_item_amount={self.give_item_amount}, "
            f"get_item_name='{self.get_item_name}', get_item_amount={self.get_item_amount}, "
            f"expire_in={self.expire_in}, accept={self.accept})"
        )
