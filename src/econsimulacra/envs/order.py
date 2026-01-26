from typing import Optional


class Order:
    def __init__(
        self,
        agent_id: int,
        is_buy: bool,
        item_name: str,
        item_amount: float | int,
        price: Optional[float],
        submitted_coords: tuple[int, ...],
        order_id: Optional[int] = None,
        ttl: Optional[int] = None,
    ) -> None:
        self.agent_id: int = agent_id
        self.is_buy: bool = is_buy
        self.item_name: str = item_name
        self.item_amount: float | int = item_amount
        self.price: Optional[float] = price
        self.submitted_coords: tuple[int, ...] = submitted_coords
        self.order_id: Optional[int] = order_id
        self.ttl: Optional[int] = ttl


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
        self.ttl: Optional[int] = ttl
