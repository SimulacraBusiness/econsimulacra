import pytest

from econsimulacra.envs import Order, SwapProposal


class TestOrder:
    def test___init__(self) -> None:
        order = Order(
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=10,
            price=5.0,
            order_id=100,
            ttl=3,
        )
        assert order.agent_id == 1
        assert order.counterparty_id == 2
        assert order.item_name == "rice"
        assert order.item_amount == 10
        assert order.price == 5.0
        assert order.order_id == 100
        assert order.expire_in == 3
        assert order.accepted_amount == 0

    def test_react_and_execute(self) -> None:
        order = Order(
            order_id=0,
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=10,
            price=5.0,
        )
        order.react(amount=4)
        assert order.accepted_amount == 4
        with pytest.raises(ValueError):
            order.react(amount=7)
        order.execute()
        assert order.item_amount == 6
        assert order.accepted_amount == 0

    def test_update_time(self) -> None:
        order = Order(
            order_id=1,
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=10,
            price=5.0,
            ttl=2,
        )
        order.update_time()
        assert order.expire_in == 1
        order.update_time()
        assert order.expire_in == 0
        order1 = Order(
            order_id=2,
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=10,
            price=5.0,
            ttl=2,
        )
        order2 = Order(
            order_id=3,
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=10,
            price=5.0,
            ttl=3,
        )
        pending_orders = [order1, order2]
        for order in pending_orders:
            order.update_time()
        assert order1.expire_in == 1
        assert order2.expire_in == 2
        o1, o2 = pending_orders
        assert o1.expire_in == 1
        assert o2.expire_in == 2

    def test_is_fulfilled(self) -> None:
        order = Order(
            order_id=4,
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=5,
            price=5.0,
            ttl=1,
        )
        assert not order.is_fulfilled()
        order.react(amount=5)
        order.execute()
        assert order.is_fulfilled()
        assert not order.is_expired()
        order = Order(
            order_id=5,
            agent_id=1,
            counterparty_id=2,
            item_name="rice",
            item_amount=5,
            price=5.0,
            ttl=1,
        )
        order.update_time()
        assert order.is_fulfilled()
        assert order.is_expired()


class TestSwapProposal:
    def test___init__(self) -> None:
        proposal = SwapProposal(
            proposer_agent_id=1,
            responder_agent_id=2,
            give_item_name="rice",
            give_item_amount=5,
            get_item_name="cash",
            get_item_amount=20,
            proposal_id=200,
            ttl=4,
        )
        assert proposal.proposer_agent_id == 1
        assert proposal.responder_agent_id == 2
        assert proposal.give_item_name == "rice"
        assert proposal.give_item_amount == 5
        assert proposal.get_item_name == "cash"
        assert proposal.get_item_amount == 20
        assert proposal.proposal_id == 200
        assert proposal.expire_in == 4
        assert proposal.accept is None

    def test_react(self) -> None:
        proposal = SwapProposal(
            proposer_agent_id=1,
            responder_agent_id=2,
            give_item_name="rice",
            give_item_amount=5,
            get_item_name="cash",
            get_item_amount=20,
            proposal_id=201,
        )
        proposal.react(accept=True)
        assert proposal.accept is True
        proposal.react(accept=False)
        assert proposal.accept is False

    def test_update_time(self) -> None:
        proposal = SwapProposal(
            proposer_agent_id=1,
            responder_agent_id=2,
            give_item_name="rice",
            give_item_amount=5,
            get_item_name="cash",
            get_item_amount=20,
            ttl=3,
            proposal_id=202,
        )
        proposal.update_time()
        assert proposal.expire_in == 2
        proposal.update_time()
        assert proposal.expire_in == 1

    def test_is_fulfilled(self) -> None:
        proposal = SwapProposal(
            proposer_agent_id=1,
            responder_agent_id=2,
            give_item_name="rice",
            give_item_amount=5,
            get_item_name="cash",
            get_item_amount=20,
            ttl=1,
            proposal_id=203,
        )
        assert not proposal.is_fulfilled()
        proposal.react(accept=True)
        assert proposal.is_fulfilled()
        assert not proposal.is_expired()
        proposal = SwapProposal(
            proposer_agent_id=1,
            responder_agent_id=2,
            give_item_name="rice",
            give_item_amount=5,
            get_item_name="cash",
            get_item_amount=20,
            ttl=1,
            proposal_id=204,
        )
        proposal.update_time()
        assert proposal.is_fulfilled()
        assert proposal.is_expired()