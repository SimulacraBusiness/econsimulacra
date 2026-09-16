import asyncio

from examples.exp_behavioral_coherence.intervention_agents import DiscountRetailer


def _retailer() -> DiscountRetailer:
    """Build a deterministic retailer with one markup schedule.

    Returns:
        Configured DiscountRetailer.

    Note:
        No environment services are needed by RuleBasedRetailer.
    """
    return DiscountRetailer(
        agent_id=1,
        agent_name="Retailer",
        env_service_dic={},
        config={
            "name": "Daily Mart",
            "inventory": {"Rice": 20},
            "discounts": [
                {
                    "start": "2025-03-07 07:00:00",
                    "end": "2025-03-31 07:00:00",
                    "discountRate": -0.1,
                    "items": ["Rice"],
                }
            ],
        },
    )


def _obs(time: str, price: float) -> dict:
    """Build the observation required by DiscountRetailer.

    Args:
        time: ISO-formatted simulation time.
        price: Current observed Rice price.

    Returns:
        Retailer observation mapping.

    Note:
        One order is included to verify normal retail behavior is retained.
    """
    return {
        "time": time,
        "self_inventory": {"Rice": 20},
        "incoming_orders": [{"order_id": 4, "item_name": "Rice", "item_amount": 2}],
        "item_name2price": [{"item_name": "Rice", "price": price}],
    }


def test_discount_retailer_combines_orders_and_noncompounding_markup() -> None:
    """Verify order acceptance and a stable ten-percent markup coexist."""
    retailer = _retailer()
    before = asyncio.run(retailer.act(_obs("2025-03-07 06:00:00", 500)))
    assert "set_prices" not in before
    assert before["reactions"][0]["accept_amount"] == 2

    active = asyncio.run(retailer.act(_obs("2025-03-07 07:00:00", 500)))
    assert active["set_prices"] == (({"item_name": "Rice", "price": 550.0}),)

    unchanged = asyncio.run(retailer.act(_obs("2025-03-07 08:00:00", 550)))
    assert "set_prices" not in unchanged
