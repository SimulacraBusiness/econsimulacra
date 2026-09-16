from __future__ import annotations

from math import isclose, isfinite
from random import Random
from typing import Any, Optional

from econsimulacra.agents import RuleBasedRetailer
from econsimulacra.date_utils import get_corresponding_value


class DiscountRetailer(RuleBasedRetailer):
    """Accept orders and set scheduled discounts or markups.

    The configuration intentionally follows the ``DiscountRestaurant`` example:
    each ``discounts`` entry has ``start``, ``end``, ``discountRate``, and
    ``items``. A negative discount rate represents a markup.
    """

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        env_service_dic: dict[str, Any],
        prng: Optional[Random] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize order handling and scheduled price changes.

        Args:
            agent_id: Environment-assigned unique agent identifier.
            agent_name: Base name configured for the retailer.
            env_service_dic: Environment services available to the agent.
            prng: Optional seeded pseudo-random number generator.
            config: Retailer configuration containing optional ``discounts``.

        Returns:
            None.

        Note:
            ``discountRate=-0.1`` sets a ten-percent markup through the same
            ``initial_price * (1 - discount_rate)`` convention as the example.
        """
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            env_service_dic=env_service_dic,
            prng=prng,
            config=config,
        )
        self.time_span2discount = self._parse_discounts(self.config)
        self.item_name2initial_price: dict[str, float] = {}

    def _parse_discounts(
        self, config: dict[str, Any]
    ) -> dict[tuple[int | str, int | str], tuple[float, tuple[str, ...]]]:
        """Parse scheduled discount entries.

        Args:
            config: Retailer configuration.

        Returns:
            Time-span mapping from periods to discount rate and item names.

        Note:
            Rates may be negative for price increases but cannot exceed one,
            because that would produce a negative target price.
        """
        result: dict[tuple[int | str, int | str], tuple[float, tuple[str, ...]]] = {}
        raw_discounts = config.get("discounts", ())
        if not isinstance(raw_discounts, (list, tuple)):
            raise TypeError("discounts must be a list or tuple.")
        for entry in raw_discounts:
            if not isinstance(entry, dict):
                raise TypeError("Each discounts entry must be a mapping.")
            for key in ("start", "end", "discountRate", "items"):
                if key not in entry:
                    raise ValueError(f"Discount entry requires {key!r}.")
            start = entry["start"]
            end = entry["end"]
            if not isinstance(start, (int, str)) or isinstance(start, bool):
                raise TypeError("Discount start must be an integer or ISO time.")
            if not isinstance(end, (int, str)) or isinstance(end, bool):
                raise TypeError("Discount end must be an integer or ISO time.")
            raw_rate = entry["discountRate"]
            if isinstance(raw_rate, bool) or not isinstance(raw_rate, (int, float)):
                raise TypeError("discountRate must be numeric.")
            rate = float(raw_rate)
            if not isfinite(rate) or rate > 1:
                raise ValueError("discountRate must be finite and no greater than 1.")
            raw_items = entry["items"]
            if not isinstance(raw_items, (list, tuple)) or not raw_items:
                raise ValueError("Discount items must be a non-empty list or tuple.")
            items = tuple(str(item) for item in raw_items)
            if any(not item for item in items):
                raise ValueError("Discount item names cannot be empty.")
            span = (start, end)
            if span in result:
                raise ValueError(f"Duplicate discount interval: {span}.")
            result[span] = (rate, items)
        return result

    def time_to_discount(
        self, current_time: int | str
    ) -> tuple[float, tuple[str, ...]]:
        """Return the price rule active at the supplied time.

        Args:
            current_time: Integer step or ISO-formatted display time.

        Returns:
            Active discount rate and affected item names, or an empty rule.

        Note:
            Interval selection delegates to the same utility used by the
            existing ``DiscountRestaurant`` example.
        """
        return get_corresponding_value(
            current_time=current_time,
            time_span2value=self.time_span2discount,
            default_value=(0.0, ()),
        )

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Accept feasible orders and request scheduled price changes.

        Args:
            obs: Observation containing time, inventory, orders, and item prices.

        Returns:
            Combined reaction and ``set_prices`` action mapping.

        Note:
            Target prices are always computed from a cached initial price, so
            repeated steps cannot compound a markup. Failed changes caused by a
            temporary stockout are retried while the schedule remains active.
        """
        action = await super().act(obs)
        if "time" not in obs:
            raise KeyError("DiscountRetailer requires the 'time' observation.")
        discount_rate, discounted_items = self.time_to_discount(obs["time"])
        if not discounted_items:
            return action
        current_prices = self._get_current_prices(obs)
        set_prices: list[dict[str, str | float]] = []
        for item_name in discounted_items:
            if item_name not in current_prices:
                raise ValueError(
                    f"Price for discount item {item_name!r} was not observed."
                )
            self.item_name2initial_price.setdefault(
                item_name, current_prices[item_name]
            )
            target_price = self.item_name2initial_price[item_name] * (
                1.0 - discount_rate
            )
            if not isclose(
                current_prices[item_name], target_price, rel_tol=1e-12, abs_tol=1e-12
            ):
                set_prices.append({"item_name": item_name, "price": target_price})
        if set_prices:
            action["set_prices"] = tuple(set_prices)
        return action

    @staticmethod
    def _get_current_prices(obs: dict[str, Any]) -> dict[str, float]:
        """Normalize the rich-information item-price observation.

        Args:
            obs: Retailer observation containing ``item_name2price`` records.

        Returns:
            Mapping from item names to current prices.

        Note:
            Rich information must be enabled for a configured DiscountRetailer.
        """
        raw_prices = obs.get("item_name2price")
        if not isinstance(raw_prices, (list, tuple)):
            raise KeyError(
                "DiscountRetailer requires the 'item_name2price' observation."
            )
        return {str(entry["item_name"]): float(entry["price"]) for entry in raw_prices}
