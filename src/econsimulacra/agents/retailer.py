from __future__ import annotations

from typing import Any

from .base import Agent


class RuleBasedRetailer(Agent[dict[str, Any]]):
    r"""Accept incoming orders up to currently available inventory.

    Args:
        agent_id: Environment-assigned unique agent identifier.
        agent_name: Base agent name; EconSimulacra may suffix the identifier.
        env_service_dic: Environment services made available to this agent.
        prng: Optional agent-specific pseudo-random generator.
        config: Retailer configuration, including inventory and optionally
            ``disabledActions``.

    Orders are processed in observation order. For request :math:`q_r` and
    remaining item availability :math:`A_{r-1}(i_r)`,

    .. math::

       y_r=\min\{q_r,A_{r-1}(i_r)\},\qquad
       A_r(i_r)=A_{r-1}(i_r)-y_r.
    """

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Accept feasible incoming orders.

        Args:
            obs: Observation containing ``self_inventory`` and optionally
                ``incoming_orders``.

        Returns:
            Reaction records for positive accepted quantities, or an empty
            mapping when reactions are disabled or nothing can be accepted.
        """
        available_inventory: dict[str, float] = self._available_inventory(obs)
        reactions = []
        for order in obs.get("incoming_orders", ()):
            reaction = self._accept_order(order, available_inventory)
            if reaction is not None:
                reactions.append(reaction)
        return {"reactions": tuple(reactions)} if reactions else {}

    @staticmethod
    def _available_inventory(obs: dict[str, Any]) -> dict[str, float]:
        """Create a mutable numeric availability ledger.

        Args:
            obs: Observation containing ``self_inventory``.

        Returns:
            Numeric copy of current inventory.
        """
        return {
            key: float(value) for key, value in obs.get("self_inventory", {}).items()
        }

    @staticmethod
    def _accept_order(
        order: dict[str, Any],
        available_inventory: dict[str, float],
    ) -> dict[str, Any] | None:
        """Accept one order against and update the availability ledger.

        Args:
            order: Incoming order with ID, item name, and requested amount.
            available_inventory: Mutable remaining inventory keyed by item name.

        Returns:
            One positive order reaction, or ``None`` when nothing is available.
        """
        item = str(order["item_name"])
        amount = min(float(order["item_amount"]), available_inventory.get(item, 0.0))
        if amount <= 0:
            return None
        available_inventory[item] -= amount
        return {
            "kind": "order",
            "id": order["order_id"],
            "accept_amount": amount,
        }
