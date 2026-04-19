from __future__ import annotations

from .llm_agent import LLMAgent

from typing import Any


class AutoReactLLMAgent(LLMAgent):
    """An LLMAgent that automatically reacts to all incoming orders and proposals.

    This wrapper enforces deterministic acceptance of transactional intents
    (i.e., `incoming_orders` and `incoming_proposals`) posterior to, or independently of,
    the LLM-generated decision. The primary objective is to eliminate non-responsive
    or economically irrational behaviors (e.g., ignoring valid orders) that often
    arise from stochastic LLM outputs in supply-side agents such as retailers
    and restaurants.

    In socio-economic EconSimulacra, supply-side agents are
    expected to process transactions reliably. However, vanilla LLMAgents may omit
    reactions due to prompt misalignment or token truncation. This wrapper guarantees
    that all valid incoming transactional requests are accepted, thereby preserving
    simulation consistency and preventing deadlocks in the market mechanism.
    """

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Automatically react to all incoming orders and proposals.

        Get the LLM-generated response and then overwrite the `reactions`
        field to ensure that all incoming transactional intents are accepted.

        Args:
            obs: The observation dictionary containing the current state of the agent,
                including any incoming orders and proposals.

        Returns:
            dict: A dictionary representing the agent's action,
                with the `reactions` field
                modified to accept all incoming transactional intents.
        """
        llm_response: dict[str, Any] = await super().act(obs=obs)
        reactions: list[dict[str, Any]] = []
        _inventory_after_reaction: dict[str, float | int] = self.get_inventory()
        if "incoming_orders" in obs:
            for incoming_order in obs["incoming_orders"]:
                judge, _inventory_after_reaction = self.judge_reaction(
                    incoming_transactional_intent=incoming_order,
                    current_inventory=_inventory_after_reaction,
                    is_order=True,
                )
                if judge:
                    reactions.append(
                        {
                            "kind": "order",
                            "id": incoming_order["order_id"],
                            "accept_amount": incoming_order["item_amount"],
                        }
                    )
            for incoming_proposal in obs["incoming_proposals"]:
                judge, _inventory_after_reaction = self.judge_reaction(
                    incoming_transactional_intent=incoming_proposal,
                    current_inventory=_inventory_after_reaction,
                    is_order=False,
                )
                if judge:
                    reactions.append(
                        {
                            "kind": "proposal",
                            "id": incoming_proposal["proposal_id"],
                            "accept": True,
                        }
                    )
        llm_response["reactions"] = reactions
        return llm_response

    def judge_reaction(
        self,
        incoming_transactional_intent: dict[str, Any],
        current_inventory: dict[str, float | int],
        is_order: bool,
    ) -> tuple[bool, dict[str, float | int]]:
        """Judge whether to react to an incoming transactional intent based on the current inventory.

        Args:
            incoming_transactional_intent: A dictionary representing the incoming order
                or proposal that the agent is evaluating.
            current_inventory: A dictionary representing the agent's current inventory
                before reacting to the intent.
            is_order: A boolean indicating whether the incoming transactional intent
                is an order (True) or a proposal (False).

        Returns:
            bool: True if the agent should react to the incoming transactional intent,
                False otherwise.

        Note:
            Always True as long as the agent has sufficient inventory to fulfill the intent.
        """
        if is_order:
            item_name: str = incoming_transactional_intent["item_name"]
            item_amount: float | int = incoming_transactional_intent["item_amount"]
        else:
            item_name: str = incoming_transactional_intent["get_item_name"]
            item_amount: float | int = incoming_transactional_intent["get_item_amount"]
        judge: bool
        if current_inventory.get(item_name, 0) >= item_amount:
            current_inventory[item_name] = (
                current_inventory.get(item_name, 0) - item_amount
            )
            judge = True
        else:
            judge = False
        return judge, current_inventory