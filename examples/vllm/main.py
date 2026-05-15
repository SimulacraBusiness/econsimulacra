import asyncio
import json
import os
import pathlib
from pathlib import Path
from random import Random
from typing import Any, Optional

from econsimulacra.agents import Agent
from econsimulacra.date_utils import get_corresponding_value
from econsimulacra.envs import Environment
from econsimulacra.events import Event
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import SimulationSummarizer, Simulator

# export LOG_TXT_PATH="log_gpt-4o-mini.txt"


class SubsidyEvent(Event):
    def __init__(self, trigger, config) -> None:
        super().__init__(trigger=trigger, config=config)
        self.subsidy_amount: int = config["subsidyAmount"]

    def execute(self, env, log=None) -> None:
        cash_name = env.cash_name
        for agent_id in env.agent_ids:
            agent = env.agent_id2agent[agent_id]
            agent.inventory_dic[cash_name] += self.subsidy_amount


class DiscountRestaurant(Agent[dict[str, Any]]):
    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        env_service_dic: dict[str, Any],
        prng: Optional[Random] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            agent_name=agent_name,
            env_service_dic=env_service_dic,
            prng=prng,
            config=config,
        )
        self.time_span2tweet: dict[tuple[int | str, int | str], str]
        self.time_span2discount_info: dict[
            tuple[int | str, int | str], tuple[float, list[str]]
        ]
        if config is not None and "tweets" in config:
            self.time_span2tweet = {
                (entry["start"], entry["end"]): entry["tweet"]
                for entry in config["tweets"]
            }
        else:
            self.time_span2tweet = {}
        if config is not None and "discounts" in config:
            self.time_span2discount_info = {
                (entry["start"], entry["end"]): (entry["discountRate"], entry["items"])
                for entry in config["discounts"]
            }
        else:
            self.time_span2discount_info = {}
        self.item_name2initial_price: dict[str, float] = {}

    def time_to_tweet(self, current_time: str | int) -> str:
        """Return the scheduled tweet."""
        return get_corresponding_value(
            current_time, self.time_span2tweet, default_value=""
        )

    def time_to_discount(self, current_time: str | int) -> tuple[float, list[str]]:
        """Return the scheduled discount information."""
        return get_corresponding_value(
            current_time, self.time_span2discount_info, default_value=(0.0, [])
        )

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Post a tweet if there is one scheduled for the current time step."""
        if "time" not in obs:
            raise KeyError("Observation must contain 'time' key.")
        current_time: int | str = obs["time"]
        tweet: str = self.time_to_tweet(current_time)
        discount_rate, discounted_items = self.time_to_discount(current_time)
        set_prices: list[dict[str, str | float]] = []
        for item_name in discounted_items:
            if item_name not in self.item_name2initial_price:
                assert "item_name2price" in obs
                item_name2price: list[dict[str, float | str]] = obs["item_name2price"]
                initial_price: Optional[float] = None
                for d in item_name2price:
                    if d["item_name"] == item_name:
                        initial_price = float(d["price"])
                        break
                if initial_price is None:
                    raise ValueError(
                        f"Initial price for item '{item_name}' not found in observation."
                    )
                else:
                    self.item_name2initial_price[item_name] = initial_price
            initial_price = self.item_name2initial_price[item_name]
            discounted_price = initial_price * (1 - discount_rate)
            set_prices.append({"item_name": item_name, "price": discounted_price})
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
        return {
            "tweet": tweet,
            "set_prices": set_prices,
            "reactions": reactions,
        }

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
        item_name: str
        item_amount: float | int
        if is_order:
            item_name = incoming_transactional_intent["item_name"]
            item_amount = incoming_transactional_intent["item_amount"]
        else:
            item_name = incoming_transactional_intent["get_item_name"]
            item_amount = incoming_transactional_intent["get_item_amount"]
        judge: bool
        if current_inventory.get(item_name, 0) >= item_amount:
            current_inventory[item_name] = (
                current_inventory.get(item_name, 0) - item_amount
            )
            judge = True
        else:
            judge = False
        return judge, current_inventory


def conduct_simulation():
    config_dic_path: Path = Path(pathlib.Path(__file__).parent, "config.json")
    logger: DictLogger = DictLogger()
    simulator: Simulator = Simulator(
        config=config_dic_path,
        env_class=Environment,
        logger=logger,
        summarizer_class=SimulationSummarizer,
    )
    simulator.register_classes([SubsidyEvent, DiscountRestaurant])
    asyncio.run(simulator.simulate(seed=42))
    logs: list[dict] = logger.logs
    log_txt_path: Path = pathlib.Path(os.environ["LOG_TXT_PATH"])
    with open(log_txt_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    conduct_simulation()
