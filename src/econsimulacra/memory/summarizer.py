from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Deque, Optional, Type

from .memory_items import (
    AgentMemory,
    ConsumptionHistoryItem,
    ExchangeHistoryItem,
    InnerThoughtHistoryItem,
    MoveHistoryItem,
    ObsHistoryItem,
    PurchaseHistoryItem,
    SaleHistoryItem,
    SetPriceHistoryItem,
    SocialHistoryItem,
    StateEvaluationHistoryItem,
)
from .obs_summarization_utils import (
    summarize_num_changes,
    summarize_observed_price_changes,
    summarize_self_tweet_frequency,
)

ObsSummarizer = Callable[[list[ObsHistoryItem]], str]


class MemorySummarizer:
    """Memory Summarizer class.

    MemorySummarizer is used to summarize the memory of the agent into
    a form that can be provided as a part of the observation to the agent.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: Optional[list[Type]] = None,
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): the configuration for the MemorySummarizer. It must contain:
                "type": The type of the summarizer.
                and may contain:
                "relativeThresholdForPriceChange": the relative threshold for significant price changes in the observed inventory. Default is 0.01 (1%).
            prng (random.Random, optional): the pseudo-random number generator to use.
                If None, a new random.Random instance will be created.
            registred_classes (list[Type]): the list of registered classes.

        """
        self.config: dict[str, Any] = config
        self.relative_threshold_for_price_change: float = self.config.get(
            "relativeThresholdForPriceChange", 0.01
        )
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = (
            registered_classes if registered_classes is not None else []
        )
        self.current_time: int | str = -1
        self.current_time_step: int = -1

    def sync_time(self, current_time: int | str, current_time_step: int) -> None:
        """Synchronize the current time and time step in the summarizer with the MemoryHandler."""
        self.current_time = current_time
        self.current_time_step = current_time_step

    def summarize_memory(self, agent_memory: AgentMemory) -> dict[str, str]:
        summary_specs: dict[str, tuple[Deque, Callable[[Deque], str]]] = {
            "move_history": (
                agent_memory.move_history,
                self._summarize_move_history,
            ),
            "consumption_history": (
                agent_memory.consumption_history,
                self._summarize_consumption_history,
            ),
            "purchase_history": (
                agent_memory.purchase_history,
                self._summarize_purchase_history,
            ),
            "sale_history": (
                agent_memory.sale_history,
                self._summarize_sale_history,
            ),
            "exchange_history": (
                agent_memory.exchange_history,
                self._summarize_exchange_history,
            ),
            "set_price_history": (
                agent_memory.set_price_history,
                self._summarize_set_price_history,
            ),
            "inner_thought_history": (
                agent_memory.inner_thought_history,
                self._summarize_inner_thought_history,
            ),
            "social_history": (
                agent_memory.social_history,
                self._summarize_social_history,
            ),
            "state_evaluation_history": (
                agent_memory.state_evaluation_history,
                self._summarize_state_evaluation_history,
            ),
            "obs_history": (
                agent_memory.obs_history,
                self._summarize_obs_history,
            ),
        }

        summarized_memory: dict[str, str] = {}
        for field_name, (history, summarize_func) in summary_specs.items():
            base_summary: str = summarize_func(history)
            summarized_memory[field_name] = base_summary
            additional_info: dict[str, Any] = self._postprocess_summary(
                field_name=field_name,
                history=history,
                base_summary=base_summary,
            )
            summarized_memory.update(additional_info)
        return summarized_memory

    def _postprocess_summary(
        self,
        field_name: str,
        history: Deque[
            ConsumptionHistoryItem
            | MoveHistoryItem
            | PurchaseHistoryItem
            | SaleHistoryItem
            | ExchangeHistoryItem
            | SetPriceHistoryItem
            | SocialHistoryItem
            | StateEvaluationHistoryItem
            | ObsHistoryItem
            | InnerThoughtHistoryItem
        ],
        base_summary: str,
    ) -> dict[str, Any]:
        """Hook for subclasses to append or transform each summary."""
        return {}

    def _summarize_move_history(self, move_history: Deque[MoveHistoryItem]) -> str:
        if not move_history:
            return "You have no movement history."
        return (
            "You have moved to "
            + " -> ".join(f"{item.pos}" for item in move_history)
            + "."
        )

    def _summarize_consumption_history(
        self, consumption_history: Deque[ConsumptionHistoryItem]
    ) -> str:
        if not consumption_history:
            return "You have no consumption history."
        return (
            "You have consumed "
            + ", ".join(
                f"{item.item_name} x {int(item.quantity)} at time {item.time}"
                for item in consumption_history
            )
            + "."
        )

    def _summarize_purchase_history(
        self, purchase_history: Deque[PurchaseHistoryItem]
    ) -> str:
        if not purchase_history:
            return "You have no purchase history."
        return (
            "You have purchased "
            + ", ".join(
                f"{item.item_name} x {int(item.quantity)} at "
                f"{int(item.price)} from agent_id {item.from_agent_id} at time {item.time}"
                for item in purchase_history
            )
            + "."
        )

    def _summarize_sale_history(self, sale_history: Deque[SaleHistoryItem]) -> str:
        if not sale_history:
            return "You have no sale history."
        return (
            "You have sold "
            + ", ".join(
                f"{item.item_name} x {int(item.quantity)} at {int(item.price)} "
                f"to agent_id {item.to_agent_id} at time {item.time}"
                for item in sale_history
            )
            + "."
        )

    def _summarize_exchange_history(
        self, exchange_history: Deque[ExchangeHistoryItem]
    ) -> str:
        if not exchange_history:
            return "You have no exchange history."
        return (
            "You have exchanged "
            + "; ".join(
                f"give {item.give_item_name} x {int(item.give_item_quantity)}, "
                f"get {item.get_item_name} x {int(item.get_item_quantity)} "
                f"with agent_id {item.counterparty_id} at time {item.time}"
                for item in exchange_history
            )
            + "."
        )

    def _summarize_set_price_history(
        self, set_price_history: Deque[SetPriceHistoryItem]
    ) -> str:
        if not set_price_history:
            return "You have no price change history."
        return (
            "You have changed price "
            + ", ".join(
                f"{item.item_name}: {int(item.old_price)} -> {int(item.new_price)} at time {item.time}"
                for item in set_price_history
            )
            + "."
        )

    def _summarize_inner_thought_history(
        self, inner_thought_history: Deque[InnerThoughtHistoryItem]
    ) -> str:
        if not inner_thought_history:
            return "You have no inner thought history."
        sorted_items: list[InnerThoughtHistoryItem] = sorted(
            inner_thought_history, key=lambda item: item.time_step
        )
        latest_item: InnerThoughtHistoryItem = sorted_items[-1]
        return (
            "Your latest inner thought is: "
            + f"'{latest_item.inner_thought}' at time {latest_item.time}"
        )

    def _summarize_social_history(
        self, social_history: Deque[SocialHistoryItem]
    ) -> str:
        if not social_history:
            return "You have no social action history."
        return (
            "Your social actions are "
            + "; ".join(
                f"{item.action} target_agent_id {item.target_agent_id} at time {item.time} "
                for item in social_history
            )
            + "."
        )

    def _summarize_state_evaluation_history(
        self, state_evaluation_history: Deque[StateEvaluationHistoryItem]
    ) -> str:
        if not state_evaluation_history:
            return "You have no state evaluation history."
        return (
            "Your state evaluations are "
            + "; ".join(
                f"Wealth: {int(item.wealth)} at time {item.time}"
                for item in state_evaluation_history
            )
            + "."
        )

    def _summarize_obs_history(self, obs_history: Deque[ObsHistoryItem]) -> str:
        if not obs_history:
            return "You have no observation history."
        if not hasattr(self, "obs_summarizer_registry"):
            self.obs_summarizer_registry: dict[str, ObsSummarizer] = (
                self._build_obs_summarizer_registry()
            )
        obs_type2items: dict[str, list[ObsHistoryItem]] = defaultdict(list)
        for item in obs_history:
            obs_type2items[item.obs_type].append(item)
        message: str = ""
        for obs_type, items in obs_type2items.items():
            if obs_type in self.obs_summarizer_registry:
                summarizer_func: ObsSummarizer = self.obs_summarizer_registry[obs_type]
                summarized_obs: str = summarizer_func(items)
                message += f"{summarized_obs} "
        if not message:
            return "You have no recognizable observation history."
        else:
            return "Your observations history are: " + message.strip()

    def _build_obs_summarizer_registry(self) -> dict[str, ObsSummarizer]:
        """Build a registry that maps observation types to their corresponding summarizer functions.

        Returns:
            dict[str, ObsSummarizer]: Dispatch table for observation summarizer functions.

        Note:
            _summarize_{obs_type} methods are defined for each observation type to summarize the observations of that type.
            You can implement specific summarization logic for each observation type in these methods.
        """
        return {
            "others_pos": self._summarize_others_pos,
            "self_salary": self._summarize_self_salary,
            "self_inventory": self._summarize_self_inventory,
            "self_tweet": self._summarize_self_tweet,
            "follow_cap": self._summarize_follow_cap,
            "num_followers": self._summarize_num_followers,
            "num_follows": self._summarize_num_follows,
            "visible_tl": self._summarize_visible_tl,
            "item_name2price": self._summarize_item_name2price,
            "others_inventory": self._summarize_others_inventory,
        }

    def _summarize_others_pos(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_self_salary(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_self_inventory(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_self_tweet(self, obs_items: list[ObsHistoryItem]) -> str:
        return summarize_self_tweet_frequency(obs_items)

    def _summarize_follow_cap(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_num_followers(self, obs_items: list[ObsHistoryItem]) -> str:
        return summarize_num_changes(obs_items, is_follow=False)

    def _summarize_num_follows(self, obs_items: list[ObsHistoryItem]) -> str:
        return summarize_num_changes(obs_items, is_follow=True)

    def _summarize_visible_tl(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_timedelta(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_item_name2price(self, obs_items: list[ObsHistoryItem]) -> str:
        return ""

    def _summarize_others_inventory(self, obs_items: list[ObsHistoryItem]) -> str:
        return summarize_observed_price_changes(
            obs_items, relative_threshold=self.relative_threshold_for_price_change
        )
