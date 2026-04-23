from __future__ import annotations

import random
from typing import Any, Callable, Deque, Optional, Type, cast

from ..sim_utils import find_class
from .base import (
    ConsumptionHistoryItem,
    ExchangeHistoryItem,
    MemorySummarizer,
    MoveHistoryItem,
    PurchaseHistoryItem,
    SaleHistoryItem,
    SetPriceHistoryItem,
    SocialHistoryItem,
    StateEvaluationItem,
)
from .stress_utils import calc_stress_from_consumption_history


class StressCalculator:
    """Stress Calculator class.

    StressCalculator is a class that calculates the stress level of
    the agent based on the memory of the agent.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): the configuration for the StressCalculator. It must contain:
                type: the type of the stress calculator.
                item2Weight: a dictionary mapping item names to their corresponding weights for stress calculation from consumption history.
                and may contain:
                maxMagnitude: the maximum magnitude of the stress level.
                    The stress level will be normalized to be between 0 and maxMagnitude.
                targetConsumptionQuantity: the target quantity to consume for stress calculation.
                windowSizeForConsumption: the size of the time window in time steps to consider for stress calculation.
                timeDecayForConsumption: the decay factor for the stress contribution of past consumption events.
                toleranceThresholdForStress: the tolerance threshold for stress.

            prng (Optional[random.Random]): the pseudo-random number generator to use.
                If None, a new random.Random instance will be created.
            registred_classes (list[Type]): the list of registered classes.
        """
        if "item2Weight" not in config:
            raise ValueError(
                "item2Weight is required to calculate stress from consumption history."
            )
        else:
            self.item_name2weight: dict[str, float] = config["item2Weight"]
        self.max_magnitude: int = config.get("maxMagnitude", 100)
        self.target_consumption_quantity: int = config.get(
            "targetConsumptionQuantity", 10
        )
        self.window_size_for_consumption: int = config.get(
            "windowSizeForConsumption", 10
        )
        self.time_decay_for_consumption: float = config.get(
            "timeDecayForConsumption", 0.8
        )
        self.tolerance_threshold_for_stress: float = config.get(
            "toleranceThresholdForStress", 0.1
        )
        self._stress_dispatch: dict[
            str,
            Callable[
                [
                    Deque[
                        ConsumptionHistoryItem
                        | MoveHistoryItem
                        | PurchaseHistoryItem
                        | SaleHistoryItem
                        | ExchangeHistoryItem
                        | SetPriceHistoryItem
                        | SocialHistoryItem
                        | StateEvaluationItem
                    ]
                ],
                tuple[Optional[int], str],
            ],
        ] = {
            "consumption_history": self._calc_stress_from_consumption_history_dispatch,
            "move_history": self._calc_stress_from_move_history_dispatch,
            "purchase_history": self._calc_stress_from_purchase_history_dispatch,
            "sale_history": self._calc_stress_from_sale_history_dispatch,
            "exchange_history": self._calc_stress_from_exchange_history_dispatch,
            "set_price_history": self._calc_stress_from_set_price_history_dispatch,
            "social_history": self._calc_stress_from_social_history_dispatch,
            "state_evaluation_history": (
                self._calc_stress_from_state_evaluation_history_dispatch
            ),
        }
        self.current_time: int | str = -1
        self.current_time_step: int = -1

    def sync_time(self, time: int | str, time_step: int) -> None:
        """Sync the current time and time step of the stress calculator."""
        self.current_time = time
        self.current_time_step = time_step

    def summarize_stress(
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
            | StateEvaluationItem
        ],
    ) -> str:
        """Summarize the stress level based on the history.

        Args:
            field_name (str): the name of the history field.
            history (Deque[...]): the history corresponding to field_name.

        Returns:
            str: the summarized stress text.
        """
        handler = self._stress_dispatch.get(field_name)
        if handler is None:
            raise ValueError(
                f"Unsupported history field for stress calculation: {field_name}"
            )
        stress_level: Optional[int]
        stress_description: str
        stress_level, stress_description = handler(history)
        if stress_level is None:
            return ""

        clipped_stress_level: int = max(0, min(int(stress_level), self.max_magnitude))
        return self._format_stress_text(
            field_name=field_name,
            stress_level=clipped_stress_level,
            stress_description=stress_description,
        )

    def _format_stress_text(
        self,
        field_name: str,
        stress_level: int,
        stress_description: str,
    ) -> str:
        """Format the stress level into a natural language text."""
        readable_field_name = field_name.removesuffix("_history").replace("_", " ")

        if stress_level == 0:
            return (
                f"Your stress level from this {readable_field_name} is 0 "
                f"out of {self.max_magnitude}. {stress_description}"
            )
        return (
            f"Your stress level from this {readable_field_name} is "
            f"{stress_level} out of {self.max_magnitude}. {stress_description}"
        )

    def _calc_stress_from_consumption_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_consumption_history(
            cast(Deque[ConsumptionHistoryItem], history)
        )

    def _calc_stress_from_move_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_move_history(
            cast(Deque[MoveHistoryItem], history)
        )

    def _calc_stress_from_purchase_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_purchase_history(
            cast(Deque[PurchaseHistoryItem], history)
        )

    def _calc_stress_from_sale_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_sale_history(
            cast(Deque[SaleHistoryItem], history)
        )

    def _calc_stress_from_exchange_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_exchange_history(
            cast(Deque[ExchangeHistoryItem], history)
        )

    def _calc_stress_from_set_price_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_set_price_history(
            cast(Deque[SetPriceHistoryItem], history)
        )

    def _calc_stress_from_social_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_social_history(
            cast(Deque[SocialHistoryItem], history)
        )

    def _calc_stress_from_state_evaluation_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_state_evaluation_history(
            cast(Deque[StateEvaluationItem], history)
        )

    def _calc_stress_from_consumption_history(
        self,
        history: Deque[ConsumptionHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the consumption history."""
        return calc_stress_from_consumption_history(
            consumption_history=history,
            current_time_step=self.current_time_step,
            max_stress=self.max_magnitude,
            target_quantity=self.target_consumption_quantity,
            window_size=self.window_size_for_consumption,
            time_decay=self.time_decay_for_consumption,
            tolerance_threshold=self.tolerance_threshold_for_stress,
            item2weight=self.item_name2weight,
        )

    def _calc_stress_from_move_history(
        self,
        history: Deque[MoveHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the move history."""
        return None, ""

    def _calc_stress_from_purchase_history(
        self,
        history: Deque[PurchaseHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the purchase history."""
        return None, ""

    def _calc_stress_from_sale_history(
        self,
        history: Deque[SaleHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the sale history."""
        return None, ""

    def _calc_stress_from_exchange_history(
        self,
        history: Deque[ExchangeHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the exchange history."""
        return None, ""

    def _calc_stress_from_set_price_history(
        self,
        history: Deque[SetPriceHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the set price history."""
        return None, ""

    def _calc_stress_from_social_history(
        self,
        history: Deque[SocialHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the social history."""
        return None, ""

    def _calc_stress_from_state_evaluation_history(
        self,
        history: Deque[StateEvaluationItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the state evaluation history."""
        return None, ""


class StressAwareSummarizer(MemorySummarizer):
    """Stress Aware Summarizer class.

    StressAwareSummarizer is a MemorySummarizer that summarizes the memory of
    the agent into a form that can be provided as a part of the observation to the agent,
    while being aware of the stress level of the agent.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): the configuration for the StressAwareSummarizer. It must contain:
                stressCalculator: a dictionary with the following keys:
                    type: the type of the stress calculator.
            prng (Optional[random.Random]): the pseudo-random number generator to use.
                If None, a new random.Random instance will be created.
            registred_classes (list[Type]): the list of registered classes.


         Note:
             StressAwareSummarizer is a MemorySummarizer that summarizes the memory of the agent into a form that can be provided as a part of the observation to the agent, while being aware of the stress level of the agent.
                The summarize_memory method is called by the MemoryHandler.get_memory method to summarize the memory of the agent.
        """
        super().__init__(config, prng, registered_classes)
        if "stressCalculator" in config:
            stress_calculator_config = config["stressCalculator"]
            if "type" not in stress_calculator_config:
                raise ValueError("StressCalculator config must contain type")
            stress_calculator_type = stress_calculator_config["type"]
            stress_calculator_class: Type[StressCalculator] = find_class(
                stress_calculator_type, registered_classes
            )
            self.stress_calculator = stress_calculator_class(
                stress_calculator_config, prng, registered_classes
            )
        else:
            raise ValueError(
                "StressAwareSummarizer requires stressCalculator in config"
            )

    def sync_time(self, current_time: int | str, current_time_step: int) -> None:
        super().sync_time(current_time, current_time_step)
        self.stress_calculator.sync_time(self.current_time, self.current_time_step)

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
            | StateEvaluationItem
        ],
        base_summary: str,
    ) -> str:
        self.stress_calculator.sync_time(self.current_time, self.current_time_step)
        stress_text: str = self.stress_calculator.summarize_stress(
            field_name=field_name,
            history=history,
        )
        if not stress_text:
            return base_summary
        return f"{base_summary} {stress_text}"
