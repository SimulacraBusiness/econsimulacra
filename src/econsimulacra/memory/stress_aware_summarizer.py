from __future__ import annotations

import random
from typing import Any, Callable, Deque, Optional, Type, cast

from ..sim_utils import find_class
from .memory_items import (
    ConsumptionHistoryItem,
    ExchangeHistoryItem,
    InnerThoughtHistoryItem,
    MoveHistoryItem,
    ObsHistoryItem,
    PurchaseHistoryItem,
    SaleHistoryItem,
    SetPriceHistoryItem,
    SleepHistoryItem,
    SocialHistoryItem,
    StateEvaluationHistoryItem,
)
from .stress_utils import (
    calc_stress_from_consumption_history,
    calc_stress_from_move_history,
    calc_stress_from_obs_history,
    calc_stress_from_sleep_history,
    calc_stress_from_state_evaluation_history,
)
from .summarizer import MemorySummarizer

ObsSummarizer = Callable[[list[ObsHistoryItem]], str]


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
                stressTypes: the list of stress types to calculate. Each stress type corresponds to a history field name,
                    such as "consumption_history", "move_history" , or "state_evaluation_history".
                item2Weight: a dictionary mapping item names to their corresponding weights for stress calculation from consumption history.
                and may contain:
                maxMagnitude: the maximum magnitude of the stress level.
                    The stress level will be normalized to be between 0 and maxMagnitude.
                targetSleepDuration: the target sleep duration for stress calculation from sleep history.
                windowSizeForSleep: the size of the time window in time steps to consider for stress calculation from sleep history.
                durationWeightForSleep: the weight for sleep duration in stress calculation from sleep history.
                regularityWeightForSleep: the weight for sleep regularity in stress calculation from sleep history
                targetConsumptionQuantity: the target quantity to consume for stress calculation from consumption history.
                windowSizeForConsumption: the size of the time window in time steps to consider for stress calculation.
                timeDecayForConsumption: the decay factor for the stress contribution of past consumption events.
                targetMoveDistance: the target distance to move for stress calculation.
                windowSizeForMove: the size of the time window in time steps to consider for stress calculation from move history.
                timeDecayForMove: the decay factor for the stress contribution of past move events.
                homeComfortWeight: the weight for home comfort in stress calculation from move history.
                toleranceThresholdForStress: the tolerance threshold for stress.
                targetBuyingPower: the target buying power for stress calculation from state evaluation history.
                targetRelativeWealth: the target relative wealth for stress calculation from state evaluation history.
                targetWealthGrowth: the target wealth growth for stress calculation from state evaluation history.
                windowSizeForStateEvaluation: the size of the time window in time steps to consider for stress calculation from state evaluation history.
                buyingPowerWeight: the weight for buying power in stress calculation from state evaluation history.
                relativeWealthWeight: the weight for relative wealth in stress calculation from state evaluation history.
                wealthDrawdownWeight: the weight for wealth drawdown in stress calculation from state evaluation history.
                targetNumNearby: the target number of nearby agents for stress calculation from obs history.

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
        self.stress_types: list[str] = config.get("stressTypes", [])
        self.max_magnitude: int = config.get("maxMagnitude", 100)
        self.target_sleep_duration: float = config.get("targetSleepDuration", 8.0)
        self.window_size_for_sleep: int = config.get("windowSizeForSleep", 24)
        self.duration_weight_for_sleep: float = config.get(
            "durationWeightForSleep", 0.8
        )
        self.regularity_weight_for_sleep: float = config.get(
            "regularityWeightForSleep", 0.2
        )
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
        self.target_move_distance: float = config.get("targetMoveDistance", 5.0)
        self.window_size_for_move: int = config.get("windowSizeForMove", 10)
        self.time_decay_for_move: float = config.get("timeDecayForMove", 0.8)
        self.home_comfort_weight: float = config.get("homeComfortWeight", 0.2)
        self.target_buying_power: float = config.get("targetBuyingPower", 100.0)
        self.target_relative_wealth: float = config.get("targetRelativeWealth", -0.1)
        self.target_wealth_growth: float = config.get("targetWealthGrowth", 0.0)
        self.window_size_for_state_evaluation: int = config.get(
            "windowSizeForStateEvaluation", 10
        )
        self.buying_power_weight: float = config.get("buyingPowerWeight", 1.0)
        self.relative_wealth_weight: float = config.get("relativeWealthWeight", 0.6)
        self.wealth_drawdown_weight: float = config.get("wealthDrawdownWeight", 0.2)
        self.target_num_nearby: int = config.get("targetNumNearby", 3)
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
                        | InnerThoughtHistoryItem
                        | SleepHistoryItem
                        | SocialHistoryItem
                        | StateEvaluationHistoryItem
                        | ObsHistoryItem
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
            "inner_thought_history": self._calc_stress_from_inner_thought_history_dispatch,
            "set_price_history": self._calc_stress_from_set_price_history_dispatch,
            "social_history": self._calc_stress_from_social_history_dispatch,
            "state_evaluation_history": (
                self._calc_stress_from_state_evaluation_history_dispatch
            ),
            "sleep_history": self._calc_stress_from_sleep_history_dispatch,
            "obs_history": self._calc_stress_from_obs_history_dispatch,
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
            | InnerThoughtHistoryItem
            | SleepHistoryItem
            | SocialHistoryItem
            | StateEvaluationHistoryItem
            | ObsHistoryItem
        ],
    ) -> tuple[Optional[int], str]:
        """Summarize the stress level based on the history.

        Args:
            field_name (str): the name of the history field.
            history (Deque[...]): the history corresponding to field_name.

        Returns:
            tuple[Optional[int], str]: the summarized stress level and description.
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
            return None, ""
        clipped_stress_level: int = max(0, min(int(stress_level), self.max_magnitude))
        return clipped_stress_level, self._format_stress_text(
            field_name, clipped_stress_level, stress_description
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

    def _calc_stress_from_sleep_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_sleep_history(
            cast(Deque[SleepHistoryItem], history)
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

    def _calc_stress_from_inner_thought_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_inner_thought_history(
            cast(Deque[InnerThoughtHistoryItem], history)
        )

    def _calc_stress_from_social_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_social_history(
            cast(Deque[SocialHistoryItem], history)
        )

    def _calc_stress_from_obs_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_obs_history(cast(Deque[ObsHistoryItem], history))

    def _calc_stress_from_state_evaluation_history_dispatch(
        self,
        history: Deque[Any],
    ) -> tuple[Optional[int], str]:
        return self._calc_stress_from_state_evaluation_history(
            cast(Deque[StateEvaluationHistoryItem], history)
        )

    def _calc_stress_from_consumption_history(
        self,
        history: Deque[ConsumptionHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the consumption history."""
        if "consumption_history" in self.stress_types:
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
        else:
            return None, ""

    def _calc_stress_from_sleep_history(
        self,
        history: Deque[SleepHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the sleep history."""
        if "sleep_history" in self.stress_types:
            return calc_stress_from_sleep_history(
                sleep_history=history,
                current_time=self.current_time,
                current_time_step=self.current_time_step,
                max_stress=self.max_magnitude,
                target_sleep_duration=self.target_sleep_duration,
                window_size=self.window_size_for_sleep,
                duration_weight=self.duration_weight_for_sleep,
                regularity_weight=self.regularity_weight_for_sleep,
                tolerance_threshold=self.tolerance_threshold_for_stress,
            )
        return None, ""

    def _calc_stress_from_move_history(
        self,
        history: Deque[MoveHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the move history."""
        if "move_history" in self.stress_types:
            return calc_stress_from_move_history(
                move_history=history,
                current_time_step=self.current_time_step,
                max_stress=self.max_magnitude,
                target_distance=self.target_move_distance,
                window_size=self.window_size_for_move,
                time_decay=self.time_decay_for_move,
                tolerance_threshold=self.tolerance_threshold_for_stress,
                home_comfort=self.home_comfort_weight,
            )
        else:
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

    def _calc_stress_from_inner_thought_history(
        self,
        history: Deque[InnerThoughtHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the inner thought history."""
        return None, ""

    def _calc_stress_from_social_history(
        self,
        history: Deque[SocialHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the social history."""
        return None, ""

    def _calc_stress_from_state_evaluation_history(
        self,
        history: Deque[StateEvaluationHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the state evaluation history."""
        if "state_evaluation_history" in self.stress_types:
            return calc_stress_from_state_evaluation_history(
                state_evaluation_history=history,
                current_time_step=self.current_time_step,
                max_stress=self.max_magnitude,
                target_buying_power=self.target_buying_power,
                target_relative_wealth=self.target_relative_wealth,
                target_wealth_growth=self.target_wealth_growth,
                window_size=self.window_size_for_state_evaluation,
                buying_power_weight=self.buying_power_weight,
                relative_wealth_weight=self.relative_wealth_weight,
                wealth_drawdown_weight=self.wealth_drawdown_weight,
                tolerance_threshold=self.tolerance_threshold_for_stress,
            )
        else:
            return None, ""

    def _calc_stress_from_obs_history(
        self,
        history: Deque[ObsHistoryItem],
    ) -> tuple[Optional[int], str]:
        """Calculate the stress level from the obs history."""
        if "obs_history" in self.stress_types:
            return calc_stress_from_obs_history(
                history,
                current_time_step=self.current_time_step,
                target_num_nearby=self.target_num_nearby,
                max_stress=self.max_magnitude,
                tolerance_threshold=self.tolerance_threshold_for_stress,
            )
        else:
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
                "type": the type of the summarizer.
                "stressCalculator": a dictionary with the following keys:
                    "type": the type of the stress calculator.
            prng (random.Random, optional): the pseudo-random number generator to use.
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
            | InnerThoughtHistoryItem
            | SocialHistoryItem
            | StateEvaluationHistoryItem
            | ObsHistoryItem
            | SleepHistoryItem
        ],
        base_summary: str,
    ) -> dict[str, Any]:
        self.stress_calculator.sync_time(self.current_time, self.current_time_step)
        stress_level, stress_reason = self.stress_calculator.summarize_stress(
            field_name=field_name,
            history=history,
        )
        if stress_level is None:
            return {}
        return {
            f"{field_name}_stress": stress_level,
            f"{field_name}_stress_reason": stress_reason,
        }
