from __future__ import annotations

from typing import Deque, Optional

from .base import ConsumptionHistoryItem


def calc_stress_from_consumption_history(
    consumption_history: Deque[ConsumptionHistoryItem],
    current_time_step: int,
    max_stress: int,
    target_quantity: int,
    window_size: int,
    time_decay: float,
    tolerance_threshold: float,
    item2weight: dict[str, float],
) -> tuple[Optional[int], str]:
    """Calculate the stress level based on the consumption history.

    Args:
        consumption_history (Deque[ConsumptionHistoryItem]):
            A deque of ConsumptionHistoryItem representing the consumption history.
        current_time_step (int): The current time step in the simulation.
        max_stress (int): The maximum stress level.
        target_quantity (int): The target quantity to consume.
        window_size (int): The size of the time window in time steps
            to consider for stress calculation.
        time_decay (float): The decay factor for
            the stress contribution of past consumption events.
        tolerance_threshold (float): The tolerance threshold for stress.
        item2weight (dict[str, float]): A dictionary mapping item names
            to their corresponding weights for stress calculation.

    Returns:
        stress_level (Optional[int]): The calculated stress level,
            None if no stress is incurred.
        stress_reason (str): The reason for the stress level.

    Note:
        The stress level is calculated based on the quantity of items consumed
        within the specified time window, weighted by their respective weights,
        and decayed over time. The quantity of items consumed are calculated
        as follows:
            sum(
                time_decay ** (current_time_step - consumption_time_step)}
                * weight_on_the_item * quantity
                for each consumption event in the time window
            )
        If the calcluated item quantity exceeds or under the target quantity,
        the stress level is calculated as:
            stress_level = min(
                max_stress,
                int(
                    abs(calculated_item_quantity - target_quantity) / target_quantity
                    * max_stress
                ),
            )
        Corner cases:
            - If consumption_history is empty, the stress level is:
                - max_stress if current_time_step >= window_size
                - 0 otherwise
    """
    quantity: float = 0.0
    for history_item in consumption_history:
        time_step: int = history_item.time_step
        if time_step < current_time_step - window_size:
            continue
        item_name: str = history_item.item_name
        if item_name not in item2weight:
            raise ValueError(f"Item '{item_name}' not found in item2weight.")
        weight: float = item2weight[item_name]
        quantity += (
            history_item.quantity
            * weight
            * (time_decay ** (current_time_step - time_step))
        )
    if quantity == 0.0:
        if current_time_step >= window_size:
            return max_stress, "You have not consumed any items. You are starving!"
        else:
            return 0, ""
    stress_level: int = min(
        max_stress,
        int(abs(quantity - target_quantity) / target_quantity * max_stress),
    )
    stress_reason: str
    if tolerance_threshold <= stress_level:
        if quantity < target_quantity:
            stress_reason = (
                "You have not consumed enough items. "
                + f"(quantity: {quantity:.1f}, target: {target_quantity:.1f})"
            )
        else:
            stress_reason = (
                "You have consumed too many items. "
                + f"(quantity: {quantity:.1f}, target: {target_quantity:.1f})"
            )
    else:
        stress_reason = "Acceptable consumption level."
    return stress_level, stress_reason
