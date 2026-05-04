from __future__ import annotations

from typing import Deque, Optional

from .base import ConsumptionHistoryItem, MoveHistoryItem


def calc_stress_from_consumption_history(
    consumption_history: Deque[ConsumptionHistoryItem],
    current_time_step: int,
    max_stress: int,
    target_quantity: int,
    window_size: int,
    time_decay: float,
    tolerance_threshold: float,
    item2weight: dict[str, float],
) -> tuple[int, str]:
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
        stress_level (int): The calculated stress level.
        stress_reason (str): The reason for the stress level.

    Note:
        The stress level is calculated based on the quantity of items consumed
        within the specified time window, weighted by their respective weights,
        and decayed over time.

        The weighted quantity is defined as:

        .. math::

            Q(t) = \\sum_{k: t_k \\in [t-W, t]}
            \\gamma^{(t - t_k)} \\, w_{i_k} \\, q_k

        where:

        - :math:`t` is the current time step
        - :math:`W` is the window size
        - :math:`\\gamma` is the time decay factor
        - :math:`w_{i_k}` is the weight of item :math:`i_k`
        - :math:`q_k` is the consumed quantity at time :math:`t_k`

        The stress level is then computed as:

        .. math::

            s(t) = \\min\\left(
            s_{\\max},
            \\left\\lfloor
            \\frac{|Q(t) - Q^*|}{Q^*} \\, s_{\\max}
            \\right\\rfloor
            \\right)

        where :math:`Q^*` is the target quantity.

        Corner cases:

        - If ``consumption_history`` is empty and
        :math:`t \\geq W`, then :math:`s(t) = s_{\\max}`.
        - If ``consumption_history`` is empty and
        :math:`t < W`, then :math:`s(t) = 0`.
    """
    assert 0 <= tolerance_threshold <= max_stress, (
        "tolerance_threshold must be between 0 and max_stress."
    )
    assert 0 <= target_quantity, "target_quantity must be non-negative."
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


def calc_stress_from_move_history(
    move_history: Deque[MoveHistoryItem],
    current_time_step: int,
    max_stress: int,
    target_distance: float,
    window_size: int,
    time_decay: float,
    tolerance_threshold: float,
    home_comfort: float,
) -> tuple[int, str]:
    """Calculate the stress level based on the move history.

    Args:
        move_history (Deque[MoveHistoryItem]):
            A deque of MoveHistoryItem representing the move history.
        current_time_step (int): The current time step in the simulation.
        max_stress (int): The maximum stress level.
        target_distance (float): The target distance to move from the initial position.
        window_size (int): The size of the time window in time steps
            to consider for stress calculation.
        time_decay (float): The decay factor for
            the stress contribution of past move events.
        tolerance_threshold (float): The tolerance threshold for stress.
        home_comfort (float): The comfort level of being at home.

    Returns:
        stress_level (int): The calculated stress level.
        stress_reason (str): The reason for the stress level.

    Note:
        The stress level is calculated based on the distance moved during the time steps
        within the specified time window, decayed over time, and adjusted by the home comfort level.

        The distance moved is defined as:

        .. math::

            D(t) = \\sum_{k: t_k \\in [t-W, t]}
            \\gamma^{(t - t_k)} \\, \\|x_k - x_{k-1}\\|

        where:

        - :math:`t` is the current time step
        - :math:`W` is the window size
        - :math:`\\gamma` is the time decay factor
        - :math:`\\|x_k - x_{k-1}\\|` is the distance moved at time :math:`t_k`

        The stress level is then computed as:

        .. math::

            s(t) = \\min\\left(
            s_{\\max},
            \\left\\lfloor
            \\frac{|D(t) - D^*|}{D^*} \\, s_{\\max} \\, (1 - h)\\textbf{1}(x_t=x_0)
            \\right\\rfloor
            \\right)

        where :math:`D^*` is the target distance and :math:`h` is the home comfort level.

        Corner cases:

        - If ``move_history`` is empty and
        :math:`t \\geq W`, then :math:`s(t) = s_{\\max}`.
        - If ``move_history`` is empty and
        :math:`t < W`, then :math:`s(t) = 0`.
    """
    assert 0 <= tolerance_threshold <= max_stress, (
        "tolerance_threshold must be between 0 and max_stress."
    )
    assert 0 <= target_distance, "target_distance must be non-negative."
    assert 0.0 <= home_comfort <= 1.0, "home_comfort must be between 0 and 1."
    distance: float = 0.0
    old_pos: Optional[tuple[int, ...]] = None
    for history_item in move_history:
        new_time_step: int = history_item.time_step
        if new_time_step < current_time_step - window_size:
            continue
        new_pos: tuple[int, ...] = history_item.pos
        if old_pos is not None:
            distance += sum(
                (new - old) ** 2 for new, old in zip(new_pos, old_pos)
            ) ** 0.5 * (time_decay ** (current_time_step - new_time_step))
        old_pos = new_pos
    if distance == 0.0:
        if current_time_step >= window_size:
            return max_stress, "You have not moved at all. You are trapped!"
        else:
            return 0, ""
    stress_level: int = min(
        max_stress,
        int(abs(distance - target_distance) / target_distance * max_stress),
    )
    assert len(move_history) > 0, (
        "move_history is empty even though the distance is not zero."
    )
    latest_history_item: MoveHistoryItem = move_history[-1]
    latest_pos: tuple[int, ...] = latest_history_item.pos
    init_pos: tuple[int, ...] = move_history[0].init_pos
    is_home: bool = init_pos == latest_pos
    if is_home:
        stress_level = int(stress_level * (1 - home_comfort))
    stress_reason: str
    if tolerance_threshold <= stress_level:
        if distance < target_distance:
            stress_reason = (
                "You have not moved enough. "
                + f"(distance: {distance:.1f}, target: {target_distance:.1f})"
            )
        else:
            stress_reason = (
                "You have moved too much. "
                + f"(distance: {distance:.1f}, target: {target_distance:.1f})"
            )
        if is_home:
            stress_reason += (
                " However, being at home makes you feel somewhat comfortable."
            )
    else:
        if is_home:
            stress_reason = "You are at home and feel comfortable. "
        else:
            stress_reason = "Acceptable movement level."
    return stress_level, stress_reason
