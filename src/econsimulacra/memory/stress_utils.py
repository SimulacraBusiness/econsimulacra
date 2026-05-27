from __future__ import annotations

from datetime import datetime
from math import cos, pi, sin, sqrt
from typing import Deque, Optional

from .base import (
    ConsumptionHistoryItem,
    MoveHistoryItem,
    SleepHistoryItem,
    StateEvaluationHistoryItem,
)


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
        tuple[int, str]:
            A tuple containing:

            - stress_level: The calculated stress level.
            - stress_reason: The reason for the stress level.

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
        tuple[int, str]:
            A tuple containing:

            - stress_level: The calculated stress level.
            - stress_reason: The reason for the stress level.

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


def calc_stress_from_state_evaluation_history(
    state_evaluation_history: Deque[StateEvaluationHistoryItem],
    current_time_step: int,
    max_stress: int,
    target_buying_power: float,
    target_relative_wealth: float,
    target_wealth_growth: float,
    window_size: int,
    tolerance_threshold: float,
    buying_power_weight: float = 1.0,
    relative_wealth_weight: float = 1.0,
    wealth_drawdown_weight: float = 1.0,
) -> tuple[int, str]:
    """Calculate economic stress based on state evaluation history.

    Args:
        state_evaluation_history (Deque[StateEvaluationHistoryItem]):
            A deque of StateEvaluationHistoryItem representing the agent's economic state history.
        current_time_step (int): The current time step in the simulation.
        max_stress (int): The maximum stress level.
        target_buying_power (float): The target buying power level.
        target_relative_wealth (float): The target relative wealth level.
        target_wealth_growth (float): The target wealth growth over the window.
        window_size (int): The size of the time window in time steps.
        tolerance_threshold (float): The threshold above which stress is reported.
        buying_power_weight (float): Weight for buying-power stress.
        relative_wealth_weight (float): Weight for relative-wealth stress.
        wealth_drawdown_weight (float): Weight for wealth-drawdown stress.

    Returns:
        tuple[int, str]:
            A tuple containing:

            - stress_level: The calculated stress level.
            - stress_reason: The reason for the stress level.

    Note:
        The stress is computed from three economic factors:

        1. Buying-power stress: stress from having insufficient purchasing power.
        2. Relative-wealth stress: stress from having less wealth than others.
        3. Wealth-drawdown stress: stress from recent decreases in wealth.

        The buying-power stress is defined as:

        .. math::

            s_{bp}(t)
            =
            \\max\\left(
            0,
            \\frac{B^* - B(t)}{B^*}
            \\right)

        where :math:`B(t)` is buying power and :math:`B^*` is the target buying power.

        The relative-wealth stress is defined as:

        .. math::

            s_{rw}(t)
            =
            \\max\\left(
            0,
            \\frac{R^* - R(t)}{|R^*| + 1 + \\epsilon}
            \\right)

        where :math:`R(t)` is relative wealth and :math:`R^*` is the target relative wealth.

        The wealth-drawdown stress is defined as:

        .. math::

            s_{dd}(t)
            =
            \\max\\left(
            0,
            \\frac{G^* - G(t)}{|G^*| + w_i(0) \\epsilon}
            \\right)

        where :math:`G(t)` is the wealth change over the window, :math:`w_i(0)` is the initial wealth, and
        :math:`G^*` is the target wealth growth.

        The total stress score is:

        .. math::

            S(t)
            =
            \\alpha s_{bp}(t)
            +
            \\beta s_{rw}(t)
            +
            \\delta s_{dd}(t)

        and the final stress level is:

        .. math::

            \\min\\left(
            s_{\\max},
            \\left\\lfloor S(t) s_{\\max} \\right\\rfloor
            \\right)
    """
    assert 0 <= tolerance_threshold <= max_stress, (
        "tolerance_threshold must be between 0 and max_stress."
    )
    assert max_stress >= 0, "max_stress must be non-negative."
    assert target_buying_power >= 0.0, "target_buying_power must be non-negative."
    assert window_size >= 0, "window_size must be non-negative."
    eps: float = 1e-8
    if len(state_evaluation_history) == 0:
        if current_time_step >= window_size:
            return max_stress, "Economic state not found."
        return 0, ""
    recent_items: list[StateEvaluationHistoryItem] = [
        item
        for item in state_evaluation_history
        if current_time_step - window_size <= item.time_step <= current_time_step
    ]
    if len(recent_items) == 0:
        if current_time_step >= window_size:
            return max_stress, "No recent economic state is available."
        return 0, ""
    latest_item: StateEvaluationHistoryItem = recent_items[-1]

    # ------------------------------------------------------------------
    # 1. Buying-power stress
    # ------------------------------------------------------------------
    buying_power_stress: float = 0.0
    buying_power_reason: Optional[str] = None
    if latest_item.buying_power is not None and target_buying_power > 0.0:
        buying_power_stress = max(
            0.0,
            (target_buying_power - latest_item.buying_power)
            / (target_buying_power + eps),
        )
        if buying_power_stress > 0.0:
            buying_power_reason = (
                "You cannot buy enough goods. "
                + f"(buying power: {latest_item.buying_power:.2f}, "
                + f"target: {target_buying_power:.2f})"
            )

    # ------------------------------------------------------------------
    # 2. Relative-wealth stress
    # ------------------------------------------------------------------
    relative_wealth_stress: float = 0.0
    relative_wealth_reason: Optional[str] = None
    if latest_item.relative_wealth is not None:
        relative_wealth_stress = max(
            0.0,
            (target_relative_wealth - latest_item.relative_wealth)
            / (abs(target_relative_wealth) + 1.0 + eps),
        )
        if relative_wealth_stress > 0.0:
            relative_wealth_reason = (
                "You have less wealth than others. "
                + f"(relative wealth: {latest_item.relative_wealth:.2f}, "
                + f"target: {target_relative_wealth:.2f})"
            )

    # ------------------------------------------------------------------
    # 3. Wealth-drawdown stress
    # ------------------------------------------------------------------
    wealth_drawdown_stress: float = 0.0
    wealth_drawdown_reason: Optional[str] = None
    first_item: StateEvaluationHistoryItem = recent_items[0]
    wealth_growth: float = latest_item.wealth - first_item.wealth
    wealth_drawdown_stress = max(
        0.0,
        (target_wealth_growth - wealth_growth)
        / (abs(target_wealth_growth) + abs(first_item.wealth) + eps),
    )
    if wealth_drawdown_stress > 0.0:
        if wealth_growth < 0.0:
            wealth_drawdown_reason = (
                "Your wealth has recently decreased. "
                + f"(wealth change: {wealth_growth:.2f})"
            )
        else:
            wealth_drawdown_reason = (
                "Your wealth has not increased enough recently. "
                + f"(wealth change: {wealth_growth:.2f}, "
                + f"target growth: {target_wealth_growth:.2f})"
            )

    # ------------------------------------------------------------------
    # Weighted aggregation
    # ------------------------------------------------------------------
    weighted_components: list[tuple[str, float, Optional[str]]] = [
        (
            "buying_power",
            buying_power_weight * buying_power_stress,
            buying_power_reason,
        ),
        (
            "relative_wealth",
            relative_wealth_weight * relative_wealth_stress,
            relative_wealth_reason,
        ),
        (
            "wealth_drawdown",
            wealth_drawdown_weight * wealth_drawdown_stress,
            wealth_drawdown_reason,
        ),
    ]
    total_stress_score: float = sum(score for _, score, _ in weighted_components)
    stress_level: int = min(
        max_stress,
        int(total_stress_score * max_stress),
    )

    # ------------------------------------------------------------------
    # Reason construction
    # ------------------------------------------------------------------
    if stress_level < tolerance_threshold:
        return stress_level, "Your economic condition is acceptable."

    active_reasons: list[tuple[str, float]] = [
        (reason, score)
        for _, score, reason in weighted_components
        if reason is not None and score > 0.0
    ]
    active_reasons.sort(key=lambda x: x[1], reverse=True)
    if len(active_reasons) == 0:
        return (
            stress_level,
            "You feel economic stress, but no dominant cause was identified.",
        )
    reason_parts: list[str] = []
    for reason, score in active_reasons:
        reason_parts.append(reason)
    stress_reason: str = " ".join(reason_parts)
    return stress_level, stress_reason


def calc_stress_from_sleep_history(
    sleep_history: Deque[SleepHistoryItem],
    current_time: int | str,
    current_time_step: int,
    max_stress: int,
    target_sleep_duration: float,
    window_size: float,
    tolerance_threshold: float,
    duration_weight: float = 0.7,
    regularity_weight: float = 0.3,
) -> tuple[int, str]:
    """Calculate sleep stress from sleep duration and sleep regularity.

    This function evaluates the agent's sleep condition over a recent
    time window by combining two stress components:

    1. sleep-duration stress, which increases when the total sleep duration
       within the window is below the target duration; and
    2. sleep-regularity stress, which increases when sleep onset and wake-up
       times are irregular across sleep episodes.

    Args:
        sleep_history (Deque[SleepHistoryItem]):
            A deque of sleep history items. Each item represents one sleep
            interval with ``start_time`` and ``end_time``.
            If any item has ``end_time is None``, this function raises
            ``ValueError`` because ongoing sleep intervals should not be
            evaluated by this function.

        current_time (int | str):
            The current simulation time. If this is an ``int``, all sleep
            times are interpreted as integer simulation steps. If this is a
            ``str``, all sleep times are parsed using ``time_format`` and
            interpreted as datetimes.

        current_time_step (int):
            The current simulation step. This is used only for detecting the
            initial phase of the simulation. If no sleep is observed and
            ``current_time_step < window_size``, the function returns zero
            stress because the agent has not yet had enough time to sleep.

        max_stress (int):
            The maximum possible stress level.

        target_sleep_duration (float):
            The target amount of sleep within the time window. The unit must
            be consistent with ``current_time`` and ``window_size``. If
            ``current_time`` is an ``int``, this is measured in simulation
            steps. If ``current_time`` is a datetime string, this is measured
            in hours.

        window_size (float):
            The size of the retrospective time window. The function evaluates
            sleep intervals overlapping with
            ``[current_time - window_size, current_time]``. If ``current_time``
            is an ``int``, this is measured in simulation steps. If
            ``current_time`` is a datetime string, this is measured in hours.

        time_format (str):
            Datetime format used to parse string-valued times. The default is
            ``"%Y-%m-%d %H:%M:%S"``.

        tolerance_threshold (float):
            Threshold below which the stress level is regarded as acceptable.

        duration_weight (float):
            Weight assigned to sleep-duration stress.

        regularity_weight (float):
            Weight assigned to sleep-regularity stress.

    Returns:
        tuple[int, str]:
            A tuple containing:

            - ``stress_level``: the calculated sleep stress level.
            - ``stress_reason``: a human-readable explanation of the stress.

    Raises:
        ValueError:
            If ``max_stress`` is negative.
        ValueError:
            If ``target_sleep_duration`` is not positive.
        ValueError:
            If ``window_size`` is not positive.
        ValueError:
            If ``tolerance_threshold`` is outside
            ``[0, max_stress]``.
        ValueError:
            If ``duration_weight`` or ``regularity_weight`` is negative.
        ValueError:
            If ``duration_weight + regularity_weight`` is not positive.
        ValueError:
            If a sleep history item has ``end_time is None``.
        ValueError:
            If a sleep interval satisfies ``end_time < start_time``.

    Notes:
        Let :math:`t` be the current time and :math:`W` be the window size.
        The retrospective evaluation window is

        .. math::

            [t - W, t].

        Each completed sleep episode :math:`i` is represented by its start
        time :math:`a_i` and end time :math:`b_i`. Only the overlap between
        the sleep interval and the evaluation window contributes to the
        observed sleep duration:

        .. math::

            d_i(t)
            =
            \\max\\left(
                0,
                \\min(b_i, t) - \\max(a_i, t - W)
            \\right).

        The total sleep duration in the window is therefore

        .. math::

            D(t) = \\sum_i d_i(t).

        Given the target sleep duration :math:`D^*`, the sleep-duration
        stress is defined as

        .. math::

            s_{\\mathrm{dur}}(t)
            =
            \\min\\left(
                s_{\\max},
                \\left\\lfloor
                \\frac{\\max(0, D^* - D(t))}{D^*}
                s_{\\max}
                \\right\\rfloor
            \\right).

        This definition penalizes sleep deficiency but does not penalize
        oversleeping. That is, if :math:`D(t) \\geq D^*`, then
        :math:`s_{\\mathrm{dur}}(t) = 0`.

        Sleep regularity is evaluated using the circular variance of sleep
        onset times and wake-up times. Let :math:`x_j \\in [0, 24)` denote
        clock times in hours. Each clock time is mapped to an angle

        .. math::

            \\theta_j = \\frac{2\\pi x_j}{24}.

        The mean resultant length is

        .. math::

            R
            =
            \\sqrt{
                \\left(
                    \\frac{1}{n} \\sum_{j=1}^n \\cos \\theta_j
                \\right)^2
                +
                \\left(
                    \\frac{1}{n} \\sum_{j=1}^n \\sin \\theta_j
                \\right)^2
            }.

        The circular variance is then

        .. math::

            V = 1 - R.

        Let :math:`V_{\\mathrm{start}}` and :math:`V_{\\mathrm{end}}`
        denote the circular variances of sleep onset times and wake-up times,
        respectively. The sleep-regularity stress is

        .. math::

            s_{\\mathrm{reg}}(t)
            =
            \\left\\lfloor
            \\frac{
                V_{\\mathrm{start}} + V_{\\mathrm{end}}
            }{2}
            s_{\\max}
            \\right\\rfloor.

        If fewer than two completed sleep episodes are available in the
        evaluation window, regularity cannot be estimated and the function
        sets

        .. math::

            s_{\\mathrm{reg}}(t) = 0.

        Finally, the total sleep stress is the weighted average of duration
        stress and regularity stress:

        .. math::

            s(t)
            =
            \\min\\left(
                s_{\\max},
                \\left\\lfloor
                \\tilde{w}_{\\mathrm{dur}} s_{\\mathrm{dur}}(t)
                +
                \\tilde{w}_{\\mathrm{reg}} s_{\\mathrm{reg}}(t)
                \\right\\rfloor
            \\right),

        where normalized weights are given by

        .. math::

            \\tilde{w}_{\\mathrm{dur}}
            =
            \\frac{w_{\\mathrm{dur}}}{
                w_{\\mathrm{dur}} + w_{\\mathrm{reg}}
            },
            \\quad
            \\tilde{w}_{\\mathrm{reg}}
            =
            \\frac{w_{\\mathrm{reg}}}{
                w_{\\mathrm{dur}} + w_{\\mathrm{reg}}
            }.

        Corner cases:

        - If ``sleep_history`` contains an ongoing sleep interval, i.e.,
          ``end_time is None``, this function raises ``ValueError``.
        - If no sleep overlaps with the evaluation window and
          ``current_time_step >= window_size``, the function returns
          ``max_stress``.
        - If no sleep overlaps with the evaluation window and
          ``current_time_step < window_size``, the function returns ``0``.
        - If only one sleep episode is available in the evaluation window,
          sleep-duration stress is computed normally, but
          sleep-regularity stress is set to ``0``.

    Examples:
        Evaluate sleep stress using integer simulation steps:

        .. code-block:: python

            stress, reason = calc_stress_from_sleep_history(
                sleep_history=sleep_history,
                current_time=48,
                current_time_step=48,
                max_stress=10,
                target_sleep_duration=8.0,
                window_size=24.0,
            )

        Evaluate sleep stress using datetime strings:

        .. code-block:: python

            stress, reason = calc_stress_from_sleep_history(
                sleep_history=sleep_history,
                current_time="2026-05-26 09:00:00",
                current_time_step=48,
                max_stress=10,
                target_sleep_duration=8.0,
                window_size=24.0,
                time_format="%Y-%m-%d %H:%M:%S",
            )
    """

    if max_stress < 0:
        raise ValueError("max_stress must be non-negative.")
    if target_sleep_duration <= 0:
        raise ValueError("target_sleep_duration must be positive.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if not 0 <= tolerance_threshold <= max_stress:
        raise ValueError("tolerance_threshold must be between 0 and max_stress.")
    if duration_weight < 0 or regularity_weight < 0:
        raise ValueError("duration_weight and regularity_weight must be non-negative.")
    if duration_weight + regularity_weight <= 0:
        raise ValueError("duration_weight + regularity_weight must be positive.")

    weight_sum: float = duration_weight + regularity_weight
    duration_weight = duration_weight / weight_sum
    regularity_weight = regularity_weight / weight_sum
    if current_time_step < window_size:
        target_sleep_duration = target_sleep_duration * (
            current_time_step / window_size
        )

    current_value: float = _to_continuous_time(current_time)
    window_start: float = current_value - window_size

    sleep_duration: float = 0.0
    start_clock_times: list[float] = []
    end_clock_times: list[float] = []

    for item in sleep_history:
        if item.end_time is None:
            raise ValueError(
                "sleep_history contains an ongoing sleep item: end_time is None."
            )

        start_value: float = _to_continuous_time(item.start_time)
        end_value: float = _to_continuous_time(item.end_time)

        if end_value < start_value:
            raise ValueError(
                f"Invalid sleep interval: end_time < start_time. "
                f"start_time={item.start_time}, end_time={item.end_time}"
            )

        overlap_start: float = max(start_value, window_start)
        overlap_end: float = min(end_value, current_value)
        overlap: float = max(0.0, overlap_end - overlap_start)

        if overlap <= 0.0:
            continue

        sleep_duration += overlap
        start_clock_times.append(_to_clock_time(item.start_time))
        end_clock_times.append(_to_clock_time(item.end_time))

    if sleep_duration == 0.0:
        if current_time_step >= window_size:
            return max_stress, "You have not slept recently. You are exhausted!"
        return 0, ""

    duration_stress = min(
        max_stress,
        int(
            max(0.0, target_sleep_duration - sleep_duration)
            / target_sleep_duration
            * max_stress
        ),
    )

    if len(start_clock_times) < 2:
        regularity_stress = 0
    else:
        start_cv = _circular_variance(start_clock_times, period=24.0)
        end_cv = _circular_variance(end_clock_times, period=24.0)
        regularity_stress = int(((start_cv + end_cv) / 2.0) * max_stress)

    stress_level = min(
        max_stress,
        int(duration_weight * duration_stress + regularity_weight * regularity_stress),
    )

    if stress_level < tolerance_threshold:
        return stress_level, "Acceptable sleep level."

    reasons: list[str] = []

    if duration_stress >= tolerance_threshold:
        reasons.append(
            "You have not slept enough. You had better go back home and sleep. "
            f"(sleep_duration: {sleep_duration:.1f}, "
            f"target: {target_sleep_duration:.1f})"
        )

    if regularity_stress >= tolerance_threshold:
        reasons.append(
            "Your sleep rhythm is irregular. You had better establish a more consistent sleep schedule. "
            f"(regularity_stress: {regularity_stress}, "
            f"duration_stress: {duration_stress})"
        )

    if not reasons:
        reasons.append(
            "Your sleep condition is slightly stressful. "
            f"(duration_stress: {duration_stress}, "
            f"regularity_stress: {regularity_stress})"
        )

    return stress_level, " ".join(reasons)


def _to_continuous_time(value: int | str) -> float:
    """Convert int time or datetime string into continuous time.

    - int is interpreted as simulation step.
    - str is interpreted as datetime and converted into hours.
    """
    if isinstance(value, int):
        return float(value)

    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt.timestamp() / 3600.0


def _to_clock_time(value: int | str) -> float:
    """Convert time into clock time in [0, 24).

    For int time, this assumes 1 step corresponds to 1 hour.
    """
    if isinstance(value, int):
        return float(value % 24)

    dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _circular_variance(values: list[float], period: float) -> float:
    """Calculate circular variance in [0, 1]."""
    if not values:
        return 0.0

    angles = [2.0 * pi * value / period for value in values]
    mean_sin = sum(sin(angle) for angle in angles) / len(angles)
    mean_cos = sum(cos(angle) for angle in angles) / len(angles)
    resultant_length = sqrt(mean_sin**2 + mean_cos**2)

    return 1.0 - resultant_length
