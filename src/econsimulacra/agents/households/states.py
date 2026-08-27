from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional, TypeAlias

MODE: TypeAlias = Literal[
    "HOME",
    "AWAY",
    "SLEEPING",
    "TRAVEL_STORE",
    "WAITING_ORDER",
    "RETURN_HOME",
    "RETURN_HOME_SLEEP",
    "RETURN_HOME_MEAL",
]


@dataclass
class HouseholdState:
    r"""Mutable state shared by the household's policy components.

    Args:
        sleep_pressure: Homeostatic sleep stock :math:`H_t`.
        hunger: Bounded hunger stock :math:`G_t`.
        last_meal_elapsed: Hours since the last meal :math:`M_t`.
        home: Fixed initial position :math:`h`, or ``None`` before first act.
        destination: Active movement destination :math:`d_t`, if any.
        mode: Finite-state activity label :math:`z_t`.
        last_step: Last processed simulation step :math:`\ell_t`, if any.
        sleeping_last_interval: Whether the preceding interval was asleep.

    The private state at decision step :math:`t` is

    .. math::

       x_t=(H_t,G_t,M_t,h,d_t,z_t,\ell_t,\sigma_t).
    """

    sleep_pressure: float
    hunger: float
    last_meal_elapsed: float
    home: Optional[tuple[int, ...]] = None
    destination: Optional[tuple[int, ...]] = None
    mode: MODE = "HOME"
    last_step: Optional[int] = None
    has_been_sleeping: bool = False


@dataclass
class DecisionContext:
    r"""Normalized observation supplied to every household policy.

    Args:
        obs: Original EconSimulacra observation :math:`O_t`.
        time_step: Nonnegative simulation step :math:`t`.
        hour: Local clock time :math:`\tau_t\in[0,24)`.
        position: Current grid position :math:`p_t`.
        inventory: Numeric self-inventory mapping :math:`I_t`.

    The adapter computes

    .. math::

       \tau_t=(\tau_0+t\Delta)\bmod 24,

    where :math:`\tau_0` is ``startHour`` and :math:`\Delta` is ``stepHours``.
    """

    obs: dict[str, Any]
    time_step: int
    hour: float
    current_pos: tuple[int, ...]
    inventory: dict[str, float]
