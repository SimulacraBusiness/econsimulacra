from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MovementState:
    """Represent an active journey to a destination with one mobility mode."""

    is_moving: bool
    destination: tuple[int, ...]
    mobility_name: str

    def __post_init__(self) -> None:
        """Validate the active movement state.

        Args:
            None.

        Returns:
            None.

        Note:
            Environment uses ``None`` rather than this class for an inactive journey.
        """
        if not isinstance(self.is_moving, bool):
            raise TypeError("is_moving must be a boolean.")
        if not self.is_moving:
            raise ValueError("MovementState represents only an active journey.")
        if not isinstance(self.destination, tuple) or not all(
            isinstance(coordinate, int) and not isinstance(coordinate, bool)
            for coordinate in self.destination
        ):
            raise TypeError("destination must be a tuple of integers.")
        if not isinstance(self.mobility_name, str) or not self.mobility_name:
            raise ValueError("mobility_name must be a non-empty string.")
