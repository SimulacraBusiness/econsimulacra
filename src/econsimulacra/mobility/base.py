from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Optional

Amount = float | int


def _validate_positive_amounts(
    values: Mapping[str, Any], field_name: str
) -> dict[str, Amount]:
    """Validate a mapping of item names to positive amounts.

    Args:
        values (Mapping[str, Any]): Item names and their configured amounts.
        field_name (str): Configuration field name used in error messages.

    Returns:
        dict[str, Amount]: A validated copy of the amounts.

    Note:
        Boolean values are rejected even though ``bool`` is a subclass of ``int``.
    """
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")

    validated: dict[str, Amount] = {}
    for item_name, amount in values.items():
        if not isinstance(item_name, str) or not item_name:
            raise ValueError(f"Keys in {field_name} must be non-empty strings.")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"Amounts in {field_name} must be numeric.")
        if not isfinite(amount) or amount <= 0:
            raise ValueError(f"Amounts in {field_name} must be finite and positive.")
        validated[item_name] = amount
    return validated


@dataclass(frozen=True)
class MobilityMode:
    """Describe the ownership, speed, and resource use of a mobility mode."""

    name: str
    velocity: int
    required_items: Mapping[str, Amount]
    consumption_per_cell: Mapping[str, Amount]
    item_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate and isolate the mobility mode's values.

        Args:
            None.

        Returns:
            None.

        Note:
            Configuration mappings are copied and exposed as read-only mappings.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Mobility mode name must be a non-empty string.")
        if (
            isinstance(self.velocity, bool)
            or not isinstance(self.velocity, int)
            or self.velocity <= 0
        ):
            raise ValueError("Mobility mode velocity must be a positive integer.")
        if self.item_name is not None and (
            not isinstance(self.item_name, str) or not self.item_name
        ):
            raise ValueError(
                "Mobility mode itemName must be a non-empty string or None."
            )

        required_items = _validate_positive_amounts(
            self.required_items, "requiredItems"
        )
        consumption_per_cell = _validate_positive_amounts(
            self.consumption_per_cell, "consumptionPerCell"
        )
        object.__setattr__(self, "required_items", MappingProxyType(required_items))
        object.__setattr__(
            self,
            "consumption_per_cell",
            MappingProxyType(consumption_per_cell),
        )

    @classmethod
    def from_config(cls, name: str, config: Mapping[str, Any]) -> "MobilityMode":
        """Create a mobility mode from a configuration mapping.

        Args:
            name (str): Name used to select the mobility mode.
            config (Mapping[str, Any]): Mobility mode configuration.

        Returns:
            MobilityMode: The validated mobility mode.

        Note:
            ``itemName``, ``requiredItems``, and ``consumptionPerCell`` are optional.
            A mode with no ``itemName`` does not require a durable inventory item.
        """
        if not isinstance(config, Mapping):
            raise TypeError(
                f"Configuration for mobility mode {name!r} must be a mapping."
            )
        if "velocity" not in config:
            raise ValueError(f"Mobility mode {name!r} requires velocity.")

        return cls(
            name=name,
            velocity=config["velocity"],
            required_items=config.get("requiredItems", {}),
            consumption_per_cell=config.get("consumptionPerCell", {}),
            item_name=config.get("itemName", None if name == "Walking" else name),
        )
