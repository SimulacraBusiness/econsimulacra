from __future__ import annotations

from math import floor, isfinite
from random import Random
from types import MappingProxyType
from typing import Any, Mapping, Optional, Type

from .base import Amount, MobilityMode


class MobilityManager:
    """Manage configured mobility modes independently of spatial access rules."""

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[Random] = None,
        registered_classes: Optional[list[Type]] = None,
    ) -> None:
        """Initialize mobility modes from service-provider configuration.

        Args:
            config (dict[str, Any]): Mobility manager configuration.
            prng (Random, optional): Shared pseudo-random number generator.
            registered_classes (list[Type], optional): Registered simulation classes.

        Returns:
            None.

        Note:
            The constructor matches the service-provider interface used by Environment.
        """
        if not isinstance(config, dict):
            raise TypeError("MobilityManager config must be a dictionary.")

        self.config = dict(config)
        self.prng = prng if prng is not None else Random()
        self.registered_classes = list(registered_classes or [])

        raw_modes = config.get("modes")
        if raw_modes is None:
            modes = {"Walking": MobilityMode("Walking", 1, {}, {}, None)}
        else:
            if not isinstance(raw_modes, Mapping):
                raise TypeError("modes must be a mapping.")
            modes = {
                name: MobilityMode.from_config(name, mode_config)
                for name, mode_config in raw_modes.items()
            }
            modes.setdefault("Walking", MobilityMode("Walking", 1, {}, {}, None))

        self.default_mode_name = config.get("defaultMode", "Walking")
        if not isinstance(self.default_mode_name, str):
            raise TypeError("defaultMode must be a string.")
        if self.default_mode_name not in modes:
            raise ValueError(
                f"Default mobility mode {self.default_mode_name!r} is not configured."
            )
        self.name2mode: Mapping[str, MobilityMode] = MappingProxyType(modes)

    def get_mode(self, mode_name: str) -> MobilityMode:
        """Return a configured mobility mode.

        Args:
            mode_name (str): Name of the requested mobility mode.

        Returns:
            MobilityMode: The requested mobility mode.

        Note:
            An unknown name raises ``ValueError`` with a configuration-facing message.
        """
        try:
            return self.name2mode[mode_name]
        except KeyError as error:
            raise ValueError(f"Unknown mobility mode: {mode_name!r}.") from error

    def get_default_mode(self) -> MobilityMode:
        """Return the configured default mobility mode.

        Args:
            None.

        Returns:
            MobilityMode: The default mobility mode.

        Note:
            The default mode is validated during manager initialization.
        """
        return self.name2mode[self.default_mode_name]

    def get_missing_required_items(
        self, inventory: Mapping[str, Amount], mode_name: str
    ) -> dict[str, Amount]:
        """Calculate item deficits that prevent ownership-based use of a mode.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            mode_name (str): Name of the mobility mode.

        Returns:
            dict[str, Amount]: Required item names and their missing amounts.

        Note:
            Consumable resources are not ownership requirements and are excluded.
        """
        mode = self.get_mode(mode_name)
        requirements = dict(mode.required_items)
        if mode.item_name is not None:
            requirements[mode.item_name] = max(requirements.get(mode.item_name, 0), 1)
        missing: dict[str, Amount] = {}
        for item_name, required_amount in requirements.items():
            available_amount = self._get_inventory_amount(inventory, item_name)
            if available_amount < required_amount:
                missing[item_name] = required_amount - available_amount
        return missing

    def is_mode_unlocked(self, inventory: Mapping[str, Amount], mode_name: str) -> bool:
        """Check whether an inventory unlocks a mobility mode.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            mode_name (str): Name of the mobility mode.

        Returns:
            bool: Whether the mobility item and every required item are present.

        Note:
            This method does not check fuel or other consumable resources.
        """
        return not self.get_missing_required_items(inventory, mode_name)

    def can_use_mode(self, inventory: Mapping[str, Amount], mode_name: str) -> bool:
        """Check whether a mobility mode can move at least one cell now.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            mode_name (str): Name of the mobility mode.

        Returns:
            bool: Whether ownership and consumable requirements are satisfied.

        Note:
            This is the validity predicate used by Environment move validation.
        """
        return self.get_effective_velocity(inventory, mode_name) > 0

    def get_unlocked_modes(self, inventory: Mapping[str, Amount]) -> list[MobilityMode]:
        """Return modes whose durable ownership requirements are satisfied.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.

        Returns:
            list[MobilityMode]: Unlocked modes in configuration order.

        Note:
            An unlocked mode may lack the consumables needed to move one cell.
        """
        return [
            mode
            for mode in self.name2mode.values()
            if self.is_mode_unlocked(inventory, mode.name)
        ]

    def get_available_modes(
        self, inventory: Mapping[str, Amount]
    ) -> list[MobilityMode]:
        """Return modes that can currently move at least one cell.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.

        Returns:
            list[MobilityMode]: Available modes in configuration order.

        Note:
            Both ownership and consumable requirements are considered.
        """
        return [
            mode
            for mode in self.name2mode.values()
            if self.can_use_mode(inventory, mode.name)
        ]

    def get_effective_velocity(
        self, inventory: Mapping[str, Amount], mode_name: str
    ) -> int:
        """Calculate the distance a mode can currently travel in one step.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            mode_name (str): Name of the mobility mode.

        Returns:
            int: Travelable cell count, capped by the configured velocity.

        Note:
            Missing ownership items or insufficient consumables yield zero velocity.
        """
        mode = self.get_mode(mode_name)
        if not self.is_mode_unlocked(inventory, mode_name):
            return 0

        effective_velocity = mode.velocity
        for item_name, amount_per_cell in mode.consumption_per_cell.items():
            available_amount = self._get_inventory_amount(inventory, item_name)
            affordable_cells = floor(available_amount / amount_per_cell + 1e-12)
            effective_velocity = min(effective_velocity, affordable_cells)
        return max(effective_velocity, 0)

    def get_usable_modes(self, inventory: Mapping[str, Amount]) -> list[MobilityMode]:
        """Return mobility modes that can move at least one cell now.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.

        Returns:
            list[MobilityMode]: Usable modes in configuration order.

        Note:
            Both ownership requirements and consumable resources are considered.
        """
        return self.get_available_modes(inventory)

    def validate_item_names(self, item_names: set[str]) -> None:
        """Validate that configured inventory references name registered Items.

        Args:
            item_names (set[str]): Item names registered by Environment.

        Returns:
            None.

        Note:
            Validation is delayed because Environment generates services before Items.
        """
        referenced_names: set[str] = set()
        for mode in self.name2mode.values():
            if mode.item_name is not None:
                referenced_names.add(mode.item_name)
            referenced_names.update(mode.required_items)
            referenced_names.update(mode.consumption_per_cell)
        missing_names = referenced_names - item_names
        if missing_names:
            raise ValueError(
                "Mobility configuration references unregistered Items: "
                + ", ".join(sorted(missing_names))
            )

    def calculate_consumption(
        self, mode_name: str, moved_cells: int
    ) -> dict[str, Amount]:
        """Calculate consumables spent for an actual movement distance.

        Args:
            mode_name (str): Name of the mobility mode.
            moved_cells (int): Number of cells actually traversed in one step.

        Returns:
            dict[str, Amount]: Consumable item names and amounts to deduct.

        Note:
            Required ownership items are durable and are never included in the result.
        """
        mode = self.get_mode(mode_name)
        if isinstance(moved_cells, bool) or not isinstance(moved_cells, int):
            raise TypeError("moved_cells must be an integer.")
        if moved_cells < 0:
            raise ValueError("moved_cells must not be negative.")
        if moved_cells > mode.velocity:
            raise ValueError("moved_cells must not exceed the mode velocity.")
        return {
            item_name: amount_per_cell * moved_cells
            for item_name, amount_per_cell in mode.consumption_per_cell.items()
        }

    def get_consumption_shortfall(
        self,
        inventory: Mapping[str, Amount],
        consumption: Mapping[str, Amount],
    ) -> dict[str, Amount]:
        """Calculate inventory deficits for a proposed consumption mapping.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            consumption (Mapping[str, Amount]): Proposed consumable deductions.

        Returns:
            dict[str, Amount]: Consumable names and their missing amounts.

        Note:
            Proposed consumption amounts must be finite and non-negative.
        """
        if not isinstance(consumption, Mapping):
            raise TypeError("consumption must be a mapping.")

        shortfall: dict[str, Amount] = {}
        for item_name, required_amount in consumption.items():
            self._validate_nonnegative_amount(required_amount, "consumption")
            available_amount = self._get_inventory_amount(inventory, item_name)
            if available_amount < required_amount:
                shortfall[item_name] = required_amount - available_amount
        return shortfall

    def can_afford_consumption(
        self,
        inventory: Mapping[str, Amount],
        consumption: Mapping[str, Amount],
    ) -> bool:
        """Check whether an inventory can cover proposed consumable deductions.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            consumption (Mapping[str, Amount]): Proposed consumable deductions.

        Returns:
            bool: Whether all deductions can be covered.

        Note:
            This check does not mutate the supplied inventory.
        """
        return not self.get_consumption_shortfall(inventory, consumption)

    @classmethod
    def _get_inventory_amount(
        cls, inventory: Mapping[str, Amount], item_name: str
    ) -> Amount:
        """Read and validate one amount from an inventory mapping.

        Args:
            inventory (Mapping[str, Amount]): Agent inventory item amounts.
            item_name (str): Name of the item to read.

        Returns:
            Amount: The stored amount, or zero when the item is absent.

        Note:
            Negative inventory values are accepted and naturally act as shortages.
        """
        if not isinstance(inventory, Mapping):
            raise TypeError("inventory must be a mapping.")
        amount = inventory.get(item_name, 0)
        cls._validate_finite_amount(amount, "inventory")
        return amount

    @staticmethod
    def _validate_finite_amount(amount: Amount, field_name: str) -> None:
        """Validate that an amount is a finite numeric value.

        Args:
            amount (Amount): Amount to validate.
            field_name (str): Field name used in error messages.

        Returns:
            None.

        Note:
            Boolean values are rejected even though ``bool`` subclasses ``int``.
        """
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError(f"Amounts in {field_name} must be numeric.")
        if not isfinite(amount):
            raise ValueError(f"Amounts in {field_name} must be finite.")

    @classmethod
    def _validate_nonnegative_amount(cls, amount: Amount, field_name: str) -> None:
        """Validate that an amount is finite and non-negative.

        Args:
            amount (Amount): Amount to validate.
            field_name (str): Field name used in error messages.

        Returns:
            None.

        Note:
            Zero is valid for a proposed consumption amount.
        """
        cls._validate_finite_amount(amount, field_name)
        if amount < 0:
            raise ValueError(f"Amounts in {field_name} must not be negative.")
