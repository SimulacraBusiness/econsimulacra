"""Tests for inventory-aware mobility modes."""

from random import Random

import pytest

from econsimulacra.mobility import (
    MobilityManager,
    MobilityMode,
    MovementState,
    build_action_schema_with_mobility,
)


@pytest.fixture
def mobility_config() -> dict:
    """Return a configuration with walking and two vehicle modes.

    Args:
        None.

    Returns:
        dict: Mobility manager configuration for tests.

    Note:
        The electric car uses two consumables to exercise limiting-resource logic.
    """
    return {
        "type": "MobilityManager",
        "defaultMode": "Walking",
        "modes": {
            "Walking": {"velocity": 1},
            "ElectricCar": {
                "velocity": 10,
                "itemName": "ElectricCar",
                "requiredItems": {"DriverLicense": 1},
                "consumptionPerCell": {
                    "Electricity": 0.2,
                    "Coolant": 0.1,
                },
            },
            "GasCar": {
                "velocity": 8,
                "itemName": "GasCar",
                "consumptionPerCell": {"Gasoline": 0.5},
            },
        },
    }


def test_constructor_is_compatible_with_service_provider_arguments(
    mobility_config: dict,
) -> None:
    """Test that Environment-style constructor arguments are accepted.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Environment can later instantiate this class without an adapter.
    """
    prng = Random(1)
    registered_classes = [MobilityMode]

    manager = MobilityManager(mobility_config, prng, registered_classes)

    assert manager.prng is prng
    assert manager.registered_classes == registered_classes
    assert manager.config == mobility_config


def test_default_configuration_provides_walking() -> None:
    """Test the fallback walking mode.

    Args:
        None.

    Returns:
        None.

    Note:
        This permits gradual adoption before explicit mobility configuration exists.
    """
    manager = MobilityManager({})

    assert manager.get_default_mode().name == "Walking"
    assert manager.get_default_mode().velocity == 1
    assert manager.get_available_modes({}) == [manager.get_default_mode()]
    assert manager.get_effective_velocity({}, "Walking") == 1


def test_mode_configuration_is_copied() -> None:
    """Test that later mutations of source configuration do not affect modes.

    Args:
        None.

    Returns:
        None.

    Note:
        Mobility modes behave as immutable configuration values.
    """
    required_items = {"ElectricCar": 1}
    config = {
        "defaultMode": "ElectricCar",
        "modes": {
            "ElectricCar": {
                "velocity": 10,
                "requiredItems": required_items,
            }
        },
    }
    manager = MobilityManager(config)

    required_items["ElectricCar"] = 2

    assert manager.get_mode("ElectricCar").required_items == {"ElectricCar": 1}


def test_missing_vehicle_makes_mode_unavailable(mobility_config: dict) -> None:
    """Test that a durable vehicle is required for its mobility mode.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Consumables alone do not grant ownership of a vehicle.
    """
    manager = MobilityManager(mobility_config)
    inventory = {
        "DriverLicense": 1,
        "Electricity": 100,
        "Coolant": 100,
    }

    assert manager.get_missing_required_items(inventory, "ElectricCar") == {
        "ElectricCar": 1
    }
    assert not manager.is_mode_unlocked(inventory, "ElectricCar")
    assert manager.get_effective_velocity(inventory, "ElectricCar") == 0
    assert [mode.name for mode in manager.get_available_modes(inventory)] == ["Walking"]


def test_multiple_owned_modes_are_available(mobility_config: dict) -> None:
    """Test simultaneous ownership of several mobility modes.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Availability concerns ownership and does not select a mode automatically.
    """
    manager = MobilityManager(mobility_config)
    inventory = {"ElectricCar": 1, "DriverLicense": 1, "GasCar": 1}

    assert [mode.name for mode in manager.get_unlocked_modes(inventory)] == [
        "Walking",
        "ElectricCar",
        "GasCar",
    ]


def test_effective_velocity_is_capped_by_configured_velocity(
    mobility_config: dict,
) -> None:
    """Test that plentiful resources do not exceed a mode's velocity.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Velocity represents the maximum cells traversable in one simulation step.
    """
    manager = MobilityManager(mobility_config)
    inventory = {
        "ElectricCar": 1,
        "DriverLicense": 1,
        "Electricity": 100,
        "Coolant": 100,
    }

    assert manager.get_effective_velocity(inventory, "ElectricCar") == 10


def test_effective_velocity_uses_limiting_consumable(mobility_config: dict) -> None:
    """Test that the scarcest consumable limits travel distance.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        A small epsilon prevents exact decimal ratios from rounding down spuriously.
    """
    manager = MobilityManager(mobility_config)
    inventory = {
        "ElectricCar": 1,
        "DriverLicense": 1,
        "Electricity": 0.7,
        "Coolant": 0.2,
    }

    assert manager.get_effective_velocity(inventory, "ElectricCar") == 2


def test_usable_modes_exclude_owned_vehicle_without_energy(
    mobility_config: dict,
) -> None:
    """Test the distinction between owned and currently usable modes.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Walking remains usable because it consumes no inventory resource.
    """
    manager = MobilityManager(mobility_config)
    inventory = {"ElectricCar": 1, "DriverLicense": 1}

    assert [mode.name for mode in manager.get_unlocked_modes(inventory)] == [
        "Walking",
        "ElectricCar",
    ]
    assert [mode.name for mode in manager.get_usable_modes(inventory)] == ["Walking"]


def test_consumption_depends_on_actual_distance(mobility_config: dict) -> None:
    """Test consumable calculation after partial movement.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Durable required items are not consumed by movement.
    """
    manager = MobilityManager(mobility_config)

    consumption = manager.calculate_consumption("ElectricCar", 3)

    assert consumption == pytest.approx({"Electricity": 0.6, "Coolant": 0.3})
    assert "ElectricCar" not in consumption


def test_consumption_affordability_and_shortfall(mobility_config: dict) -> None:
    """Test affordability checks without inventory mutation.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        The manager reports deductions but leaves mutation to a later integration layer.
    """
    manager = MobilityManager(mobility_config)
    inventory = {"Electricity": 0.5, "Coolant": 1.0}
    consumption = manager.calculate_consumption("ElectricCar", 3)

    assert manager.get_consumption_shortfall(inventory, consumption) == pytest.approx(
        {"Electricity": 0.1}
    )
    assert not manager.can_afford_consumption(inventory, consumption)
    assert inventory == {"Electricity": 0.5, "Coolant": 1.0}


@pytest.mark.parametrize("velocity", [0, -1, 1.5, True])
def test_invalid_velocity_is_rejected(velocity: object) -> None:
    """Test rejection of invalid velocity values.

    Args:
        velocity (object): Invalid parameterized velocity.

    Returns:
        None.

    Note:
        A positive integer velocity keeps movement measured in grid cells.
    """
    with pytest.raises(ValueError, match="positive integer"):
        MobilityManager(
            {
                "defaultMode": "Invalid",
                "modes": {"Invalid": {"velocity": velocity}},
            }
        )


@pytest.mark.parametrize(
    ("field_name", "amount"),
    [
        ("requiredItems", 0),
        ("requiredItems", -1),
        ("consumptionPerCell", 0),
        ("consumptionPerCell", -0.1),
        ("consumptionPerCell", True),
    ],
)
def test_invalid_mode_amount_is_rejected(field_name: str, amount: object) -> None:
    """Test rejection of invalid requirement and consumption amounts.

    Args:
        field_name (str): Mode configuration field under test.
        amount (object): Invalid configured amount.

    Returns:
        None.

    Note:
        Configured amounts must be finite positive numbers.
    """
    with pytest.raises((TypeError, ValueError)):
        MobilityManager(
            {
                "defaultMode": "Invalid",
                "modes": {"Invalid": {"velocity": 1, field_name: {"Item": amount}}},
            }
        )


def test_invalid_manager_configuration_is_rejected() -> None:
    """Test malformed mode mappings and absent default modes.

    Args:
        None.

    Returns:
        None.

    Note:
        Configuration errors are detected before the simulation starts.
    """
    with pytest.raises(TypeError, match="modes must be a mapping"):
        MobilityManager({"modes": []})
    with pytest.raises(ValueError, match="is not configured"):
        MobilityManager(
            {
                "defaultMode": "Missing",
                "modes": {"Car": {"velocity": 2}},
            }
        )


def test_unknown_mode_is_rejected(mobility_config: dict) -> None:
    """Test lookup of an unconfigured mode.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Explicit failure prevents silently falling back to an unintended mode.
    """
    manager = MobilityManager(mobility_config)

    with pytest.raises(ValueError, match="Unknown mobility mode"):
        manager.get_mode("Bicycle")


@pytest.mark.parametrize("moved_cells", [-1, 11])
def test_invalid_movement_distance_is_rejected(
    mobility_config: dict, moved_cells: int
) -> None:
    """Test invalid actual movement distances.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.
        moved_cells (int): Invalid movement distance.

    Returns:
        None.

    Note:
        Consumption is calculated for one step and cannot exceed mode velocity.
    """
    manager = MobilityManager(mobility_config)

    with pytest.raises(ValueError):
        manager.calculate_consumption("ElectricCar", moved_cells)


def test_non_integer_movement_distance_is_rejected(mobility_config: dict) -> None:
    """Test rejection of fractional and boolean movement distances.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Grid travel distance is represented as a cell count.
    """
    manager = MobilityManager(mobility_config)

    with pytest.raises(TypeError):
        manager.calculate_consumption("ElectricCar", 0.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        manager.calculate_consumption("ElectricCar", True)


def test_mobility_item_and_additional_requirements_unlock_mode(
    mobility_config: dict,
) -> None:
    """Test that both the mobility Item and additional Items are required.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Consumables affect usability but do not replace ownership requirements.
    """
    manager = MobilityManager(mobility_config)

    assert not manager.is_mode_unlocked({"ElectricCar": 1}, "ElectricCar")
    assert manager.is_mode_unlocked(
        {"ElectricCar": 1, "DriverLicense": 1}, "ElectricCar"
    )
    assert manager.can_use_mode(
        {
            "ElectricCar": 1,
            "DriverLicense": 1,
            "Electricity": 0.2,
            "Coolant": 0.1,
        },
        "ElectricCar",
    )


def test_validate_item_names_checks_all_inventory_references(
    mobility_config: dict,
) -> None:
    """Test delayed validation against Environment Item registration.

    Args:
        mobility_config (dict): Mobility manager configuration fixture.

    Returns:
        None.

    Note:
        Walking has no itemName and therefore needs no registered Item.
    """
    manager = MobilityManager(mobility_config)
    all_items = {
        "ElectricCar",
        "DriverLicense",
        "Electricity",
        "Coolant",
        "GasCar",
        "Gasoline",
    }

    manager.validate_item_names(all_items)
    with pytest.raises(ValueError, match="Gasoline"):
        manager.validate_item_names(all_items - {"Gasoline"})


def test_movement_state_accepts_only_active_valid_journeys() -> None:
    """Test MovementState invariants.

    Args:
        None.

    Returns:
        None.

    Note:
        Inactive state is represented by ``None`` in Environment.
    """
    state = MovementState(True, (1, 2), "ElectricCar")

    assert state.destination == (1, 2)
    with pytest.raises(ValueError, match="active journey"):
        MovementState(False, (1, 2), "Walking")


def test_action_schema_is_dynamic_and_does_not_mutate_base() -> None:
    """Test per-agent mobility enums and concurrent-schema isolation.

    Args:
        None.

    Returns:
        None.

    Note:
        Each request receives a deep copy because LLM clients are shared.
    """
    base_schema = {"type": "object", "properties": {}, "required": []}

    car_schema = build_action_schema_with_mobility(
        base_schema, ["Walking", "ElectricCar"]
    )
    walking_schema = build_action_schema_with_mobility(base_schema, ["Walking"])

    assert base_schema == {"type": "object", "properties": {}, "required": []}
    assert car_schema["properties"]["mobility"]["enum"] == [
        "Walking",
        "ElectricCar",
    ]
    assert walking_schema["properties"]["mobility"]["enum"] == ["Walking"]
