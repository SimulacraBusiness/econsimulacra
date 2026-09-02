from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class CellAccess:
    """Access controls interpreted by ``GridSpace``."""

    traversable: bool = True
    spawnable: bool = True

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        defaults: "CellAccess",
    ) -> "CellAccess":
        """Create cell access controls from a partial configuration.

        Args:
            config (Mapping[str, Any]): Partial access configuration.
            defaults (CellAccess): Values used for omitted configuration keys.

        Returns:
            CellAccess: The validated access controls.

        Note:
            Only ``traversable`` and ``spawnable`` are interpreted as built-in
            access controls. Both values must be booleans when specified.
        """
        traversable: Any = config.get("traversable", defaults.traversable)
        spawnable: Any = config.get("spawnable", defaults.spawnable)
        if not isinstance(traversable, bool):
            raise TypeError("Cell access 'traversable' must be a boolean.")
        if not isinstance(spawnable, bool):
            raise TypeError("Cell access 'spawnable' must be a boolean.")
        return cls(traversable=traversable, spawnable=spawnable)


@dataclass(frozen=True)
class Cell:
    """A grid cell containing access controls and arbitrary attributes."""

    access: CellAccess = field(default_factory=CellAccess)
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        defaults: "Cell",
    ) -> "Cell":
        """Create a cell from a partial configuration.

        Args:
            config (Mapping[str, Any]): Partial cell configuration containing
                optional ``access`` and ``attrs`` mappings.
            defaults (Cell): Cell whose values are inherited when omitted.

        Returns:
            Cell: The validated cell with merged arbitrary attributes.

        Note:
            Attribute names and values are not interpreted, allowing simulations
            to attach domain-specific information without changing ``GridSpace``.
        """
        access_config: Any = config.get("access", {})
        attrs_config: Any = config.get("attrs", {})
        if not isinstance(access_config, Mapping):
            raise TypeError("Cell 'access' must be a mapping.")
        if not isinstance(attrs_config, Mapping):
            raise TypeError("Cell 'attrs' must be a mapping.")
        attrs: dict[str, Any] = dict(defaults.attrs)
        attrs.update(attrs_config)
        return cls(
            access=CellAccess.from_config(access_config, defaults.access),
            attrs=attrs,
        )

    def copy(self) -> "Cell":
        """Return a copy whose attributes can be safely exposed to callers.

        Args:
            None.

        Returns:
            Cell: A cell containing a shallow copy of the attribute mapping.

        Note:
            Attribute values themselves are not deep-copied so that arbitrary
            runtime objects may be stored without imposing copy semantics.
        """
        return Cell(access=self.access, attrs=dict(self.attrs))
