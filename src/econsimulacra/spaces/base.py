from collections import defaultdict
from collections.abc import Iterable, Mapping
from random import Random
from typing import Any, DefaultDict, Optional, Type

from .cell import Cell, CellAccess
from .space_utils import (
    Position,
    find_shortest_path,
    iter_grid_positions,
    iter_moore_neighbors,
)


class GridSpace:
    """Grid space containing agents and extensible cell information."""

    def __init__(
        self,
        config: dict[str, Any],
        registered_classes: list[Type],
        prng: Random,
    ) -> None:
        """Initialize a grid space.

        Args:
            config (dict[str, Any]): Grid configuration. ``gridSize`` is required.
                ``allowInitialColocatedAgents``, ``cellDefaults``, and ``cells``
                are optional.
            registered_classes (list[Type]): Classes registered by the environment
                for custom simulation components.
            prng (Random): Pseudo-random number generator used for placement.

        Returns:
            None.

        Note:
            ``cellDefaults`` and each entry in ``cells`` may contain ``access``
            and ``attrs`` mappings. Only ``traversable`` and ``spawnable`` in
            ``access`` have built-in meanings.
        """
        self.config: dict[str, Any] = config
        self.prng: Random = prng
        self.registered_classes: list[Type] = registered_classes
        self.space_size: Position = self._parse_space_size(config=config)
        allow_initial_colocation: Any = config.get("allowInitialColocatedAgents", True)
        if not isinstance(allow_initial_colocation, bool):
            raise TypeError("'allowInitialColocatedAgents' must be a boolean.")
        self.allow_init_colocated_agents: bool = allow_initial_colocation
        self.default_cell: Cell = self._parse_default_cell(config=config)
        self.pos2cell: dict[Position, Cell] = {}
        self._load_configured_cells(config=config)
        self.agent_id2init_pos: dict[int, Position] = {}
        self.pos2agent_ids: DefaultDict[Position, set[int]] = defaultdict(set)
        self.agent_id2pos: dict[int, Position] = {}

    def _parse_space_size(self, config: Mapping[str, Any]) -> Position:
        """Parse and validate the configured grid size.

        Args:
            config (Mapping[str, Any]): Grid-space configuration.

        Returns:
            tuple[int, ...]: The validated size of every grid dimension.

        Note:
            JSON arrays and Python tuples are both accepted. Boolean values are
            rejected even though ``bool`` is an ``int`` subclass.
        """
        if "gridSize" not in config:
            raise ValueError("GridSpace requires 'gridSize' in config.")
        raw_size: Any = config["gridSize"]
        if not isinstance(raw_size, (list, tuple)) or not raw_size:
            raise TypeError("'gridSize' must be a non-empty list or tuple.")
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size <= 0
            for size in raw_size
        ):
            raise ValueError("Every value in 'gridSize' must be a positive integer.")
        return tuple(raw_size)

    def _parse_default_cell(self, config: Mapping[str, Any]) -> Cell:
        """Parse the default cell configuration.

        Args:
            config (Mapping[str, Any]): Grid-space configuration.

        Returns:
            Cell: The cell inherited by positions without an explicit override.

        Note:
            An omitted ``cellDefaults`` entry produces a traversable and
            spawnable cell with no custom attributes, preserving old behavior.
        """
        raw_defaults: Any = config.get("cellDefaults", {})
        if not isinstance(raw_defaults, Mapping):
            raise TypeError("'cellDefaults' must be a mapping.")
        return Cell.from_config(config=raw_defaults, defaults=Cell())

    def _load_configured_cells(self, config: Mapping[str, Any]) -> None:
        """Load explicit cell overrides from configuration.

        Args:
            config (Mapping[str, Any]): Grid-space configuration.

        Returns:
            None.

        Note:
            ``cells`` is a list of mappings containing a ``pos`` array and
            optional ``access`` and ``attrs`` mappings. Duplicate positions are
            rejected to prevent silently shadowed map data.
        """
        raw_cells: Any = config.get("cells", [])
        if not isinstance(raw_cells, list):
            raise TypeError("'cells' must be a list.")
        configured_positions: set[Position] = set()
        for raw_cell in raw_cells:
            if not isinstance(raw_cell, Mapping):
                raise TypeError("Each entry in 'cells' must be a mapping.")
            raw_pos: Any = raw_cell.get("pos")
            if not isinstance(raw_pos, (list, tuple)):
                raise TypeError("Each configured cell requires a list or tuple 'pos'.")
            pos: Position = tuple(raw_pos)
            self._check_bounds(pos=pos)
            if pos in configured_positions:
                raise ValueError(f"Cell position {pos} is configured more than once.")
            configured_positions.add(pos)
            cell: Cell = Cell.from_config(config=raw_cell, defaults=self.default_cell)
            self._store_cell(pos=pos, cell=cell)

    def _store_cell(self, pos: Position, cell: Cell) -> None:
        """Store a cell override only when it differs from the default.

        Args:
            pos (tuple[int, ...]): Position whose cell information is updated.
            cell (Cell): Complete cell information to store.

        Returns:
            None.

        Note:
            Removing default-valued entries keeps the cell map sparse.
        """
        if cell == self.default_cell:
            self.pos2cell.pop(pos, None)
        else:
            self.pos2cell[pos] = Cell(access=cell.access, attrs=dict(cell.attrs))

    def get_space_size(self) -> Position:
        """Get the shape of the grid space.

        Args:
            None.

        Returns:
            tuple[int, ...]: The dimensions of the grid space.

        Note:
            The returned tuple is immutable.
        """
        return self.space_size

    def _check_bounds(self, pos: Position) -> None:
        """Check whether a position is within the grid bounds.

        Args:
            pos (tuple[int, ...]): Position to validate.

        Returns:
            None.

        Note:
            ``ValueError`` is raised for a dimensional mismatch or an out-of-range
            coordinate.
        """
        if len(pos) != len(self.space_size):
            raise ValueError(
                f"Position {pos} has different dimension than space size {self.space_size}."
            )
        for dim, coordinate in enumerate(pos):
            if not isinstance(coordinate, int) or isinstance(coordinate, bool):
                raise ValueError(
                    f"Coordinate {coordinate} at dimension {dim} is not an integer."
                )
            if coordinate < 0 or coordinate >= self.space_size[dim]:
                raise ValueError(
                    f"Coordinate {coordinate} at dimension {dim} is out of bounds "
                    f"for space size {self.space_size}."
                )

    def get_cell(self, pos: Position) -> Cell:
        """Get cell information at a position.

        Args:
            pos (tuple[int, ...]): Position whose cell information is requested.

        Returns:
            Cell: A safe copy of the effective cell information.

        Note:
            Positions without explicit overrides inherit ``default_cell``.
        """
        self._check_bounds(pos=pos)
        return self.pos2cell.get(pos, self.default_cell).copy()

    def get_cell_attrs(self, pos: Position) -> dict[str, Any]:
        """Get arbitrary attributes at a position.

        Args:
            pos (tuple[int, ...]): Position whose attributes are requested.

        Returns:
            dict[str, Any]: A shallow copy of the cell attribute mapping.

        Note:
            Mutating the returned dictionary does not mutate the grid. Use
            ``update_cell_attrs`` for runtime interventions.
        """
        return dict(self.get_cell(pos=pos).attrs)

    def update_cell_access(
        self,
        pos: Position,
        traversable: Optional[bool] = None,
        spawnable: Optional[bool] = None,
    ) -> None:
        """Update built-in access controls at a position.

        Args:
            pos (tuple[int, ...]): Position to update.
            traversable (Optional[bool]): New traversal permission, or ``None``
                to keep the current value.
            spawnable (Optional[bool]): New initial-placement permission, or
                ``None`` to keep the current value.

        Returns:
            None.

        Note:
            This method supports interventions such as temporary road closures.
            Existing agents at the position are not displaced.
        """
        current: Cell = self.get_cell(pos=pos)
        access_updates: dict[str, Any] = {}
        if traversable is not None:
            access_updates["traversable"] = traversable
        if spawnable is not None:
            access_updates["spawnable"] = spawnable
        access: CellAccess = CellAccess.from_config(
            config=access_updates,
            defaults=current.access,
        )
        self._store_cell(pos=pos, cell=Cell(access=access, attrs=current.attrs))

    def update_cell_attrs(self, pos: Position, updates: Mapping[str, Any]) -> None:
        """Merge arbitrary attributes into a cell.

        Args:
            pos (tuple[int, ...]): Position to update.
            updates (Mapping[str, Any]): Attribute names and replacement values.

        Returns:
            None.

        Note:
            Attribute values are deliberately not interpreted by ``GridSpace``.
        """
        if not isinstance(updates, Mapping):
            raise TypeError("Cell attribute updates must be a mapping.")
        current: Cell = self.get_cell(pos=pos)
        attrs: dict[str, Any] = dict(current.attrs)
        attrs.update(updates)
        self._store_cell(pos=pos, cell=Cell(access=current.access, attrs=attrs))

    def can_spawn(self, agent_id: int, pos: Position) -> bool:
        """Determine whether an agent may initially spawn at a position.

        Args:
            agent_id (int): ID of the agent being placed.
            pos (tuple[int, ...]): Candidate initial position.

        Returns:
            bool: Whether the cell permits initial placement.

        Note:
            The base implementation only checks ``CellAccess.spawnable``.
            Subclasses may use ``agent_id`` and arbitrary attributes to add rules.
        """
        del agent_id
        return self.get_cell(pos=pos).access.spawnable

    def can_enter(
        self,
        agent_id: Optional[int],
        current_pos: Position,
        new_pos: Position,
    ) -> bool:
        """Determine whether an agent may enter a position.

        Args:
            agent_id (Optional[int]): ID of the moving agent when available.
            current_pos (tuple[int, ...]): Position from which the agent moves.
            new_pos (tuple[int, ...]): Candidate destination position.

        Returns:
            bool: Whether the candidate cell permits traversal.

        Note:
            The base implementation only checks ``CellAccess.traversable``.
            The other arguments are extension points for agent-specific rules.
        """
        del agent_id, current_pos
        return self.get_cell(pos=new_pos).access.traversable

    def iter_neighbors(
        self,
        agent_id: Optional[int],
        pos: Position,
    ) -> Iterable[Position]:
        """Yield positions reachable by one geometric movement.

        Args:
            agent_id (Optional[int]): ID of the moving agent when available.
            pos (tuple[int, ...]): Position whose neighbors are requested.

        Returns:
            Iterable[tuple[int, ...]]: In-bounds neighboring positions.

        Note:
            The base implementation uses a Moore neighborhood to preserve the
            existing diagonal movement behavior. Subclasses may override it for
            mobility-specific neighborhoods.
        """
        del agent_id
        self._check_bounds(pos=pos)
        return iter_moore_neighbors(pos=pos, space_size=self.space_size)

    def get_spawnable_positions(self, agent_id: int) -> list[Position]:
        """Collect valid initial-position candidates for an agent.

        Args:
            agent_id (int): ID of the agent being placed.

        Returns:
            list[tuple[int, ...]]: Spawnable cells satisfying colocation settings.

        Note:
            When initial colocation is disabled, occupied positions are excluded.
        """
        positions: list[Position] = []
        for pos in iter_grid_positions(space_size=self.space_size):
            if not self.can_spawn(agent_id=agent_id, pos=pos):
                continue
            if not self.allow_init_colocated_agents and self.pos2agent_ids.get(pos):
                continue
            positions.append(pos)
        return positions

    def get_pos(self, agent_id: int) -> Position:
        """Get an agent's current position.

        Args:
            agent_id (int): ID of the agent whose position is requested.

        Returns:
            tuple[int, ...]: The agent's current position.

        Note:
            ``ValueError`` is raised when the agent is not in this space.
        """
        if agent_id not in self.agent_id2pos:
            raise ValueError(f"Agent ID {agent_id} not found in grid space.")
        return self.agent_id2pos[agent_id]

    def get_agents(self, pos: Position) -> set[int]:
        """Get agent IDs at a position.

        Args:
            pos (tuple[int, ...]): Position to inspect.

        Returns:
            set[int]: A copy of the agent IDs at the position.

        Note:
            Returning a copy prevents callers from mutating spatial indexes.
        """
        self._check_bounds(pos=pos)
        return set(self.pos2agent_ids.get(pos, set()))

    def place_agent(self, agent_id: int, pos: Optional[Position]) -> None:
        """Place an agent at an explicit or randomly selected initial position.

        Args:
            agent_id (int): ID of the agent to place.
            pos (Optional[tuple[int, ...]]): Explicit position, or ``None`` to
                choose uniformly from currently valid spawnable candidates.

        Returns:
            None.

        Note:
            Both explicit and random placement respect ``spawnable`` and
            ``allowInitialColocatedAgents``.
        """
        if pos is None:
            candidates: list[Position] = self.get_spawnable_positions(agent_id=agent_id)
            if not candidates:
                raise ValueError(
                    f"Could not find a valid spawnable position for agent ID {agent_id}."
                )
            pos = self.prng.choice(candidates)
        self._check_placement(agent_id=agent_id, pos=pos)
        self.agent_id2pos[agent_id] = pos
        self.pos2agent_ids[pos].add(agent_id)
        self.agent_id2init_pos[agent_id] = pos

    def _check_placement(self, agent_id: int, pos: Position) -> None:
        """Validate an initial agent placement.

        Args:
            agent_id (int): ID of the agent to place.
            pos (tuple[int, ...]): Proposed initial position.

        Returns:
            None.

        Note:
            This check applies only to initial placement. Movement intentionally
            permits colocated agents regardless of the initial setting.
        """
        self._check_bounds(pos=pos)
        if agent_id in self.agent_id2pos:
            raise ValueError(
                f"Agent ID {agent_id} is already placed in the grid space."
            )
        if not self.can_spawn(agent_id=agent_id, pos=pos):
            raise ValueError(
                f"Position {pos} is not spawnable for agent ID {agent_id}."
            )
        if not self.allow_init_colocated_agents and self.pos2agent_ids.get(pos):
            raise ValueError(
                f"Position {pos} is already occupied by other agents, and "
                "initially colocated agents are not allowed."
            )

    def get_colocated_agents(self, agent_id: int) -> set[int]:
        """Get other agents sharing an agent's current position.

        Args:
            agent_id (int): ID of the reference agent.

        Returns:
            set[int]: IDs of other agents at the same position.

        Note:
            The reference agent is excluded from the returned set.
        """
        pos: Position = self.get_pos(agent_id=agent_id)
        return set(self.pos2agent_ids.get(pos, set())) - {agent_id}

    def get_near_agents(self, center_pos: Position, max_distance: int = 1) -> set[int]:
        """Get agent IDs within a Manhattan-distance radius.

        Args:
            center_pos (tuple[int, ...]): Center of the query.
            max_distance (int): Inclusive maximum Manhattan distance.

        Returns:
            set[int]: IDs of agents within the specified radius.

        Note:
            For compatibility, the center must currently contain at least one
            agent. ``max_distance`` must be non-negative.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")
        self._check_bounds(pos=center_pos)
        if center_pos not in self.pos2agent_ids:
            raise ValueError(f"Position {center_pos} not found in grid space.")
        near_agent_ids: set[int] = set()
        for other_agent_id, other_pos in self.agent_id2pos.items():
            distance: int = sum(
                abs(center_coordinate - other_coordinate)
                for center_coordinate, other_coordinate in zip(center_pos, other_pos)
            )
            if distance <= max_distance:
                near_agent_ids.add(other_agent_id)
        return near_agent_ids

    def get_nearby_info(
        self, agent_id: int, max_distance: int = 1
    ) -> dict[Position, Cell]:
        """Get a mapping of nearby positions to their cell information.

        Args:
            agent_id (int): ID of the reference agent.
            max_distance (int): Inclusive maximum Manhattan distance.

        Returns:
            dict[Position, Cell]: Mapping from positions to cell information.

        Note:
            The reference agent's position is included. ``max_distance`` must be
            non-negative.
        """
        if max_distance < 0:
            raise ValueError("max_distance must be non-negative.")
        center_pos: Position = self.get_pos(agent_id=agent_id)
        nearby_info: dict[Position, Cell] = {}
        for pos in iter_grid_positions(space_size=self.space_size):
            distance: int = sum(
                abs(center_coordinate - other_coordinate)
                for center_coordinate, other_coordinate in zip(center_pos, pos)
            )
            if distance <= max_distance:
                nearby_info[pos] = self.get_cell(pos=pos)
        return nearby_info

    def remove_agent(self, agent_id: int) -> None:
        """Remove an agent from the grid space.

        Args:
            agent_id (int): ID of the agent to remove.

        Returns:
            None.

        Note:
            The recorded initial position is retained for compatibility with
            movement-history consumers.
        """
        pos: Position = self.get_pos(agent_id=agent_id)
        del self.agent_id2pos[agent_id]
        self.pos2agent_ids[pos].remove(agent_id)
        if not self.pos2agent_ids[pos]:
            del self.pos2agent_ids[pos]

    def move_agent(self, agent_id: int, new_pos: Position) -> None:
        """Move an agent directly to a traversable position.

        Args:
            agent_id (int): ID of the agent to move.
            new_pos (tuple[int, ...]): Position to enter.

        Returns:
            None.

        Note:
            This low-level operation validates bounds and traversal access but
            does not require adjacency and does not reject colocated agents.
        """
        old_pos: Position = self.get_pos(agent_id=agent_id)
        self._check_bounds(pos=new_pos)
        if not self.can_enter(
            agent_id=agent_id,
            current_pos=old_pos,
            new_pos=new_pos,
        ):
            raise ValueError(
                f"Position {new_pos} is not traversable for agent ID {agent_id}."
            )
        if old_pos == new_pos:
            return
        self.pos2agent_ids[old_pos].remove(agent_id)
        if not self.pos2agent_ids[old_pos]:
            del self.pos2agent_ids[old_pos]
        self.agent_id2pos[agent_id] = new_pos
        self.pos2agent_ids[new_pos].add(agent_id)

    def move_many_agents(self, agent_id2new_pos: dict[int, Position]) -> None:
        """Move multiple agents after validating every target position.

        Args:
            agent_id2new_pos (dict[int, tuple[int, ...]]): Mapping from agent IDs
                to direct movement targets.

        Returns:
            None.

        Note:
            Validation occurs before mutation to avoid partial updates. Colocation
            is intentionally allowed during movement.
        """
        for agent_id, new_pos in agent_id2new_pos.items():
            old_pos: Position = self.get_pos(agent_id=agent_id)
            self._check_bounds(pos=new_pos)
            if not self.can_enter(
                agent_id=agent_id,
                current_pos=old_pos,
                new_pos=new_pos,
            ):
                raise ValueError(
                    f"Position {new_pos} is not traversable for agent ID {agent_id}."
                )
        for agent_id, new_pos in agent_id2new_pos.items():
            self.move_agent(agent_id=agent_id, new_pos=new_pos)

    def calc_next_pos(
        self,
        current_pos: Position,
        destination_pos: Position,
        velocity: int = 1,
        agent_id: Optional[int] = None,
    ) -> Optional[Position]:
        """Calculate the next position along a shortest traversable path.

        Args:
            current_pos (tuple[int, ...]): Current agent position.
            destination_pos (tuple[int, ...]): Requested destination position.
            velocity (int): Maximum number of path edges traversed this step.
            agent_id (Optional[int]): ID used by custom access and neighbor rules.

        Returns:
            Optional[tuple[int, ...]]: The position reached this step, or ``None``
                when no traversable path exists.

        Note:
            ``velocity`` must be a positive integer. Intermediate cells remain
            part of the path, so higher velocity cannot skip over obstacles.
        """
        self._check_bounds(pos=current_pos)
        self._check_bounds(pos=destination_pos)
        if not isinstance(velocity, int) or isinstance(velocity, bool) or velocity <= 0:
            raise ValueError("velocity must be a positive integer.")
        path: Optional[list[Position]] = find_shortest_path(
            start=current_pos,
            goal=destination_pos,
            get_neighbors=lambda pos: self.iter_neighbors(agent_id=agent_id, pos=pos),
            can_traverse=lambda old_pos, new_pos: self.can_enter(
                agent_id=agent_id,
                current_pos=old_pos,
                new_pos=new_pos,
            ),
        )
        if path is None:
            return None
        return path[min(velocity, len(path) - 1)]
