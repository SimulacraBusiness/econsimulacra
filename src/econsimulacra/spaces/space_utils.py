from collections.abc import Callable, Iterable
from heapq import heappop, heappush
from itertools import count, product
from typing import Optional

Position = tuple[int, ...]


def iter_grid_positions(space_size: Position) -> Iterable[Position]:
    """Yield every position in a grid.

    Args:
        space_size (tuple[int, ...]): The size of each grid dimension.

    Returns:
        Iterable[tuple[int, ...]]: All positions in lexicographic order.

    Note:
        Each dimension must have a positive size. Positions are generated lazily
        so callers may filter them without first materializing the whole grid.
    """
    if not space_size or any(size <= 0 for size in space_size):
        raise ValueError("Every grid dimension must have a positive size.")
    yield from product(*(range(size) for size in space_size))


def chebyshev_distance(pos1: Position, pos2: Position) -> int:
    """Calculate the Chebyshev distance between two positions.

    Args:
        pos1 (tuple[int, ...]): The first position.
        pos2 (tuple[int, ...]): The second position.

    Returns:
        int: The greatest coordinate difference across all dimensions.

    Note:
        Chebyshev distance is an admissible heuristic when one movement may
        change every coordinate by at most one, as in a Moore neighborhood.
    """
    if len(pos1) != len(pos2):
        raise ValueError("Positions must have the same dimension.")
    return max((abs(a - b) for a, b in zip(pos1, pos2)), default=0)


def manhattan_distance(pos1: Position, pos2: Position) -> int:
    """Calculate the Manhattan distance between two positions.

    Args:
        pos1 (tuple[int, ...]): The first position.
        pos2 (tuple[int, ...]): The second position.

    Returns:
        int: The sum of coordinate differences across all dimensions.

    Note:
        Pathfinding uses this distance only to break ties between equal A*
        priorities, preferring diagonal progress compatible with prior movement.
    """
    if len(pos1) != len(pos2):
        raise ValueError("Positions must have the same dimension.")
    return sum(abs(a - b) for a, b in zip(pos1, pos2))


def iter_moore_neighbors(
    pos: Position,
    space_size: Position,
) -> Iterable[Position]:
    """Yield in-bounds Moore-neighborhood positions.

    Args:
        pos (tuple[int, ...]): The center position.
        space_size (tuple[int, ...]): The size of each grid dimension.

    Returns:
        Iterable[tuple[int, ...]]: Neighboring positions in deterministic order.

    Note:
        The center position is excluded. In two dimensions this yields up to
        eight positions, preserving the existing diagonal movement behavior.
    """
    if len(pos) != len(space_size):
        raise ValueError("Position and space size must have the same dimension.")
    for offsets in product((-1, 0, 1), repeat=len(pos)):
        if all(offset == 0 for offset in offsets):
            continue
        neighbor: Position = tuple(
            coordinate + offset for coordinate, offset in zip(pos, offsets)
        )
        if all(
            0 <= coordinate < space_size[dim] for dim, coordinate in enumerate(neighbor)
        ):
            yield neighbor


def _reconstruct_path(
    came_from: dict[Position, Position],
    current: Position,
) -> list[Position]:
    """Reconstruct a path from predecessor links.

    Args:
        came_from (dict[tuple[int, ...], tuple[int, ...]]): Mapping from each
            visited position to its predecessor.
        current (tuple[int, ...]): The final position of the path.

    Returns:
        list[tuple[int, ...]]: The path ordered from start to final position.

    Note:
        The start position has no entry in ``came_from`` and terminates the
        reconstruction loop.
    """
    path: list[Position] = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def find_shortest_path(
    start: Position,
    goal: Position,
    get_neighbors: Callable[[Position], Iterable[Position]],
    can_traverse: Callable[[Position, Position], bool],
) -> Optional[list[Position]]:
    """Find a shortest traversable path using A* search.

    Args:
        start (tuple[int, ...]): The initial position.
        goal (tuple[int, ...]): The destination position.
        get_neighbors (Callable): Function returning neighbors of a position.
        can_traverse (Callable): Function deciding whether an edge may be used.

    Returns:
        Optional[list[tuple[int, ...]]]: A path including both endpoints, or
            ``None`` when no traversable path exists.

    Note:
        Every edge has unit cost. The traversal callback keeps domain-specific
        access rules outside this generic pathfinding utility.
    """
    if len(start) != len(goal):
        raise ValueError("Start and goal positions must have the same dimension.")
    if start == goal:
        return [start]

    sequence = count()
    open_heap: list[tuple[int, int, int, Position]] = []
    start_h: int = chebyshev_distance(start, goal)
    heappush(
        open_heap,
        (start_h, manhattan_distance(start, goal), next(sequence), start),
    )
    came_from: dict[Position, Position] = {}
    best_distance: dict[Position, int] = {start: 0}

    while open_heap:
        _, _, _, current = heappop(open_heap)
        current_distance: int = best_distance[current]
        if current == goal:
            return _reconstruct_path(came_from=came_from, current=current)

        for neighbor in get_neighbors(current):
            if not can_traverse(current, neighbor):
                continue
            candidate_distance: int = current_distance + 1
            if candidate_distance >= best_distance.get(
                neighbor, candidate_distance + 1
            ):
                continue
            came_from[neighbor] = current
            best_distance[neighbor] = candidate_distance
            heuristic: int = chebyshev_distance(neighbor, goal)
            heappush(
                open_heap,
                (
                    candidate_distance + heuristic,
                    manhattan_distance(neighbor, goal),
                    next(sequence),
                    neighbor,
                ),
            )
    return None
