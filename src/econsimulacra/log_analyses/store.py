from collections import defaultdict
from typing import TypeVar

from .records import BaseRecord

T = TypeVar("T", bound=BaseRecord)


class RecordStore:
    """A record store that organizes records
    by type and agent ID for efficient retrieval. Basic usage:

    >>> records = load_from_file("path/to/log.txt")
    >>> store = RecordStore(records)
    >>> move_records = store.get_by_type("move")
    >>> agent_1_records = store.get_by_agent(1)
    """

    def __init__(self, records: list[BaseRecord]) -> None:
        """Initialization.

        Args:
            records (list[BaseRecord]): A list of records to store.
        """
        self.records = records
        self.by_type: dict[str, list[BaseRecord]] = defaultdict(list)
        self.by_agent: dict[int, list[BaseRecord]] = defaultdict(list)

        for record in records:
            self.by_type[record.type].append(record)

            agent_id = getattr(record, "agent_id", None)
            if agent_id is not None:
                self.by_agent[int(agent_id)].append(record)

    def get_by_type(self, record_type: str) -> list[BaseRecord]:
        """Get records by type.

        Args:
            record_type (str): The type of records to retrieve. Example: "move", "order", etc.

        Returns:
            list[BaseRecord]: A list of records matching the specified type.
        """
        return self.by_type.get(record_type, [])

    def get_by_agent(self, agent_id: int) -> list[BaseRecord]:
        """Get records by agent ID.

        Args:
            agent_id (int): The ID of the agent whose records to retrieve.

        Returns:
            list[BaseRecord]: A list of records associated with the specified agent ID.
        """
        return self.by_agent.get(agent_id, [])

    def typed(self, cls: type[T]) -> list[T]:
        """Get records of a specific class.

        Args:
            cls (type[T]): The class of records to retrieve. Example: `MoveRecord`, `OrderRecord`, etc.

        Returns:
            list[T]: A list of records matching the specified class.
        """
        return [e for e in self.records if isinstance(e, cls)]
