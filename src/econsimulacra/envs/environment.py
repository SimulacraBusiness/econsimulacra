from typing import Generic, TypeVar

ObsT = TypeVar("ObsT")


class Environment(Generic[ObsT]):
    def __init__(self) -> None:
        pass
