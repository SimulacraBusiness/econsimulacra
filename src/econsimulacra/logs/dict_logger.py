from .base import Log
from .base import Logger


class DictLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.logs: list[dict] = []

    def _process_log_default(self, log: Log) -> None:
        self.logs.append(log.to_dict())