import json
import pathlib
from pathlib import Path
from typing import Optional

from .base import Log, Logger


class DictLogger(Logger):
    """Logger implementation that records processed logs in memory and optionally saves them.

    Processed logs are serialized to dictionaries via Log.to_dict() and stored in self.logs.
    If txt_save_path is provided, save() writes the records in JSON Lines format to the specified path.
    """

    def __init__(self, txt_save_path: Optional[Path] = None) -> None:
        """Initialization.

        Args:
            txt_save_path (Optional[Path]): Output path for JSON Lines log export.
                If None, save() does nothing. Defaults to None.
        """
        super().__init__()
        self.logs: list[dict] = []
        self.txt_save_path: Optional[Path] = txt_save_path

    def _process_log_default(self, log: Log) -> None:
        """Process a log by converting it to a dict and storing it in memory."""
        self.logs.append(log.to_dict())

    def save(self) -> None:
        """Save collected logs to txt_save_path in JSON Lines format.

        Notes:
            - The parent directory is created if it does not exist.
        """
        if self.txt_save_path is not None:
            pathlib.Path(self.txt_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.txt_save_path, "w") as f:
                for log_dic in self.logs:
                    f.write(json.dumps(log_dic, ensure_ascii=False) + "\n")
