from .base import Log
from .base import Logger
import json
from pathlib import Path
import pathlib
from typing import Optional



class DictLogger(Logger):
    def __init__(self, txt_save_path: Optional[Path] = None) -> None:
        super().__init__()
        self.logs: list[dict] = []
        self.txt_save_path: Optional[Path] = txt_save_path

    def _process_log_default(self, log: Log) -> None:
        self.logs.append(log.to_dict())

    def save(self) -> None:
        if self.txt_save_path is not None:
            pathlib.Path(self.txt_save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.txt_save_path, "w") as f:
                for log_dic in self.logs:
                    f.write(f"{json.dumps(log_dic)}\n")
