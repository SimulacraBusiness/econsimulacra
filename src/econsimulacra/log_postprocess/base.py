from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

LogProcessorFunc = Callable[[dict[str, Any]], dict[str, Any]]


class LogPostProcessor:
    """Post-process JSONL simulation logs.

    This class applies registered processors to log records
    based on the log 'type'. Basic usage:

    >>> processor = LogPostProcessor()
    >>> @processor.register("tweet")
    >>> def process_tweet(log: dict[str, Any]) -> dict[str, Any]:
    >>>     # Custom processing for tweet logs
    >>>     return log_record
    >>> processor.process_file("log.txt", "processed_log.txt")
    """

    def __init__(self) -> None:
        self._processors: dict[str, list[LogProcessorFunc]] = {}

    def register(self, log_type: str):
        """Decorator to register a log processor for a specific log type.

        Args:
            log_type: The 'type' field in logs that this processor will handle.
                Ex) "agent_generation", "tweet", "move", "order", etc.
                See also: econsimulacra.logs
        """

        def decorator(func: LogProcessorFunc) -> LogProcessorFunc:
            self._processors.setdefault(log_type, []).append(func)
            return func

        return decorator

    def add_processor(self, log_type: str, processor_func: LogProcessorFunc) -> None:
        """Add a log processor function for a specific log type.

        Args:
            log_type: The 'type' field in logs that this processor will handle.
            processor_func: A function that takes a log record dict and returns
                a processed log record dict.

        Note:
            This method is an alternative to using the @register decorator. Basic usage:

            >>> processor = LogPostProcessor()
            >>> def process_tweet(log: dict[str, Any]) -> dict[str, Any]:
            >>>     # Custom processing for tweet logs
            >>>     return log_record
            >>> processor.add_processor("tweet", process_tweet)
        """
        self._processors.setdefault(log_type, []).append(processor_func)

    def process_log(self, log: dict[str, Any]) -> dict[str, Any]:
        """Process a single log record using registered processors.

        Args:
            log: A log record dict that must contain a 'type' field.

        Returns:
            The processed log record dict after applying all relevant processors.
        """
        log_type = log.get("type")
        if not log_type:
            raise ValueError("Log record must have a 'type' field.")
        for processor in self._processors.get(log_type, []):
            log = processor(log)
        return log

    def process_file(self, input_file: str | Path, output_file: str | Path) -> None:
        """Process a JSONL log file and write the processed logs to a new file.

        Args:
            input_file: Path to the input JSONL log file.
            output_file: Path to the output JSONL log file where processed logs will be saved.
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        with (
            input_path.open("r", encoding="utf-8") as infile,
            output_path.open("w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                log = json.loads(line)
                processed_log = self.process_log(log)
                outfile.write(json.dumps(processed_log, ensure_ascii=False) + "\n")
