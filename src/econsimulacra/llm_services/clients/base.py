from abc import ABC, abstractmethod
from ..constant import DEFAULT_ACTION_JSON_SCHEMA
import json
import pathlib
from pathlib import Path
from typing import Any
from typing import Optional


class LLMClient(ABC):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config

    @abstractmethod
    async def generate_response(self, prompt: str) -> dict[str, Any]:
        pass

    def _get_json_schema(self, config: dict[str, Any]) -> str:
        """get JSON schema for structured generation from config or use default schema."""
        schema_path_str: Optional[str] = config.get("json_schema_path")
        if schema_path_str is not None:
            schema_path: Path = pathlib.Path(schema_path_str).resolve()
            if not schema_path.exists():
                raise FileNotFoundError(f"JSON schema file not found at: {schema_path}")
            json_schema: dict[str, Any] = json.loads(
                schema_path.read_text(encoding="utf-8")
            )
        else:
            json_schema = DEFAULT_ACTION_JSON_SCHEMA
        json_schema_str: str = json.dumps(json_schema, ensure_ascii=False)
        return json_schema_str
