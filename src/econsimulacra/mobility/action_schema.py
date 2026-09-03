from __future__ import annotations

import copy
from typing import Any, Iterable


def build_action_schema_with_mobility(
    base_schema: dict[str, Any], mobility_names: Iterable[str]
) -> dict[str, Any]:
    """Build an isolated action schema constrained to available mobility names.

    Args:
        base_schema (dict[str, Any]): Base JSON schema for an agent action.
        mobility_names (Iterable[str]): Mobility names available to one agent.

    Returns:
        dict[str, Any]: A deep-copied schema with a mobility enum.

    Note:
        Walking is used when the supplied collection is empty. The input schema is
        never mutated, which is required when LLM clients are shared concurrently.
    """
    names = list(dict.fromkeys(mobility_names))
    if not names:
        names = ["Walking"]
    schema = copy.deepcopy(base_schema)
    properties = schema.setdefault("properties", {})
    properties["mobility"] = {"type": "string", "enum": names}
    required = schema.setdefault("required", [])
    if "mobility" not in required:
        required.append("mobility")
    return schema
