from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Optional


def get_description(
    path_str: Optional[str],
    default_description: str,
) -> str:
    """Get description from a given file path or use the default description.

    Args:
        path_str (str, optional): The file path to the description text file.
            If None, the default description will be used.
        default_description (str): The default description to use if path_str is None.

    Returns:
        str: The description obtained from the file or the default description.

    Note:
        See also:
            econsimulacra.llm_services.PromptBuilder._get_obs_action_description()
            econsimulacra.llm_services.PersonaBuilder._get_persona_description()
    """
    desc: str
    if path_str is not None:
        desc_path: Path = pathlib.Path(path_str).resolve()
        if not desc_path.exists():
            raise FileNotFoundError(f"Description file not found at: {desc_path}")
        desc = desc_path.read_text(encoding="utf-8")
    else:
        desc = default_description
    return desc
