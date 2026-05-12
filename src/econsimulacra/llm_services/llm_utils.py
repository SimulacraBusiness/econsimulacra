from __future__ import annotations

import json
import pathlib
from pathlib import Path
from typing import Optional


def get_description(
    path_str: Optional[str],
    default_description: dict[str, str] | str,
    remove_keys: Optional[list[str]] = None,
) -> str:
    """Get description from a given file path or use the default description.

    Args:
        path_str (str, optional): The file path to the description text file.
            If None, the default description will be used.
        default_description (dict[str, str]): The default description to use if path_str is None.
        remove_keys (list[str], optional): A list of keys to remove from the description dictionary.
            If None, no keys will be removed.

    Returns:
        str: The description obtained from the file or the default description.

    Note:
        See also:
            econsimulacra.llm_services.PromptBuilder._get_obs_action_description()
            econsimulacra.llm_services.PersonaBuilder._get_persona_description()
    """
    desc: str
    desc_path: Path
    if isinstance(default_description, dict):
        desc_dic: dict[str, str]
        if path_str is not None:
            desc_path = pathlib.Path(path_str).resolve()
            if not desc_path.exists():
                raise FileNotFoundError(f"Description file not found at: {desc_path}")
            if not desc_path.suffix.lower() == ".json":
                raise ValueError(f"Description file must be a JSON file: {desc_path}")
            desc_dic = json.loads(desc_path.read_text(encoding="utf-8"))  # type: ignore
        else:
            desc_dic = default_description
        if remove_keys is not None:
            for key in remove_keys:
                if key in desc_dic:
                    del desc_dic[key]
        desc = json.dumps(desc_dic, ensure_ascii=False)
    elif isinstance(default_description, str):
        if path_str is not None:
            desc_path = pathlib.Path(path_str).resolve()
            if not desc_path.exists():
                raise FileNotFoundError(f"Description file not found at: {desc_path}")
            desc = desc_path.read_text(encoding="utf-8")
        else:
            desc = default_description
    else:
        raise ValueError(f"Invalid default_description: {default_description}")
    return desc
