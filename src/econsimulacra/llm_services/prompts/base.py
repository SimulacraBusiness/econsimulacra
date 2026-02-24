from ..constant import DEFAULT_ACTION_DESCRIPTION
from ..constant import DEFAULT_OBS_DESCRIPTION
import json
import pathlib
from pathlib import Path
from typing import Any
from typing import Optional


class PromptBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config
        self.obs_desc, self.action_desc = self._get_obs_action_description(config)

    def build_prompt(self, obs: dict[str, Any]) -> str:
        obs = self._truncate_floats(obs)
        obs_str: str = json.dumps(obs, ensure_ascii=False)
        prompt: str = "You are a member of the society. Based on the following observation, decide the action to take.\n" + \
                    f"Observation description: {self.obs_desc}\nAction description: {self.action_desc}\nRespond in JSON format." + \
                    f" Respond in JSON format.\nObservation: {obs_str}"
        return prompt

    def _truncate_floats(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._truncate_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._truncate_floats(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._truncate_floats(v) for v in obj)
        elif isinstance(obj, float):
            return int(obj)

        else:
            return obj

    def _get_obs_action_description(self, config: dict[str, Any]) -> tuple[str, str]:
        """get description of observations and actions from config for better LLM understanding."""
        obs_desc_path_str: Optional[str] = config.get("obs_description_path")
        if obs_desc_path_str is not None:
            obs_desc_path: Path = pathlib.Path(obs_desc_path_str).resolve()
            if not obs_desc_path.exists():
                raise FileNotFoundError(f"Observation description file not found at: {obs_desc_path}")
            obs_desc: str = obs_desc_path.read_text(encoding="utf-8")
        else:
            obs_desc: str = json.dumps(DEFAULT_OBS_DESCRIPTION, ensure_ascii=False)
        action_desc_path_str: Optional[str] = config.get("action_description_path")
        if action_desc_path_str is not None:
            action_desc_path: Path = pathlib.Path(action_desc_path_str).resolve()
            if not action_desc_path.exists():
                raise FileNotFoundError(f"Action description file not found at: {action_desc_path}")
            action_desc: str = action_desc_path.read_text(encoding="utf-8")
        else:
            action_desc: str = json.dumps(DEFAULT_ACTION_DESCRIPTION, ensure_ascii=False)
        return obs_desc, action_desc