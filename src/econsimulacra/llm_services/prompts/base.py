from ..constant import DEFAULT_ACTION_DESCRIPTION
from ..constant import DEFAULT_OBS_DESCRIPTION
import json
import pathlib
from pathlib import Path
import random
from typing import Any
from typing import Optional


class PromptBuilder:
    def __init__(
        self, config: dict[str, Any], prng: Optional[random.Random] = None
    ) -> None:
        self.config: dict[str, Any] = config
        self.obs_desc, self.action_desc = self._get_obs_action_description(config)
        self.prng: random.Random = prng if prng is not None else random.Random()

    def build_prompt(self, obs: dict[str, Any]) -> str:
        """translate the observation into a prompt for LLM input.

        Args:
            obs (dict[str, Any]): the observation to translate into a prompt for LLM input

        Note:
            Called by LLMAgent.act
        """
        obs = self._truncate_floats(obs)
        obs_str: str = json.dumps(obs, ensure_ascii=False)
        prompt: str = (
            "\nYou are a member of the society. Based on the following observation, decide the action to take.\n"
            + f"Observation description: {self.obs_desc}\nAction description: {self.action_desc}"
            + f"\nObservation: {obs_str}\nRespond in JSON format."
        )
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
        obs_desc: str
        if obs_desc_path_str is not None:
            obs_desc_path: Path = pathlib.Path(obs_desc_path_str).resolve()
            if not obs_desc_path.exists():
                raise FileNotFoundError(
                    f"Observation description file not found at: {obs_desc_path}"
                )
            obs_desc = obs_desc_path.read_text(encoding="utf-8")
        else:
            obs_desc = json.dumps(DEFAULT_OBS_DESCRIPTION, ensure_ascii=False)
        action_desc_path_str: Optional[str] = config.get("action_description_path")
        action_desc: str
        if action_desc_path_str is not None:
            action_desc_path: Path = pathlib.Path(action_desc_path_str).resolve()
            if not action_desc_path.exists():
                raise FileNotFoundError(
                    f"Action description file not found at: {action_desc_path}"
                )
            action_desc = action_desc_path.read_text(encoding="utf-8")
        else:
            action_desc = json.dumps(DEFAULT_ACTION_DESCRIPTION, ensure_ascii=False)
        return obs_desc, action_desc
