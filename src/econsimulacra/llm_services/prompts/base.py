import json
import pathlib
import random
from pathlib import Path
from typing import Any, Optional, Type

from ..constant import (
    DEFAULT_ACTION_DESCRIPTION,
    DEFAULT_OBS_DESCRIPTION,
    DEFAULT_SIMULATION_DESCRIPTION,
)


class PromptBuilder:
    """Prompt Builder class. Prompt builders are responsible for generating prompts
    except for the persona description (if applicable)
    i.e., they translate the observation into a prompt for LLM input.
    You can implement your own prompt builder by inheriting this class and implementing the build_prompt method.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): The configuration dictionary. This may include:
                - "simulationDescriptionPath": (optional) the file path to the simulation description text file.
                    If not provided, a default simulation description will be used.
                - "obsDescriptionPath": (optional) the file path to the observation description text file.
                    If not provided, a default observation description will be used.
                - "actionDescriptionPath": (optional) the file path to the action description text file.
                    If not provided, a default action description will be used.
            prng (Optional[random.Random]): The random number generator.
        """
        self.config: dict[str, Any] = config
        self.simulation_desc: str = self._get_simulation_description(config)
        self.obs_desc, self.action_desc = self._get_obs_action_description(config)
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = registered_classes

    def build_prompt(self, obs: dict[str, Any]) -> str:
        """Translate the observation into a prompt for LLM input.

        Args:
            obs (dict[str, Any]): the observation to translate into a prompt for LLM input

        Returns:
            str: the generated prompt for LLM input

        Note:
            Called by LLMAgent.act
        """
        obs = self._truncate_floats(obs)
        obs_str: str = json.dumps(obs, ensure_ascii=False)
        prompt: str = (
            f"\n{self.simulation_desc}\n"
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

    def _get_simulation_description(self, config: dict[str, Any]) -> str:
        """Get simulation description from config for better LLM understanding.

        Args:
            config (dict[str, Any]): The configuration dictionary. This may include:
                - "simulationDescriptionPath": (optional) the file path to the simulation description text file.
                    If not provided, a default simulation description will be used.

        Returns:
            str: simulation description
        """
        sim_desc_path_str: Optional[str] = config.get("simulationDescriptionPath")
        sim_desc: str
        if sim_desc_path_str is not None:
            sim_desc_path: Path = pathlib.Path(sim_desc_path_str).resolve()
            if not sim_desc_path.exists():
                raise FileNotFoundError(
                    f"Simulation description file not found at: {sim_desc_path}"
                )
            sim_desc = sim_desc_path.read_text(encoding="utf-8")
        else:
            sim_desc = DEFAULT_SIMULATION_DESCRIPTION
        return sim_desc

    def _get_obs_action_description(self, config: dict[str, Any]) -> tuple[str, str]:
        """Get description of observations and actions from config for better LLM understanding.

        Args:
            config (dict[str, Any]): The configuration dictionary. This may include:
                - "obsDescriptionPath": (optional) the file path to the observation description text file.
                    If not provided, a default observation description will be used.
                - "actionDescriptionPath": (optional) the file path to the action description text file.
                    If not provided, a default action description will be used.

        Returns:
            tuple[str, str]: observation and action descriptions

        Note:
            For the default observation and action descriptions, see also:
                - econsimulacra.constant.DEFAULT_OBS_DESCRIPTION
                - econsimulacra.constant.DEFAULT_ACTION_DESCRIPTION, respectively.
        """
        obs_desc_path_str: Optional[str] = config.get("obsDescriptionPath")
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
        action_desc_path_str: Optional[str] = config.get("actionDescriptionPath")
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
