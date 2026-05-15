import json
import pathlib
import random
from pathlib import Path
from typing import Any, Optional, Type

from .base import PromptBuilder


class NameBasedPromptBuilder(PromptBuilder):
    """PromptBuilder class that uses different simulation descriptions
    depending on agent name patterns.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): The configuration dictionary. It must include:
                - "name2SimulationDescriptionPath":
                    dict[str, str]
                    Mapping from name pattern to simulation description path.

                    Example:
                    {
                        "Household": "...",
                        "Government": "...",
                        "others": "..."
                    }

                and may include:
                - "obsDescriptionPath"
                - "actionDescriptionPath"
                - "maskedObses"
                - "disabledActions"

            prng (random.Random, optional): Random number generator.
            registered_classes (list[Type], optional): List of registered classes.
        """
        super().__init__(
            config=config,
            prng=prng,
            registered_classes=registered_classes,
        )
        self.name2simulation_desc: dict[str, str] = self._load_simulation_descriptions(
            config
        )

    def _load_simulation_descriptions(
        self,
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Load simulation descriptions from config.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            dict[str, str]:
                Mapping from name pattern to loaded description text.
        """
        if "name2SimulationDescriptionPath" not in config:
            raise ValueError(
                "The configuration dictionary must include "
                + "'name2SimulationDescriptionPath'."
            )

        path_map: dict[str, str] = config["name2SimulationDescriptionPath"]

        if "others" not in path_map:
            raise ValueError(
                "'name2SimulationDescriptionPath' must include "
                + "'others' as fallback."
            )

        loaded_descs: dict[str, str] = {}

        for name_pattern, path_str in path_map.items():
            path: Path = pathlib.Path(path_str).resolve()

            if not path.exists():
                raise FileNotFoundError(
                    "Simulation description file not found at: " + f"{path}"
                )

            loaded_descs[name_pattern] = path.read_text(encoding="utf-8")

        return loaded_descs

    def build_prompt(self, obs: dict[str, Any]) -> str:
        """Translate observation into LLM prompt.

        Args:
            obs (dict[str, Any]): Observation.

        Returns:
            str: Prompt.
        """
        obs = self._truncate_floats(obs)

        simulation_desc: str = self._get_simulation_description_for_obs(obs)

        obs_str: str = json.dumps(obs, ensure_ascii=False)

        prompt: str = (
            f"\n{simulation_desc}\n"
            + f"Observation description: {self.obs_desc}\n"
            + f"Action description: {self.action_desc}\n"
            + f"Observation: {obs_str}\n"
            + "Respond in JSON format."
        )

        return prompt

    def _get_simulation_description_for_obs(
        self,
        obs: dict[str, Any],
    ) -> str:
        """Select simulation description based on self_name.

        Matching rule:
        - If pattern is contained in self_name, use that description.
        - If multiple patterns match, the first inserted one is used.
        - If nothing matches, use 'others'.

        Args:
            obs (dict[str, Any]): Observation.

        Returns:
            str: Simulation description.
        """
        self_name: str = obs["self_name"]

        for pattern, desc in self.name2simulation_desc.items():
            if pattern == "others":
                continue

            if pattern in self_name:
                return desc

        return self.name2simulation_desc["others"]
