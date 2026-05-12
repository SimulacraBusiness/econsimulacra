from __future__ import annotations

import asyncio
import json
import os
import pathlib
import random
from pathlib import Path
from typing import Any, Optional, Type

from econsimulacra.envs import Environment
from econsimulacra.llm_services import PromptBuilder
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import SimulationSummarizer, Simulator


class PromptBuilderWithRetailer(PromptBuilder):
    """PromptBuilder class that use simulation description separately
    for retailers and households.
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict[str, Any]): The configuration dictionary. This must include:
                - "simulationDescriptionPathForRetailers": the file path
                    to the simulation description text file for retailers.
                - "simulationDescriptionPathForHouseholds": the file path
                    to the simulation description text file for households.
                and may include:
                - "obsDescriptionPath": (optional) the file path
                    to the observation description text file.
                    If not provided, a default observation description will be used.
                - "actionDescriptionPath": (optional) the file path
                    to the action description text file.
                    If not provided, a default action description will be used.
            prng (Optional[random.Random]): The random number generator.
        """
        super().__init__(config=config, prng=prng, registered_classes=registered_classes)
        self.config: dict[str, Any] = config
        self.simulation_desc4retailers: str
        self.simulation_desc4households: str
        self.simulation_desc4retailers, self.simulation_desc4households = (
            self._get_tupled_simulation_description(config)
        )

    def _get_tupled_simulation_description(
        self, config: dict[str, Any]
    ) -> tuple[str, str]:
        """Get the simulation description for retailers and households.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Returns:
            tuple[str, str]: The simulation description
                for retailers and households.
        """
        if "simulationDescriptionPathForRetailers" not in config:
            raise ValueError(
                "The configuration dictionary must include "
                + "'simulationDescriptionPathForRetailers'."
            )
        else:
            sim_desc_path4retailers_str: str = config[
                "simulationDescriptionPathForRetailers"
            ]
            sim_desc_path4retailers: Path = pathlib.Path(
                sim_desc_path4retailers_str
            ).resolve()
            if not sim_desc_path4retailers.exists():
                raise FileNotFoundError(
                    "Simulation description file not found at: "
                    + f"{sim_desc_path4retailers}"
                )
            sim_desc4retailers = sim_desc_path4retailers.read_text(encoding="utf-8")
        if "simulationDescriptionPathForHouseholds" not in config:
            raise ValueError(
                "The configuration dictionary must include "
                + "'simulationDescriptionPathForHouseholds'."
            )
        else:
            sim_desc_path4households_str: str = config[
                "simulationDescriptionPathForHouseholds"
            ]
            sim_desc_path4households: Path = pathlib.Path(
                sim_desc_path4households_str
            ).resolve()
            if not sim_desc_path4households.exists():
                raise FileNotFoundError(
                    "Simulation description file not found at: "
                    + f"{sim_desc_path4households}"
                )
            sim_desc4households = sim_desc_path4households.read_text(encoding="utf-8")
        return sim_desc4retailers, sim_desc4households

    def build_prompt(self, obs: dict[str, Any]) -> str:
        """Translate the observation into a prompt for LLM input.

        Args:
            obs (dict[str, Any]): the observation
                to translate into a prompt for LLM input

        Returns:
            str: the generated prompt for LLM input

        Note:
            Called by LLMAgent.act
        """
        obs = self._truncate_floats(obs)
        simulation_desc: str = self._get_simulation_description_for_obs(obs)
        obs_str: str = json.dumps(obs, ensure_ascii=False)
        prompt: str = (
            f"\n{simulation_desc}\n"
            + f"Observation description: {self.obs_desc}\n"
            + f"Action description: {self.action_desc}"
            + f"\nObservation: {obs_str}\nRespond in JSON format."
        )
        return prompt

    def _get_simulation_description_for_obs(self, obs: dict[str, Any]) -> str:
        self_name: str = obs["self_name"]
        if "Household" in self_name:
            return self.simulation_desc4households
        else:
            return self.simulation_desc4retailers


# export LOG_TXT_PATH="log.txt"


def conduct_simulation():
    config_dic_path: Path = pathlib.Path(__file__).parent / "config.json"
    logger: DictLogger = DictLogger()
    simulator: Simulator = Simulator(
        config=config_dic_path,
        env_class=Environment,
        logger=logger,
        summarizer_class=SimulationSummarizer,
    )
    simulator.register_classes([PromptBuilderWithRetailer])
    asyncio.run(simulator.simulate(seed=42))
    logs: list[dict] = logger.logs
    log_txt_path: Path = pathlib.Path(os.environ["LOG_TXT_PATH"])
    with open(log_txt_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    conduct_simulation()
