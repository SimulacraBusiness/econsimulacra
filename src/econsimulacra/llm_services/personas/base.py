import json
import random
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

from ..llm_utils import get_description


class PersonaBuilder(ABC):
    """Persona Builder class (abstract class).

    You can implement your own persona builder by inheriting this class and implementing the build_persona method.
    Currently, Big5PersonaBuilder is implemented as a built-in option, which builds personas based on the Big5 personality traits.

    See also: econsimulacra.llm_services.personas.big5.Big5PersonaBuilder
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the persona builder. This must include:
                - "type": the type of persona builder to use (e.g., "Big5PersonaBuilder").
                and may include:
                - "personaDescriptionPath": the file path to the persona description,
                    which is a text file describing the persona
                    (e.g., for Big5PersonaBuilder, it describes the Big 5 traits and how they affect personality).
                - other persona builder-specific parameters (e.g., for Big5PersonaBuilder,
                    "maxMagnitude": the maximum magnitude for each Big 5 trait, which is a non-negative integer and the default value is 1).
            prng (random.Random, optional): An optional instance of random.Random for reproducible randomness.
                If not provided, a new instance will be created.
        """
        self.config: dict[str, Any] = config
        self.persona_desc: str = self._get_persona_description(config)
        self.agent_id2persona_dic: dict[int, dict[str, Any]] = {}
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: list[Type] = registered_classes

    @abstractmethod
    def build_persona(self, agent_id: int, agent_config: dict) -> None:
        """Register the agent to agent_id2persona_dic.

        Args:
            agent_id (int): agent_id of the agent to build persona for
            agent_config (dict): config of the agent to build persona for, which is the same as the one in env_config["agents"][agent_name]

        Note:
            Called when LLMAgent is initialized.
            See also: econsimulacra.agents.llm_agent.LLMAgent._setup_env_services()
        """
        pass

    def get_persona(self, agent_id: int) -> Optional[dict[str, Any]]:
        """Get the persona for the agent with the given agent_id."""
        return self.agent_id2persona_dic.get(agent_id)

    def _get_persona_description(self, config: dict[str, Any]) -> str:
        """Get the description of the persona for the agent with the given agent_id.

        Args:
            config (dict[str, Any]): The configuration dictionary. This may include:
                - "personaDescriptionPath": (optional) the file path to the persona description text file.
                    If not provided, a default persona description will be used.

        Returns:
            str: the description of the persona for the agent
        """
        return get_description(
            path_str=config.get("personaDescriptionPath"),
            default_description="Role-playing as the following persona.\n",
        )

    def build_persona_prompt(self, agent_id: int) -> str:
        """Build persona prompt for the agent with the given agent_id.

        Args:
            agent_id (int): agent_id of the agent to build persona prompt for

        Returns:
            str: persona prompt for the agent

        Note:
            Called when LLMAgent.act is called.
            persona prompt contains the description and the persona information of the agent,
            and is used as part of the prompt for generation.
        """
        if agent_id not in self.agent_id2persona_dic:
            raise ValueError(f"Agent ID {agent_id} not found in persona builder.")
        persona_str: str = json.dumps(
            self.agent_id2persona_dic[agent_id], ensure_ascii=False
        )
        return self.persona_desc + persona_str

    def assign_name(self, agent_id: int, default_name: str, config: dict) -> str:
        return default_name
