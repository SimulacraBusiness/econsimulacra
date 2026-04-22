import json
import random
from abc import ABC, abstractmethod
from typing import Any, Optional


class PersonaBuilder(ABC):
    """Persona Builder class (abstract class).

    You can implement your own persona builder by inheriting this class and implementing the build_persona method.
    Currently, Big5PersonaBuilder is implemented as a built-in option, which builds personas based on the Big5 personality traits.

    See also:
        - econsimulacra.llm_services.personas.big5.Big5PersonaBuilder
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[random.Random] = None,
        registered_classes: Optional[dict[str, type]] = None,
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the persona builder. This must include:
                - "type": the type of persona builder to use (e.g., "Big5PersonaBuilder").
                and may include:
                - other persona builder-specific parameters (e.g., for Big5PersonaBuilder,
                    "maxMagnitude": the maximum magnitude for each Big 5 trait, which is a non-negative integer and the default value is 1).
            prng (random.Random, optional): An optional instance of random.Random for reproducible randomness.
                If not provided, a new instance will be created.
        """
        self.config: dict[str, Any] = config
        self.agent_id2persona_dic: dict[int, dict[str, Any]] = {}
        self.prng: random.Random = prng if prng is not None else random.Random()
        self.registered_classes: Optional[dict[str, type]] = registered_classes

    @abstractmethod
    def build_persona(self, agent_id: int, agent_config: dict) -> None:
        """Register the agent to agent_id2persona_dic.

        Args:
            agent_id (int): agent_id of the agent to build persona for
            agent_config (dict): config of the agent to build persona for, which is the same as the one in env_config["agents"][agent_name]

        Note:
            Called when LLMAgent is initialized.
            See also:
                econsimulacra.agents.llm_agent.LLMAgent._setup_env_services()
        """
        pass

    def get_persona(self, agent_id: int) -> Optional[dict[str, Any]]:
        """Get the persona for the agent with the given agent_id."""
        return self.agent_id2persona_dic.get(agent_id)

    def get_persona_description(self, agent_id: int) -> str:
        """Get the description of the persona for the agent with the given agent_id."""
        persona_desc: str = "Role-playing as the following persona.\n"
        return persona_desc

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
        persona_desc: str = self.get_persona_description(agent_id=agent_id)
        persona_str: str = json.dumps(
            self.agent_id2persona_dic[agent_id], ensure_ascii=False
        )
        return persona_desc + persona_str

    def assign_name(self, agent_id: int, default_name: str, config: dict) -> str:
        return default_name
