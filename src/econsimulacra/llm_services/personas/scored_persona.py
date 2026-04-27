import random
from typing import Any, Optional, Type

from .base import PersonaBuilder


class ScoredPersonaBuilder(PersonaBuilder):
    """Persona builder that builds personas with scores.

    The persona is represented as a dictionary of attributes
    and their corresponding scores.
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
                - "type": the type of persona builder to use (e.g., "ScoredPersonaBuilder").
                - "attributes": a list of attributes to include in the persona, which is a list of strings.
                and may include:
                - "maxMagnitude": the maximum magnitude for each trait,
                    which is a non-negative integer and the default value is 1).
            prng (random.Random, optional): An optional instance of random.Random for reproducible randomness.
                If not provided, a new instance will be created.
        """
        super().__init__(
            config=config, prng=prng, registered_classes=registered_classes
        )
        self.attributes: list[str] = config["attributes"]
        self.max_magnitude: int = config.get("maxMagnitude", 1)

    def build_persona(self, agent_id: int, agent_config: dict) -> None:
        """Register the agent to agent_id2persona_dic with random scores for each attribute.

        Args:
            agent_id (int): agent_id of the agent to build persona for
            agent_config (dict): config of the agent to build persona for,
                which is the same as the one in env_config["agents"][agent_name]

        Note:
            Called when LLMAgent is initialized.
            See also: econsimulacra.agents.llm_agent.LLMAgent._setup_env_services()
        """
        if agent_id in self.agent_id2persona_dic:
            raise ValueError(f"Agent ID {agent_id} already exists in persona builder.")
        trait_value_dic: dict[str, int] = {
            attribute: self.prng.randint(0, self.max_magnitude)
            for attribute in self.attributes
        }
        self.agent_id2persona_dic[agent_id] = trait_value_dic
