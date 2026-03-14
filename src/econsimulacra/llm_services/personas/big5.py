from .base import PersonaBuilder
import random
from typing import Any
from typing import Optional


class Big5PersonaBuilder(PersonaBuilder):
    """Persona builder that builds personas based on the Big 5 personality traits."""

    def __init__(
        self, config: dict[str, Any], prng: Optional[random.Random] = None
    ) -> None:
        """Initialization.

        Args:
            config (dict): Configuration dictionary for the persona builder. This must include:
                - "type": the type of persona builder to use (e.g., "Big5PersonaBuilder").
                and may include:
                - "maxMagnitude": the maximum magnitude for each Big 5 trait, which is a non-negative integer and the default value is 1).
            prng (random.Random, optional): An optional instance of random.Random for reproducible randomness.
                If not provided, a new instance will be created.
        """
        super().__init__(config=config, prng=prng)
        self.max_magnitude: int = config.get("maxMagnitude", 1)

    def build_persona(self, agent_id: int, agent_config: dict) -> None:
        """Build persona based on the Big 5 personality traits.

        Args:
            agent_id (int): agent_id of the agent to build persona for
            agent_config (dict): config of the agent to build persona for, which is the same as the one in env_config["agents"][agent_name]
        """
        if agent_id in self.agent_id2persona_dic:
            raise ValueError(f"Agent ID {agent_id} already exists in persona builder.")
        big5_traits = [
            "Openness",
            "Conscientiousness",
            "Extraversion",
            "Agreeableness",
            "Neuroticism",
        ]
        trait_value_dic = {
            trait: self.prng.randint(0, self.max_magnitude) for trait in big5_traits
        }
        self.agent_id2persona_dic[agent_id] = trait_value_dic

    def get_persona_description(self, agent_id: int) -> str:
        return (
            "Role-playing as a person with the following Big 5 personality traits. "
            + "The value of each trait is an integer between 0 and "
            + str(self.max_magnitude)
            + ", where a higher value indicates a stronger presence of the trait.\n"
        )
