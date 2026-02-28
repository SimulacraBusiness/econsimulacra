from .base import PersonaBuilder
from typing import Any


class Big5PersonaBuilder(PersonaBuilder):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config=config)
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
