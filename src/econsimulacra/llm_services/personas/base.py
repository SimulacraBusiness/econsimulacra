from abc import ABC, abstractmethod
import json
import random
from typing import Any
from typing import Optional


class PersonaBuilder(ABC):
    def __init__(
        self, config: dict[str, Any], prng: Optional[random.Random] = None
    ) -> None:
        self.config: dict[str, Any] = config
        self.agent_id2persona_dic: dict[int, dict[str, Any]] = {}
        self.prng: random.Random = prng if prng is not None else random.Random()

    @abstractmethod
    def build_persona(self, agent_id: int, agent_config: dict) -> None:
        """register the agent to agent_id2persona_dic.

        Args:
            agent_id (int): agent_id of the agent to build persona for
            agent_config (dict): config of the agent to build persona for, which is the same as the one in env_config["agents"][agent_name]

        Note:
            Called when LLMAgent is initialized.
        """
        pass

    def get_persona_description(self, agent_id: int) -> str:
        persona_desc: str = "Role-playing as the following persona.\n"
        return persona_desc

    def build_persona_prompt(self, agent_id: int) -> str:
        """build persona prompt for the agent with the given agent_id.

        Args:
            agent_id (int): agent_id of the agent to build persona prompt for

        Returns:
            str: persona prompt for the agent

        Note:
            Called when LLMAgent.act is called.
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
