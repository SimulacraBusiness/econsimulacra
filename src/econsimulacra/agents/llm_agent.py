from .base import Agent
from ..llm_services import LLMClient
from ..llm_services import PersonaBuilder
from ..llm_services import PromptBuilder
from typing import Any
from typing import Optional


class LLMAgent(Agent[dict[str, Any]]):
    def _setup_env_services(self, env_service_dic: dict[str, Any]) -> None:
        """setup environment services for the agent based on self.service_dic.

        See also:
            econsimulacra.envs.base._generate_service_providers
            econsimulacra.agents.LLMAgent
        """
        if "llmClient" not in env_service_dic:
            raise ValueError(
                f"LLMAgent {self.agent_name} requires 'llmClient' in env_service_dic."
            )
        self.llm_client: LLMClient = env_service_dic["llmClient"]
        self.persona_builder: Optional[PersonaBuilder] = None
        if "personaBuilder" in env_service_dic:
            self.persona_builder = env_service_dic["personaBuilder"]
            self.persona_builder.build_persona(
                agent_id=self.agent_id,
                agent_config=self.config.get("personaConfig", {}),
            )
        if "promptBuilder" not in env_service_dic:
            raise ValueError(
                f"LLMAgent {self.agent_name} requires 'promptBuilder' in env_service_dic."
            )
        self.prompt_builder: PromptBuilder = env_service_dic["promptBuilder"]

    def self_assign_name(self, config):
        super().self_assign_name(config)
        if self.persona_builder is not None:
            self.agent_name = self.persona_builder.assign_name(
                agent_id=self.agent_id,
                default_name=self.agent_name,
                config=config.get("personaConfig", {}),
            )

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        persona_prompt: str = ""
        if self.persona_builder is not None:
            persona_prompt = self.persona_builder.build_persona_prompt(
                agent_id=self.agent_id
            )
        obs_prompt: str = self.prompt_builder.build_prompt(obs=obs)
        prompt: str = persona_prompt + obs_prompt
        llm_response: dict[str, Any] = await self.llm_client.generate_response(
            prompt=prompt
        )
        return llm_response
