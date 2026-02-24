from .base import Agent
from ..llm_services import LLMClient
from ..llm_services import PromptBuilder
from typing import Any


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
        if "promptBuilder" not in env_service_dic:
            raise ValueError(
                f"LLMAgent {self.agent_name} requires 'promptBuilder' in env_service_dic."
            )
        self.prompt_builder: PromptBuilder = env_service_dic["promptBuilder"]

    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        prompt: str = self.prompt_builder.build_prompt(obs=obs)
        llm_response: dict[str, Any] = await self.llm_client.generate_response(
            prompt=prompt
        )
        return llm_response
