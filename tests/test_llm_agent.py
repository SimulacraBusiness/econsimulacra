from econsimulacra.agents import LLMAgent
from econsimulacra.llm_services import Big5PersonaBuilder, LLMClient, PromptBuilder


class DummyClient(LLMClient):
    async def generate_response(self, prompt: str) -> dict[str, str]:
        return {"response": f"Echo: {prompt}"}


class DummpyBig5PersonaBuilder(Big5PersonaBuilder):
    def assign_name(self, agent_id: int, default_name: str, config: dict) -> str:
        return f"Dummy{agent_id}"


class TestLLMAgent:
    def test_init(self):
        prompt_builder = PromptBuilder(config={})
        persona_builder = Big5PersonaBuilder(config={"maxMagnitude": 5})
        llm_client = DummyClient(config={"modelName": "dummy"})
        env_service_dic = {
            "promptBuilder": prompt_builder,
            "personaBuilder": persona_builder,
            "llmClient": llm_client,
        }
        for agent_id in range(30):
            agent = LLMAgent(
                agent_id=agent_id,
                agent_name=f"Agent{agent_id}",
                config={},
                env_service_dic=env_service_dic,
            )
            assert agent_id in persona_builder.agent_id2persona_dic
            persona_dic = persona_builder.agent_id2persona_dic[agent_id]
            assert all(
                trait in persona_dic
                for trait in [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ]
            )
            assert all(
                0 <= persona_dic[trait] <= 5
                for trait in [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ]
            )
        for agent_id in range(30, 60):
            agent = LLMAgent(
                agent_id=agent_id,
                agent_name=f"Agent{agent_id}",
                config={"name": "TestAgent", "modelName": "dummy"},
                env_service_dic=env_service_dic,
            )
            assert agent.agent_name == f"TestAgent{agent_id}"
        env_service_dic["personaBuilder"] = DummpyBig5PersonaBuilder(
            config={"maxMagnitude": 5}
        )
        for agent_id in range(60, 90):
            agent = LLMAgent(
                agent_id=agent_id,
                agent_name=f"Agent{agent_id}",
                config={"modelName": "dummy"},
                env_service_dic=env_service_dic,
            )
            assert agent.agent_name == f"Dummy{agent_id}"
