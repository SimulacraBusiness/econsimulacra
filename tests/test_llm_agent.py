import asyncio

from econsimulacra.agents import AutoReactLLMAgent, LLMAgent
from econsimulacra.llm_services import LLMClient, PromptBuilder, ScoredPersonaBuilder


class DummyClient(LLMClient):
    async def generate_response(self, prompt: str) -> dict[str, str]:
        return {"response": f"Echo: {prompt}"}


class DummpyBig5PersonaBuilder(ScoredPersonaBuilder):
    def assign_name(self, agent_id: int, default_name: str, config: dict) -> str:
        return f"Dummy{agent_id}"


class TestLLMAgent:
    def test_init(self):
        prompt_builder = PromptBuilder(config={})
        persona_builder = ScoredPersonaBuilder(
            config={
                "maxMagnitude": 5,
                "attributes": [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ],
            }
        )
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
            config={
                "maxMagnitude": 5,
                "attributes": [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ],
            }
        )
        for agent_id in range(60, 90):
            agent = LLMAgent(
                agent_id=agent_id,
                agent_name=f"Agent{agent_id}",
                config={"modelName": "dummy"},
                env_service_dic=env_service_dic,
            )
            assert agent.agent_name == f"Dummy{agent_id}"


class TestAutoReactLLMAgent:
    def test_init(self):
        prompt_builder = PromptBuilder(config={})
        persona_builder = DummpyBig5PersonaBuilder(
            config={
                "maxMagnitude": 5,
                "attributes": [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ],
            }
        )
        llm_client = DummyClient(config={"modelName": "dummy"})
        env_service_dic = {
            "promptBuilder": prompt_builder,
            "personaBuilder": persona_builder,
            "llmClient": llm_client,
        }
        agent = AutoReactLLMAgent(
            agent_id=0,
            agent_name="AutoReactAgent",
            config={"modelName": "dummy"},
            env_service_dic=env_service_dic,
        )
        assert isinstance(agent, AutoReactLLMAgent)

    def test_judge_reaction(self):
        prompt_builder = PromptBuilder(config={})
        persona_builder = DummpyBig5PersonaBuilder(
            config={
                "maxMagnitude": 5,
                "attributes": [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ],
            }
        )
        llm_client = DummyClient(config={"modelName": "dummy"})
        env_service_dic = {
            "promptBuilder": prompt_builder,
            "personaBuilder": persona_builder,
            "llmClient": llm_client,
        }
        agent = AutoReactLLMAgent(
            agent_id=0,
            agent_name="AutoReactAgent",
            config={"modelName": "dummy"},
            env_service_dic=env_service_dic,
        )
        current_inventory = {"itemA": 10, "itemB": 5.0}
        incoming_order = {"item_name": "itemA", "item_amount": 3}
        judge, _current_inventory = agent.judge_reaction(
            incoming_transactional_intent=incoming_order,
            current_inventory=current_inventory,
            is_order=True,
        )
        incoming_order = {"item_name": "itemA", "item_amount": 9}
        judge, _current_inventory = agent.judge_reaction(
            incoming_transactional_intent=incoming_order,
            current_inventory=_current_inventory,
            is_order=True,
        )
        assert not judge
        incoming_proposal = {"get_item_name": "itemB", "get_item_amount": 4}
        judge, _current_inventory = agent.judge_reaction(
            incoming_transactional_intent=incoming_proposal,
            current_inventory=_current_inventory,
            is_order=False,
        )
        assert judge
        incoming_proposal = {"get_item_name": "itemB", "get_item_amount": 2}
        judge, _current_inventory = agent.judge_reaction(
            incoming_transactional_intent=incoming_proposal,
            current_inventory=_current_inventory,
            is_order=False,
        )
        assert not judge

    def test_act(self):
        prompt_builder = PromptBuilder(config={})
        persona_builder = DummpyBig5PersonaBuilder(
            config={
                "maxMagnitude": 5,
                "attributes": [
                    "Openness",
                    "Conscientiousness",
                    "Extraversion",
                    "Agreeableness",
                    "Neuroticism",
                ],
            }
        )
        llm_client = DummyClient(config={"modelName": "dummy"})
        env_service_dic = {
            "promptBuilder": prompt_builder,
            "personaBuilder": persona_builder,
            "llmClient": llm_client,
        }
        agent = AutoReactLLMAgent(
            agent_id=0,
            agent_name="AutoReactAgent",
            config={
                "modelName": "dummy",
                "inventory": {"itemA": 10, "itemB": 5.0},
            },
            env_service_dic=env_service_dic,
        )
        obs = {
            "incoming_orders": [
                {"order_id": 1, "item_name": "itemA", "item_amount": 3},
                {"order_id": 2, "item_name": "itemA", "item_amount": 15},
            ],
            "incoming_proposals": [
                {"proposal_id": 1, "get_item_name": "itemB", "get_item_amount": 4},
                {"proposal_id": 2, "get_item_name": "itemB", "get_item_amount": 6},
            ],
        }
        llm_response = asyncio.run(agent.act(obs=obs))
        reactions = llm_response["reactions"]
        assert len(reactions) == 2
        assert reactions[0] == {"kind": "order", "id": 1, "accept_amount": 3}
        assert reactions[1] == {"kind": "proposal", "id": 1, "accept": True}
