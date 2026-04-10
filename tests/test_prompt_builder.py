import json

from econsimulacra.llm_services import (
    DEFAULT_ACTION_DESCRIPTION,
    DEFAULT_OBS_DESCRIPTION,
    PromptBuilder,
)


class TestPromptBuilder:
    def test_build_prompt(self) -> None:
        prompt_builder = PromptBuilder(config={})
        obs: dict[str, str] = {"key1": "value1", "key2": 2.5}
        prompt: str = prompt_builder.build_prompt(obs)
        assert (
            f"Observation description: {json.dumps(DEFAULT_OBS_DESCRIPTION, ensure_ascii=False)}"
            in prompt
        )
        assert (
            f"Action description: {json.dumps(DEFAULT_ACTION_DESCRIPTION, ensure_ascii=False)}"
            in prompt
        )
        assert '"key1": "value1"' in prompt
        assert '"key2": 2' in prompt
        config: dict[str, str] = {
            "obsDescriptionPath": "tests/dummy_obs_description.txt",
            "actionDescriptionPath": "tests/dummy_action_description.txt",
        }
        prompt_builder = PromptBuilder(config=config)
        prompt: str = prompt_builder.build_prompt(obs)
        assert "Observation description: This is a dummy obs description." in prompt
        assert "Action description: This is a dummy action description." in prompt
        assert '"key1": "value1"' in prompt
        assert '"key2": 2' in prompt
