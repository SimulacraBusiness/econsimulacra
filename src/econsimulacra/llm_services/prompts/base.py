from typing import Any


class PromptBuilder:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config: dict[str, Any] = config

    def build_prompt(self, obs: dict[str, Any]) -> str:
        return str(obs)
