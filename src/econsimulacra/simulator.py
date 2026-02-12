import json
from .logs.base import Logger
from .envs.base import Environment
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Type


class Simulator:
    def __init__(
        self,
        config: dict[str, Any] | Path,
        env_class: Type[Environment],
        logger: Optional[Logger] = None,
    ) -> None:
        if isinstance(config, Path):
            config_path: Path = config
            self.config: dict[str, Any] = json.load(open(config_path, "r"))
        else:
            self.config: dict[str, Any] = config
        self.config = self._convert_list_to_tuple(self.config)
        self.env: Environment = env_class(config=self.config, logger=logger)

    def _convert_list_to_tuple(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                self._convert_list_to_tuple(k): self._convert_list_to_tuple(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return tuple(self._convert_list_to_tuple(item) for item in obj)
        elif isinstance(obj, tuple):
            return tuple(self._convert_list_to_tuple(item) for item in obj)
        else:
            return obj

    def simulate(self, seed: Optional[int] = None) -> None:
        self.env.reset(seed=seed)
        num_steps: int = self.config["simulation"]["numSteps"]
        for _ in range(num_steps):
            all_actions_dic: dict[int, dict[str, Any]] = {}
            for agent_id in self.env.agent_ids:
                agent = self.env.agent_id2agent[agent_id]
                obs = self.env.get_observations(agent_id=agent_id)
                action_dic = agent.act(obs=obs)
                all_actions_dic[agent_id] = action_dic
            self.env.step(all_actions_dic=all_actions_dic)
        if self.env.logger is not None:
            self.env.logger.save()

    def register_classes(self, class_list: list[Type]) -> None:
        self.env.register_classes(class_list)
