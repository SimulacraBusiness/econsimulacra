import asyncio
import json
from .agents.base import Agent
from .logs.base import Logger
from .envs.base import Environment
from pathlib import Path
from typing import Any
from typing import Generic
from typing import Optional
from typing import Type
from typing import TypeVar

ObsT = TypeVar("ObsT")


class Simulator(Generic[ObsT]):
    def __init__(
        self,
        config: dict[str, Any] | Path,
        env_class: Type[Environment],
        logger: Optional[Logger] = None,
    ) -> None:
        self.config: dict[str, Any]
        if isinstance(config, Path):
            config_path: Path = config
            self.config = json.load(open(config_path, "r"))
        else:
            self.config = config
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

    async def simulate(
        self,
        seed: Optional[int] = None,
        parallel_batch_size: Optional[int] = None,
    ) -> None:
        def _chunked(seq: list[int], size: int) -> list[list[int]]:
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        parallel_batch_size = 1 if parallel_batch_size is None else parallel_batch_size
        self.env.reset(seed=seed)
        num_steps: int = self.config["simulation"]["numSteps"]
        for _ in range(num_steps):
            all_actions_dic: dict[int, dict[str, Any]] = {}

            async def _act_one(agent_id: int) -> tuple[int, dict[str, Any]]:
                agent: Agent = self.env.agent_id2agent[agent_id]
                obs: ObsT = self.env.get_observations(agent_id=agent_id)
                action_dic: dict[str, Any] = await agent.act(obs=obs)
                return agent_id, action_dic

            for batch in _chunked(self.env.agent_ids, parallel_batch_size):
                results: list[tuple[int, dict[str, Any]]] = await asyncio.gather(
                    *[_act_one(agent_id) for agent_id in batch]
                )
                all_actions_dic.update(dict(results))
            self.env.step(all_actions_dic=all_actions_dic)
        if self.env.logger is not None:
            self.env.logger.save()

    def register_classes(self, class_list: list[Type]) -> None:
        self.env.register_classes(class_list)
