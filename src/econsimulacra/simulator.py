import asyncio
import json
from .agents.base import Agent
from .logs.base import Logger
from .llm_services import LLMClient
from .envs.base import Environment
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from tqdm import tqdm
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
        print_summary: bool = True,
    ) -> None:
        def _chunked(seq: list[int], size: int) -> list[list[int]]:
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        parallel_batch_size = 1 if parallel_batch_size is None else parallel_batch_size
        self.env.reset(seed=seed)
        if print_summary:
            self.summarize_start(self.env)
        num_steps: int = self.config["simulation"]["numSteps"]
        for _ in tqdm(
            range(num_steps), desc="Simulating", unit="step", ncols=80, leave=True
        ):
            all_actions_dic: dict[int, dict[str, Any]] = {}

            async def _act_one(agent_id: int) -> tuple[int, dict[str, Any]]:
                agent: Agent = self.env.agent_id2agent[agent_id]
                obs: ObsT = self.env.get_observations(agent_id=agent_id)
                action_dic: dict[str, Any] = await agent.act(obs=obs)
                action_dic = self._convert_list_to_tuple(action_dic)
                return agent_id, action_dic

            for batch in _chunked(self.env.agent_ids, parallel_batch_size):
                results: list[tuple[int, dict[str, Any]]] = await asyncio.gather(
                    *[_act_one(agent_id) for agent_id in batch]
                )
                all_actions_dic.update(dict(results))
            self.env.step(all_actions_dic=all_actions_dic)
        if self.env.logger is not None:
            self.env.logger.save()
        if print_summary:
            print()
            self.summarize_end(self.env)

    def register_classes(self, class_list: list[Type]) -> None:
        self.env.register_classes(class_list)

    def summarize_start(self, env: Environment) -> None:
        console: Console = Console()
        tree: Tree = Tree("Simulation Configuration")
        tree.add(f"[green]Seed[/green]: {env.seed}")
        tree.add(
            f"[green]Number of Steps[/green]: {self.config['simulation']['numSteps']}"
        )
        tree.add(f"[green]Grid Space[/green]: {env.grid_space.space_size}")
        tree.add(f"[green]Follow Cap[/green]: {env.social_network.follow_cap}")
        items_branch: Tree = tree.add("[green]Items[/green]")
        for item_name, item in env.item_name2item.items():
            item_branch: Tree = items_branch.add(f"[green]{item_name}[/green]")
            item_branch.add(
                f"[green]Total Amount[/green]: {env.get_total_amount(item_name=item_name):.1f}"
            )
            if item_name == env.cash_name:
                item_branch.add("[green]Cash[/green]")
            else:
                item_branch.add(
                    f"[green]Initial Price[/green]: {item.price:.1f} {env.cash_name}"
                )
        agents_branch: Tree = tree.add("[green]Agents[/green]")
        agents_branch.add(
            f"[green]Number of Households[/green]: {len(env.household_ids)}"
        )
        for agent_id, agent in env.agent_id2agent.items():
            if agent_id not in env.household_ids:
                agent_branch: Tree = agents_branch.add(
                    f"[green]Agent {agent_id}[/green]"
                )
                agent_branch.add(f"[green]Name[/green]: {agent.agent_name}")
                for item_name in env.item_name2item.keys():
                    agent_branch.add(
                        f"[green]{item_name}[/green]: {agent.get_item_amount(item_name=item_name):.1f}"
                    )
                agent_branch.add(
                    f"[green]Receive Rich Info[/green]: {agent.is_rich_info_allowed}"
                )
                agent_branch.add(
                    f"[green]Provide Info for All Agents[/green]: {agent.provide_info4all_agents()}"
                )
                agent_branch.add(
                    f"[green]Provide Info for Co-Located Agents[/green]: {agent.provide_info4co_located_agents()}"
                )
                agent_branch.add(
                    f"[green]Provide Info for Allowed Agents[/green]: {agent.provide_info4allowed_agents()}"
                )
        if "llmClient" in env.service_dic:
            client: LLMClient = env.service_dic["llmClient"]
            llm_branch: Tree = tree.add("[green]LLM Client[/green]")
            llm_branch.add(f"[green]Model[/green]: {client.config['model_name']}")
        console.print(Panel(tree, title="[bold green]Summary[/bold green]"))

    def summarize_end(self, env: Environment) -> None:
        console: Console = Console()
        table: Table = Table(title="Invalid Actions", show_lines=True)
        table.add_column("Action Type", justify="center", style="cyan", no_wrap=True)
        table.add_column("Number", justify="center", style="magenta")
        for action_type, count in env.invalid_action_dic.items():
            table.add_row(action_type, str(count))
        console.print(table)
