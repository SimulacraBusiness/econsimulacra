from __future__ import annotations
import asyncio
import json
from .agents.base import Agent
from .envs.base import Environment
from .envs.time_translator import TimeTranslator
from .llm_services.clients import LLMClient
from .llm_services.personas import PersonaBuilder
from .llm_services.prompts import PromptBuilder
from .logs.base import Logger
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
        summarizer_class: Optional[Type[SimulationSummarizer]] = None,
    ) -> None:
        self.config: dict[str, Any]
        if isinstance(config, Path):
            config_path: Path = config
            self.config = json.load(open(config_path, "r"))
        else:
            self.config = config
        self.config = self._convert_list_to_tuple(self.config)
        self.parallel_batch_size: Optional[int] = self.config["simulation"].get(
            "parallelBatchSize"
        )
        self.env: Environment = env_class(config=self.config, logger=logger)
        self.summarizer: Optional[SimulationSummarizer] = (
            summarizer_class(self.env) if summarizer_class is not None else None
        )

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
    ) -> None:
        def _chunked(seq: list[int], size: int) -> list[list[int]]:
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        parallel_batch_size = 1 if self.parallel_batch_size is None else self.parallel_batch_size
        self.env.reset(seed=seed)
        if self.summarizer is not None:
            self.summarizer.summarize_start()
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
        if self.summarizer is not None:
            self.summarizer.summarize_end()

    def register_classes(self, class_list: list[Type]) -> None:
        self.env.register_classes(class_list)


class SimulationSummarizer:
    def __init__(self, env: Environment) -> None:
        self.env: Environment = env

    def summarize_start(self) -> None:
        console: Console = Console()
        tree: Tree = Tree("Simulation Configuration")
        tree.add(f"[green]Seed[/green]: {self.env.seed}")
        if "parallelBatchSize" in self.env.config["simulation"]:
            tree.add(
                f"[green]Parallel Batch Size[/green]: {self.env.config['simulation']['parallelBatchSize']}"
            )
        tree.add(
            f"[green]Number of Steps[/green]: {self.env.config['simulation']['numSteps']}"
        )
        tree.add(f"[green]Grid Space[/green]: {self.env.grid_space.get_space_size()}")
        social_network_branch: Tree = tree.add("[green]Social Network[/green]")
        social_network_branch.add(
            f"[green]Follow Cap[/green]: {self.env.social_network.follow_cap}"
        )
        recsys = self.env.social_network.rec_sys
        recsys_branch: Tree = social_network_branch.add(
            f"[green]Recommender System: {recsys.__class__.__name__}[/green]"
        )
        recsys_branch.add(f"[green]Max Recommendations[/green]: {recsys.max_recs}")
        recsys_branch.add(
            f"[green]Randomized Recommendations[/green]: {recsys.is_randomized}"
        )
        if recsys.is_randomized:
            recsys_branch.add(f"[green]Temperature[/green]: {recsys.temperature}")
        items_branch: Tree = tree.add("[green]Items[/green]")
        for item_name, item in self.env.item_name2item.items():
            item_branch: Tree = items_branch.add(f"[green]{item_name}[/green]")
            item_branch.add(
                f"[green]Total Amount[/green]: {self.env.get_total_amount(item_name=item_name):.1f}"
            )
            if item_name == self.env.cash_name:
                item_branch.add("[green]Cash[/green]")
            else:
                item_branch.add(
                    f"[green]Initial Price[/green]: {item.price:.1f} {self.env.cash_name}"
                )
        agents_branch: Tree = tree.add("[green]Agents[/green]")
        agents_branch.add(
            f"[green]Number of Households[/green]: {len(self.env.household_ids)}"
        )
        for agent_id, agent in self.env.agent_id2agent.items():
            if agent_id not in self.env.household_ids:
                agent_branch: Tree = agents_branch.add(
                    f"[green]Agent {agent_id}[/green]"
                )
                agent_branch.add(f"[green]Name[/green]: {agent.agent_name}")
                for item_name in self.env.item_name2item.keys():
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
        if len(self.env.service_dic) > 0:
            service_branch: Tree = tree.add("[green]Environment Services[/green]")
            for service_name, service in self.env.service_dic.items():
                if isinstance(service, LLMClient):
                    llm_client_branch: Tree = service_branch.add(
                        f"[green]LLM Client: {service.__class__.__name__}[/green]"
                    )
                    llm_client_branch.add(
                        f"[green]Model Name[/green]: {service.model_name}"
                    )
                    if hasattr(service, "max_concurrent_generations"):
                        llm_client_branch.add(
                            f"[green]Max Concurrent Generations[/green]: {service.max_concurrent_generations}"
                        )
                elif isinstance(service, PersonaBuilder):
                    persona_builder_branch: Tree = service_branch.add(
                        f"[green]Persona Builder: {service.__class__.__name__}[/green]"
                    )
                    if hasattr(service, "max_magnitude"):
                        persona_builder_branch.add(
                            f"[green]Max Magnitude[/green]: {service.max_magnitude}"
                        )
                elif isinstance(service, PromptBuilder):
                    service_branch.add(
                        f"[green]Prompt Builder: {service.__class__.__name__}[/green]"
                    )
                elif isinstance(service, TimeTranslator):
                    time_translator_branch: Tree = service_branch.add(
                        f"[green]Time Translator: {service.__class__.__name__}[/green]"
                    )
                    if hasattr(service, "start_datetime"):
                        time_translator_branch.add(
                            f"[green]Start Datetime[/green]: {str(service.start_datetime)}"
                        )
                    if hasattr(service, "end_datetime"):
                        time_translator_branch.add(
                            f"[green]End Datetime[/green]: {str(service.end_datetime)}"
                        )
                    if hasattr(service, "time_delta"):
                        time_translator_branch.add(
                            f"[green]Time Delta[/green]: {str(service.time_delta)}"
                        )
        print()
        console.print(Panel(tree, title="[bold green]Summary[/bold green]"))
        print()

    def summarize_end(self) -> None:
        console: Console = Console()
        table: Table = Table(title="Invalid Actions", show_lines=True)
        table.add_column("Action Type", justify="center", style="cyan", no_wrap=True)
        table.add_column("Number", justify="center", style="magenta")
        for action_type, count in self.env.invalid_action_dic.items():
            table.add_row(action_type, str(count))
        print()
        console.print(table)
        print()
        example_agent_id: int = self.env.household_ids[0]
        agent: Agent = self.env.agent_id2agent[example_agent_id]
        if hasattr(agent, "last_prompt") and agent.last_prompt != "":
            console.print(
                Panel(
                    agent.last_prompt,
                    title=f"[bold green]Agent {example_agent_id}'s Last Prompt[/bold green]",
                )
            )
