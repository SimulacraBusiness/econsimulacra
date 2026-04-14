from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Generic, Optional, Type, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from tqdm import tqdm

from .agents.base import Agent
from .envs.base import Environment
from .envs.event import Event
from .envs.time_translator import TimeTranslator
from .llm_services.clients import LLMClient
from .llm_services.personas import PersonaBuilder
from .llm_services.prompts import PromptBuilder
from .logs.base import Logger

ObsT = TypeVar("ObsT")


class Simulator(Generic[ObsT]):
    """Simulator class.

    The Simulator class is responsible for running the simulation. Basic usage:

    >>> config_path = pathlib.Path("path/to/config.json")
    >>> logger = DictLogger()
    >>> simulator = Simulator(
    >>>     config=config_path,
    >>>     env_class=Environment,
    >>>     logger=logger,
    >>>     summarizer_class=SimulationSummarizer,
    >>> )
    >>> asyncio.run(simulator.simulate(seed=42))
    """

    def __init__(
        self,
        config: dict[str, Any] | Path,
        env_class: Type[Environment],
        logger: Optional[Logger] = None,
        summarizer_class: Optional[Type[SimulationSummarizer]] = None,
    ) -> None:
        """Initialization.

        Args:
            config (dict or Path): The configuration for the simulation
                See also econsimulacra.envs.base.Environment for the required and optional configuration fields.
            env_class (Type[Environment]): The environment class to use for the simulation. This must be a subclass of Environment.
            logger (Logger, optional): An optional logger instance for logging simulation data. If not provided, no logging will be performed.
            summarizer_class (Type[SimulationSummarizer], optional): An optional summarizer class for summarizing the simulation.
                This must be a subclass of SimulationSummarizer. If not provided, no summarization will be performed.
        """
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
        """Execute the full simulation loop asynchronously.

        Args:
            - seed (int, optional): Random seed.

        Note:
            This method is the main entry point for running a simulation episode.
            It resets the environment, iteratively collects actions from all agents,
            applies the joint action to the environment at each step, and finalizes
            optional logging and summarization at the end of the run.

            The simulation proceeds as follows:

            1. The environment is reset with the given random seed.
            2. If a summarizer is configured, the simulation start is recorded.
            3. For each simulation step:
            - observations are generated for all agents,
            - each agent asynchronously computes its action via :meth:`act`,
            - agents are evaluated in chunks determined by ``parallel_batch_size``,
                so that multiple agents can act concurrently without launching all
                coroutines at once,
            - list-valued actions are recursively converted into tuples for
                downstream consistency and hashability,
            - the resulting joint action dictionary is passed to
                :meth:`self.env.step`.
            4. After all steps are completed, the logger is saved if present.
            5. If a summarizer is configured, the simulation end is recorded.

            This method is intentionally asynchronous because agent decision-making
            may involve I/O-bound or high-latency components such as LLM inference,
            API calls, or remote services. By using :func:`asyncio.gather`, the
            simulator can evaluate multiple agents concurrently within each batch,
            improving throughput while still preserving step-level synchronization:
            all actions for a step are collected before the environment advances.

            Concurrency Model:
                Agent actions are computed concurrently within each batch, but the
                environment transition itself is performed once per step after all
                actions have been collected. Therefore, this method implements
                synchronous environment stepping with asynchronous per-agent action
                generation.
        """

        def _chunked(seq: list[int], size: int) -> list[list[int]]:
            return [seq[i : i + size] for i in range(0, len(seq), size)]

        parallel_batch_size = (
            1 if self.parallel_batch_size is None else self.parallel_batch_size
        )
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
                    item_amount: int = int(agent.get_item_amount(item_name=item_name))
                    agent_branch.add(
                        f"[green]{item_name}[/green]: {item_amount}"
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
        events: list[Event] = self.env.event_manager.events
        if len(events) > 0:
            event_branch: Tree = tree.add("[green]Events[/green]")
            for event in self.env.event_manager.events:
                event_branch_: Tree = event_branch.add(
                    f"[green]{event.__class__.__name__}[/green]"
                )
                if event.trigger.at is not None:
                    event_branch_.add(f"[green]Trigger at[/green]: {event.trigger.at}")
                if event.trigger.every is not None:
                    event_branch_.add(
                        f"[green]Trigger every[/green]: {event.trigger.every}"
                    )
                if len(event.trigger.logs) > 0:
                    event_branch_.add(
                        f"[green]Trigger with logs[/green]: {[log.__name__ for log in event.trigger.logs]}"
                    )
                if event.trigger.between is not None:
                    event_branch_.add(
                        f"[green]Trigger between[/green]: {event.trigger.between}"
                    )
                if event.trigger.probability is not None:
                    event_branch_.add(
                        f"[green]Trigger probability[/green]: {event.trigger.probability}"
                    )
        if len(self.env.service_dic) > 0:
            service_branch: Tree = tree.add("[green]Environment Services[/green]")
            for _, service in self.env.service_dic.items():
                if isinstance(service, LLMClient):
                    llm_client_branch: Tree = service_branch.add(
                        f"[green]LLM Client: {service.__class__.__name__}[/green]"
                    )
                    llm_client_branch.add(
                        f"[green]Model Name[/green]: {service.model_name}"
                    )
                    max_concurrent_generations: Optional[int] = getattr(
                        service, "max_concurrent_generations", None
                    )
                    if max_concurrent_generations is not None:
                        llm_client_branch.add(
                            f"[green]Max Concurrent Generations[/green]: {max_concurrent_generations}"
                        )
                elif isinstance(service, PersonaBuilder):
                    persona_builder_branch: Tree = service_branch.add(
                        f"[green]Persona Builder: {service.__class__.__name__}[/green]"
                    )
                    max_magnitude: Optional[int] = getattr(
                        service, "max_magnitude", None
                    )
                    if max_magnitude is not None:
                        persona_builder_branch.add(
                            f"[green]Max Magnitude[/green]: {max_magnitude}"
                        )
                elif isinstance(service, PromptBuilder):
                    service_branch.add(
                        f"[green]Prompt Builder: {service.__class__.__name__}[/green]"
                    )
                elif isinstance(service, TimeTranslator):
                    time_translator_branch: Tree = service_branch.add(
                        f"[green]Time Translator: {service.__class__.__name__}[/green]"
                    )
                    start_datetime: Optional[Any] = getattr(
                        service, "start_datetime", None
                    )
                    end_datetime: Optional[Any] = getattr(service, "end_datetime", None)
                    time_delta: Optional[Any] = getattr(service, "time_delta", None)
                    if start_datetime is not None:
                        time_translator_branch.add(
                            f"[green]Start Datetime[/green]: {str(start_datetime)}"
                        )
                    if end_datetime is not None:
                        time_translator_branch.add(
                            f"[green]End Datetime[/green]: {str(end_datetime)}"
                        )
                    if time_delta is not None:
                        time_translator_branch.add(
                            f"[green]Time Delta[/green]: {str(time_delta)}"
                        )
                else:
                    service_branch.add(
                        f"[green]Service: ({service.__class__.__name__})[/green]"
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
        last_prompt: Optional[str] = getattr(agent, "last_prompt", None)
        if last_prompt is not None:
            if last_prompt != "":
                console.print(
                    Panel(
                        last_prompt,
                        title=f"[bold green]Agent {example_agent_id}'s Last Prompt[/bold green]",
                    )
                )
