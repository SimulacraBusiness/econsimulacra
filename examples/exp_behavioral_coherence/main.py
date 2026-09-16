from __future__ import annotations

import argparse
import asyncio
import copy
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from intervention_agents import DiscountRetailer

from econsimulacra.envs import Environment
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import SimulationSummarizer, Simulator

EXPERIMENT_DIR = Path(__file__).resolve().parent


def _resolve_experiment_paths(config: dict[str, Any], config_path: Path) -> None:
    """Resolve experiment-owned resource paths relative to a configuration.

    Args:
        config: Mutable simulation configuration.
        config_path: Source JSON path whose parent anchors relative resources.

    Returns:
        None.

    Note:
        EconSimulacra receives the resulting absolute path, making candidate
        tweet loading independent of the process working directory.
    """
    intervention = (
        config.get("Household", {})
        .get("socialRule", {})
        .get("candidateTweetIntervention")
    )
    if not isinstance(intervention, dict) or "path" not in intervention:
        return
    resource_path = Path(intervention["path"])
    if not resource_path.is_absolute():
        intervention["path"] = str((config_path.parent / resource_path).resolve())


def conduct_simulation(
    config_path: Path,
    condition: str,
    model_label: str,
    seed_start: int,
    num_seeds: int,
    summarize: bool = False,
) -> None:
    """Run one configured condition for a contiguous seed range.

    Args:
        config_path: JSON simulation configuration.
        condition: Output-directory label for the intervention condition.
        model_label: Output-directory label for the evaluated agent model.
        seed_start: First pseudo-random seed.
        num_seeds: Number of consecutive seeds to simulate.
        summarize: Whether to print the rich configuration summary per run.

    Returns:
        None.

    Note:
        A deep copy is passed to every Simulator so mutable runtime state cannot
        leak between repetitions.
    """
    config_path = config_path.resolve()
    config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
    _resolve_experiment_paths(config=config, config_path=config_path)
    output_dir = EXPERIMENT_DIR / "logs" / condition / model_label
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in range(seed_start, seed_start + num_seeds):
        logger = DictLogger()
        simulator: Simulator[Any] = Simulator(
            config=copy.deepcopy(config),
            env_class=Environment,
            logger=logger,
            summarizer_class=SimulationSummarizer if summarize else None,
        )
        simulator.register_classes([DiscountRetailer])
        asyncio.run(simulator.simulate(seed=seed))
        log_path = output_dir / f"{seed}.txt"
        with log_path.open("w", encoding="utf-8") as file:
            for log in logger.logs:
                file.write(json.dumps(log, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line options for a simulation batch.

    Args:
        argv: Optional explicit argument sequence used by tests.

    Returns:
        Parsed command-line namespace.

    Note:
        The non-LLM baseline is the default so routine validation never starts a
        vLLM server unintentionally.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=EXPERIMENT_DIR / "config_baseline.json",
    )
    parser.add_argument("--condition", default="baseline")
    parser.add_argument("--model-label", default="rule-based")
    parser.add_argument("--seed-start", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    if args.num_seeds <= 0:
        parser.error("--num-seeds must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Parse arguments and execute the requested simulation batch.

    Args:
        argv: Optional explicit argument sequence used by tests.

    Returns:
        None.

    Note:
        Keeping argument parsing separate makes path and default behavior easy
        to test without launching a simulation.
    """
    args = parse_args(argv)
    conduct_simulation(
        config_path=args.config,
        condition=args.condition,
        model_label=args.model_label,
        seed_start=args.seed_start,
        num_seeds=args.num_seeds,
        summarize=args.summarize,
    )


if __name__ == "__main__":
    main()
