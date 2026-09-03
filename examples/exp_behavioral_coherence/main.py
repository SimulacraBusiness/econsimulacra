from __future__ import annotations

import asyncio
import json
import os
import pathlib
from pathlib import Path

from econsimulacra.envs import Environment
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import SimulationSummarizer, Simulator



def conduct_simulation():
    config_dic_path: Path = pathlib.Path(__file__).parent / "config_baseline_llm.json"
    for seed in range(42, 52):
        logger: DictLogger = DictLogger()
        simulator: Simulator = Simulator(
            config=config_dic_path,
            env_class=Environment,
            logger=logger,
            summarizer_class=SimulationSummarizer,
        )
        asyncio.run(simulator.simulate(seed=seed))
        logs: list[dict] = logger.logs
        llm_type: str = "gpt-oss-20b"
        log_txt_path: Path = pathlib.Path(f"logs/baseline/{llm_type}/{seed}.txt")
        with open(log_txt_path, "w", encoding="utf-8") as f:
            for log in logs:
                f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    conduct_simulation()
