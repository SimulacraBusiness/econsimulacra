import asyncio
from econsimulacra.envs import Environment
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import Simulator
from econsimulacra.simulator import SimulationSummarizer
import json
import os
import pathlib
from pathlib import Path
# export LOG_TXT_PATH="log.txt"


def conduct_simulation():
    config_dic_path: Path = Path(pathlib.Path(__file__).parent, "config.json")
    logger: DictLogger = DictLogger()
    simulator: Simulator = Simulator(
        config=config_dic_path,
        env_class=Environment,
        logger=logger,
        summarizer_class=SimulationSummarizer,
    )
    asyncio.run(simulator.simulate(seed=42, parallel_batch_size=2))
    logs: list[dict] = logger.logs
    log_txt_path: Path = pathlib.Path(os.environ["LOG_TXT_PATH"])
    with open(log_txt_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    conduct_simulation()
