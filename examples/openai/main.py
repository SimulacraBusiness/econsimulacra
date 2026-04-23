import asyncio
import json
import os
import pathlib
from pathlib import Path

from econsimulacra.envs import Environment
from econsimulacra.events import Event
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import SimulationSummarizer, Simulator

# export LOG_TXT_PATH="log_gpt-4o-mini.txt"


class SubsidyEvent(Event):
    def __init__(self, trigger, config) -> None:
        super().__init__(trigger=trigger, config=config)
        self.subsidy_amount: int = config["subsidyAmount"]

    def execute(self, env, log=None) -> None:
        cash_name = env.cash_name
        for agent_id in env.agent_ids:
            agent = env.agent_id2agent[agent_id]
            agent.inventory_dic[cash_name] += self.subsidy_amount


def conduct_simulation():
    config_dic_path: Path = Path(pathlib.Path(__file__).parent, "config.json")
    logger: DictLogger = DictLogger()
    simulator: Simulator = Simulator(
        config=config_dic_path,
        env_class=Environment,
        logger=logger,
        summarizer_class=SimulationSummarizer,
    )
    simulator.register_classes([SubsidyEvent])
    asyncio.run(simulator.simulate(seed=42))
    logs: list[dict] = logger.logs
    log_txt_path: Path = pathlib.Path(os.environ["LOG_TXT_PATH"])
    with open(log_txt_path, "w", encoding="utf-8") as f:
        for log in logs:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    conduct_simulation()
