from econsimulacra.logs import AgentGenerationLog
from econsimulacra.logs import MoveLog
from econsimulacra.logs import DictLogger


class DummyLogger(DictLogger):
    def __init__(self) -> None:
        super().__init__()
        self._dispatch_table = {MoveLog: self._process_move_log}

    def _process_move_log(self, log: MoveLog) -> None:
        d = log.to_dict()
        d["tag"] = "movement"
        self.logs.append(d)


class TestDictLogger:
    def test_process_logs(self) -> None:
        logger = DictLogger()
        log1 = AgentGenerationLog(
            time=0,
            time_step=0,
            agent_id=1,
            agent_type="DummyAgent",
            agent_name="AgentA",
            inventory_dic={"Cash": 100},
        )
        log2 = MoveLog(time=1, time_step=1, agent_id=1, old_pos=(0, 0), new_pos=(1, 1))
        log1.read_and_write(logger)
        log2.read_and_write(logger)
        logger.process_logs()
        assert logger.logs == [
            {
                "type": "agent_generation",
                "time": 0,
                "time_step": 0,
                "agent_id": 1,
                "agent_type": "DummyAgent",
                "agent_name": "AgentA",
                "inventory_Cash": 100,
            },
            {
                "type": "move",
                "time": 1,
                "time_step": 1,
                "agent_id": 1,
                "old_pos": (0, 0),
                "new_pos": (1, 1),
            },
        ]
        logger = DummyLogger()
        log1.read_and_write(logger)
        log2.read_and_write(logger)
        logger.process_logs()
        assert logger.logs == [
            {
                "type": "agent_generation",
                "time": 0,
                "time_step": 0,
                "agent_id": 1,
                "agent_type": "DummyAgent",
                "agent_name": "AgentA",
                "inventory_Cash": 100,
            },
            {
                "type": "move",
                "time": 1,
                "time_step": 1,
                "agent_id": 1,
                "old_pos": (0, 0),
                "new_pos": (1, 1),
                "tag": "movement",
            },
        ]
