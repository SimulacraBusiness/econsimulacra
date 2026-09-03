from econsimulacra.logs import (
    AgentGenerationLog,
    DictLogger,
    MobilityConsumptionLog,
    MoveLog,
    MovementInterruptedLog,
)


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
            wealth=100,
            inventory_dic={"Cash": 100},
            persona_dic={"trait1": "value1"},
        )
        log2 = MoveLog(
            time=1,
            time_step=1,
            agent_id=1,
            old_pos=(0, 0),
            new_pos=(1, 1),
            new_pos_description=None,
            init_pos=(0, 0),
        )
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
                "wealth": 100,
                "inventory_Cash": 100,
                "persona_trait1": "value1",
            },
            {
                "type": "move",
                "time": 1,
                "time_step": 1,
                "agent_id": 1,
                "old_pos": (0, 0),
                "new_pos": (1, 1),
                "new_pos_description": None,
                "init_pos": (0, 0),
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
                "wealth": 100,
                "inventory_Cash": 100,
                "persona_trait1": "value1",
            },
            {
                "type": "move",
                "time": 1,
                "time_step": 1,
                "agent_id": 1,
                "old_pos": (0, 0),
                "new_pos": (1, 1),
                "new_pos_description": None,
                "init_pos": (0, 0),
                "tag": "movement",
            },
        ]

    def test_mobility_logs_serialize_isolated_values(self) -> None:
        """Test movement metadata and dedicated mobility event serialization.

        Args:
            None.

        Returns:
            None.

        Note:
            Mutating constructor dictionaries after logging does not alter events.
        """
        consumption = {"Electricity": 1.0}
        missing_items = {"Electricity": 0.1}
        move_log = MoveLog(
            time=1,
            time_step=1,
            agent_id=1,
            old_pos=(0, 0),
            new_pos=(10, 0),
            new_pos_description=None,
            init_pos=(0, 0),
            mobility_name="ElectricCar",
            moved_cells=10,
        )
        consumption_log = MobilityConsumptionLog(1, 1, 1, "ElectricCar", consumption)
        interruption_log = MovementInterruptedLog(
            2,
            2,
            1,
            (15, 0),
            "ElectricCar",
            "mobility_unavailable",
            missing_items,
        )
        consumption["Electricity"] = 5
        missing_items["Electricity"] = 5

        assert move_log.to_dict()["mobility_name"] == "ElectricCar"
        assert move_log.to_dict()["moved_cells"] == 10
        assert consumption_log.consumption == {"Electricity": 1.0}
        assert interruption_log.missing_items == {"Electricity": 0.1}
