from econsimulacra.logs import (
    AgentGenerationLog,
    ConsumptionLog,
    MoveLog,
    SpaceAssignLog,
    StateEvaluationLog
)
from econsimulacra.memory import MemoryHandler, StressAwareSummarizer, StressCalculator


class TestMemoryHandler:
    config = {
        "type": "MemoryHandler",
        "memoryLength": 10,
        "memorySummarizer": {
            "type": "StressAwareSummarizer",
            "stressCalculator": {
                "type": "StressCalculator",
                "stressTypes": ["consumption_history", "move_history", "state_evaluation_history"],
                "item2Weight": {"Yen": 0.0, "Rice": 10.0, "Apple": 1.0},
                "maxMagnitude": 100,
                "targetConsumptionQuantity": 15,
                "windowSizeForConsumption": 20,
                "timeDecayForConsumption": 0.9,
                "targetMoveDistance": 10.0,
                "windowSizeForMove": 20,
                "timeDecayForMove": 0.9,
                "homeComfortWeight": 0.2,
                "targetBuyingPower": 80.0,
                "targetRelativeWealth": -0.2,
                "targetWealthGrowth": 0.1,
                "windowSizeForStateEvaluation": 20,
                "buyingPowerWeight": 1.5,
                "relativeWealthWeight": 0.9,
                "wealthDrawdownWeight": 0.3,
                "toleranceThresholdForStress": 20,
            },
        },
    }

    def test_init(self):
        memory_handler = MemoryHandler(self.config)
        assert memory_handler.memory_length == 10
        assert isinstance(memory_handler.memory_summarizer, StressAwareSummarizer)
        summarizer = memory_handler.memory_summarizer
        stress_calculator = summarizer.stress_calculator
        assert isinstance(summarizer.stress_calculator, StressCalculator)
        assert stress_calculator.max_magnitude == 100
        assert stress_calculator.target_consumption_quantity == 15
        assert stress_calculator.window_size_for_consumption == 20
        assert stress_calculator.time_decay_for_consumption == 0.9
        assert stress_calculator.tolerance_threshold_for_stress == 20
        assert stress_calculator.target_move_distance == 10.0
        assert stress_calculator.window_size_for_move == 20
        assert stress_calculator.time_decay_for_move == 0.9
        assert stress_calculator.home_comfort_weight == 0.2
        assert stress_calculator.target_buying_power == 80.0
        assert stress_calculator.target_relative_wealth == -0.2
        assert stress_calculator.target_wealth_growth == 0.1
        assert stress_calculator.window_size_for_state_evaluation == 20
        assert stress_calculator.buying_power_weight == 1.5
        assert stress_calculator.relative_wealth_weight == 0.9
        assert stress_calculator.wealth_drawdown_weight == 0.3
        assert stress_calculator.tolerance_threshold_for_stress == 20

    def test_summarize_memory(self):
        memory_handler = MemoryHandler(self.config)
        assert isinstance(memory_handler.memory_summarizer, StressAwareSummarizer)
        summarizer = memory_handler.memory_summarizer
        log0 = AgentGenerationLog(
            time=0,
            time_step=0,
            agent_id=1,
            agent_type="Dummy",
            agent_name="Dummy",
            wealth=10000,
            inventory_dic={
                "Yen": 10000,
                "Rice": 10,
                "Apple": 10,
            },
            persona_dic={"trait1": "value1"},
        )
        log00 = SpaceAssignLog(
            agent_id=1,
            pos=(0, 0),
        )
        memory_handler.update(log=log0)
        memory_handler.update(log=log00)
        assert memory_handler.current_time == 0
        assert memory_handler.current_time_step == 0
        assert summarizer.current_time == 0
        assert summarizer.current_time_step == 0
        log1 = ConsumptionLog(
            time=1, time_step=1, agent_id=1, item_name="Rice", item_amount=1
        )
        log2 = ConsumptionLog(
            time=2, time_step=2, agent_id=1, item_name="Apple", item_amount=5
        )
        memory_handler.update(log=log1)
        memory_handler.update(log=log2)
        assert memory_handler.current_time == 2
        assert memory_handler.current_time_step == 2
        assert summarizer.current_time == 2
        assert summarizer.current_time_step == 2
        assert len(memory_handler.agent_id2memory) == 1
        consumption_history = memory_handler.agent_id2memory[1].consumption_history
        calculator = summarizer.stress_calculator
        score, _ = calculator._calc_stress_from_consumption_history(
            history=consumption_history,
        )
        assert score == int((15 - (10 * 1 * 0.9 + 1 * 5)) / 15 * 100)
        d = memory_handler.get_memory(agent_id=1)
        assert (
            d["move_history"]
            == "You have moved to (0, 0). Your stress level from this move is 0 out of 100. "
        )
        assert d["consumption_history"] == (
            "You have consumed Rice x 1 at time 1, Apple x 5 at time 2. "
            "Your stress level from this consumption is 6 out of 100. "
            "Acceptable consumption level."
        )
        log3 = MoveLog(
            time=3,
            time_step=3,
            agent_id=1,
            old_pos=(0, 0),
            new_pos=(0, 1),
            init_pos=(0, 0),
        )
        memory_handler.update(log=log3)
        d = memory_handler.get_memory(agent_id=1)
        assert d["move_history"] == (
            "You have moved to (0, 0) -> (0, 1). "
            "Your stress level from this move is 90 out of 100. "
            "You have not moved enough. (distance: 1.0, target: 10.0)"
        )
        log4 = MoveLog(
            time=4,
            time_step=4,
            agent_id=1,
            old_pos=(0, 1),
            new_pos=(0, 0),
            init_pos=(0, 0),
        )
        memory_handler.update(log=log4)
        d = memory_handler.get_memory(agent_id=1)
        assert d["move_history"] == (
            "You have moved to (0, 0) -> (0, 1) -> (0, 0). "
            "Your stress level from this move is 64 out of 100. "
            "You have not moved enough. (distance: 1.9, target: 10.0) "
            "However, being at home makes you feel somewhat comfortable."
        )
        log5 = StateEvaluationLog(
            time=5,
            time_step=5,
            agent_id=1,
            wealth=9000,
            relative_wealth=-0.3,
            buying_power=70.0,
            inventory_dic={
                "Yen": 9000,
                "Rice": 9,
                "Apple": 5,
            },
            persona_dic={"trait1": "value1"},
        )
        memory_handler.update(log=log5)
        d = memory_handler.get_memory(agent_id=1)
        assert d["state_evaluation_history"] == (
            "Your state evaluations are Wealth: 10000 at time 0; Wealth: 9000 at time 5. "
            "Your stress level from this state evaluation is 29 out of 100. "
            "You cannot buy enough goods. (buying power: 70.00, target: 80.00) "
            "You have less wealth than others. (relative wealth: -0.30, target: -0.20) "
            "Your wealth has recently decreased. (wealth change: -1000.00)"
        )
