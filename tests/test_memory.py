from econsimulacra.logs import AgentGenerationLog, ConsumptionLog
from econsimulacra.memory import MemoryHandler, StressAwareSummarizer, StressCalculator


class TestMemoryHandler:
    config = {
        "type": "MemoryHandler",
        "memoryLength": 10,
        "memorySummarizer": {
            "type": "StressAwareSummarizer",
            "stressCalculator": {
                "type": "StressCalculator",
                "item2Weight": {"Yen": 0.0, "Rice": 10.0, "Apple": 1.0},
                "maxMagnitude": 100,
                "targetConsumptionQuantity": 15,
                "windowSizeForConsumption": 20,
                "timeDecayForConsumption": 0.9,
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
            inventory_dic={
                "Yen": 10000,
                "Rice": 10,
                "Apple": 10,
            },
        )
        memory_handler.update(log=log0)
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
        assert d["move_history"] == "You have no movement history."
        assert d["consumption_history"] == (
            "You have consumed Rice x 1 at time 1, Apple x 5 at time 2. "
            "Your stress level from this consumption is 6 out of 100. "
            "Acceptable consumption level."
        )
