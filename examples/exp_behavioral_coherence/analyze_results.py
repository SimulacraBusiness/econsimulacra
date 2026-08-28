from econsimulacra.log_analyses import (
    ActionCounter,
    AgentBehaviorStatsAnalyzer,
    AnalysisManager,
    ConsumerClusterAnalyzer,
    MoveDistanceAnalyzer,
    PriceAnalyzer,
    RecordStore,
    StoreSalesAnalyzer,
    StressActionAnalyzer,
    StressAnalyzer,
    TemporalDynamicsAnalyzer,
    load_from_file,
)

if __name__ == "__main__":
    manager = AnalysisManager(
        analyzers=[
            ActionCounter(),
            AgentBehaviorStatsAnalyzer(exclude_agent_ids=[1,2,3]),
            StoreSalesAnalyzer(),
            PriceAnalyzer(),
            StressActionAnalyzer(exclude_agent_ids=[1,2,3]),
            StressAnalyzer(exclude_agent_ids=[1,2,3]),
            ConsumerClusterAnalyzer(
                window_size=24,
                k_candidates=(2, 4, 6, 8, 10),
                exclude_items=("Yen",),
                is_consumption=True,
                normalize=True,
            ),
            MoveDistanceAnalyzer(window_size=4),
            TemporalDynamicsAnalyzer(exclude_agent_ids=[1,2,3]),
        ]
    )
    store = RecordStore(load_from_file("log_baseline.txt"))
    results = manager.run_all(
        store,
        render_summary=True,
        figs_save_path="baseline",
    )
