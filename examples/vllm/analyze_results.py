from econsimulacra.log_analyses import (
    ActionCounter,
    AgentBehaviorStatsAnalyzer,
    AnalysisManager,
    ConsumerClusterAnalyzer,
    FollowerCounter,
    ItemSalesAnalyzer,
    MoveDistanceAnalyzer,
    PriceAnalyzer,
    RecordStore,
    StoreSalesAnalyzer,
    StressActionAnalyzer,
    StressAnalyzer,
    load_from_file,
)

if __name__ == "__main__":
    manager = AnalysisManager(
        analyzers=[
            ActionCounter(),
            AgentBehaviorStatsAnalyzer(exclude_agent_ids=[30, 31]),
            FollowerCounter(),
            StoreSalesAnalyzer(),
            ItemSalesAnalyzer(),
            PriceAnalyzer(),
            StressActionAnalyzer(exclude_agent_ids=[30, 31]),
            StressAnalyzer(exclude_agent_ids=[30, 31]),
            ConsumerClusterAnalyzer(
                window_size=24,
                k_candidates=(2, 4, 6, 8, 10),
                exclude_items=("Yen",),
                is_consumption=True,
                normalize=True,
            ),
            MoveDistanceAnalyzer(window_size=4),
        ]
    )
    store = RecordStore(load_from_file("log_gpt-oss-120b/42.txt"))
    results = manager.run_all(
        store,
        render_summary=True,
        figs_save_path="results/gpt-oss-120b/42",
    )
