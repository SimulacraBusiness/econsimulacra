from econsimulacra.log_analyses import (
    ActionCounter,
    AgentBehaviorStatsAnalyzer,
    AnalysisManager,
    ConsumerClusterAnalyzer,
    FollowerCounter,
    ItemSalesAnalyzer,
    PriceAnalyzer,
    RecordStore,
    StoreSalesAnalyzer,
    StressAnalyzer,
    TopicAnalyzer,
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
            StressAnalyzer(exclude_agent_ids=[30, 31]),
            TopicAnalyzer(exclude_agent_ids=[31]),
            TopicAnalyzer(is_inner_thought=True, exclude_agent_ids=[31]),
            ConsumerClusterAnalyzer(
                window_size=12,
                k_candidates=(2, 3, 4, 5, 6, 7),
                exclude_items=("Yen",),
                is_consumption=True,
                normalize=True,
            ),
        ]
    )
    stores = RecordStore(load_from_file("log_gpt-oss-20b.txt"))
    results = manager.run_all(
        stores,
        render_summary=True,
        figs_save_path="results",
    )
