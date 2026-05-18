from econsimulacra.log_analyses import (
    ActionCounter,
    AgentBehaviorStatsAnalyzer,
    AnalysisManager,
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
            AgentBehaviorStatsAnalyzer(),
            FollowerCounter(),
            StoreSalesAnalyzer(),
            ItemSalesAnalyzer(),
            PriceAnalyzer(),
            StressAnalyzer(),
            TopicAnalyzer(exclude_agent_ids=[31]),
            TopicAnalyzer(is_inner_thought=True, exclude_agent_ids=[31]),
        ]
    )
    stores = RecordStore(load_from_file("log_gpt-oss-20b.txt"))
    results = manager.run_all(
        stores,
        render_summary=True,
        figs_save_path="results",
    )
