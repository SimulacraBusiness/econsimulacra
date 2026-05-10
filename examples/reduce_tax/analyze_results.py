from econsimulacra.log_analyses import (
    ActionCounter,
    AgentBehaviorStatsAnalyzer,
    AnalysisManager,
    FollowerCounter,
    ItemSalesAnalyzer,
    RecordStore,
    StoreSalesAnalyzer,
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
        ]
    )
    stores = RecordStore(load_from_file("log.txt"))
    results = manager.run_all(
        stores,
        render_summary=True,
        figs_save_path="qwen_results",
    )
