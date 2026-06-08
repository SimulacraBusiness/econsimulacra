import pathlib

from econsimulacra.log_analyses import (
    AnalysisManager,
    RecordStore,
    StressActionAnalyzer,
    TopicSalesAnalyzer,
    load_from_file,
)

if __name__ == "__main__":
    parent_dir = pathlib.Path("log_gpt-oss-120b")
    paths = list(parent_dir.glob("*.txt"))
    stores = []
    for path in paths:
        records = load_from_file(str(path))
        store = RecordStore(records)
        stores.append(store)
    manager = AnalysisManager(
        analyzers=[
            StressActionAnalyzer(exclude_agent_ids=[30, 31]),
            TopicSalesAnalyzer(topic_words=["Pizza", "pizza"], window_size=24),
        ]
    )
    results = manager.run_all_w_multi_stores(
        stores,
        render_summary=True,
        figs_save_path="results_all/gpt-oss-120b",
    )
