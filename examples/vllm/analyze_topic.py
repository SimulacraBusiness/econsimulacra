import pathlib

import matplotlib.pyplot as plt

from econsimulacra.log_analyses import RecordStore, TopicAnalyzer, load_from_file

if __name__ == "__main__":
    results_path = pathlib.Path("results/llama3-70b/topics")
    parent_dir = pathlib.Path("log_llama3-70b")
    paths = list(parent_dir.glob("*.txt"))
    stores = []
    for path in paths:
        records = load_from_file(str(path))
        store = RecordStore(records)
        stores.append(store)
    analyzer = TopicAnalyzer(exclude_agent_ids=[31])
    results = analyzer.analyze_stores(stores)
    for i, result in enumerate(results):
        figs = analyzer.draw_figs(result)
        for name, fig in figs.items():
            fig.tight_layout()
            fig.savefig(
                results_path / f"{name}_{i}.pdf",
                format="pdf",
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.02,
            )
            plt.close(fig)
