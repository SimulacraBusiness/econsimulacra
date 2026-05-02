from datetime import datetime

from econsimulacra.log_analyses import (
    ActionCounter,
    AnalysisManager,
    FollowerCounter,
    RecordStore,
    SalesAnalyzer,
    load_from_file,
)


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


class TestAnalysisManager:
    file_path = "tests/dummy_log.txt"

    def test_run_all(self):
        manager = AnalysisManager(
            analyzers=[
                ActionCounter(),
                FollowerCounter(),
                SalesAnalyzer(),
            ]
        )
        stores = RecordStore(load_from_file(self.file_path))
        results = manager.run_all(stores)
        assert results["action_count"] == {
            "move": 1,
            "tweet": 1,
            "follow": 1,
            "unfollow": 1,
            "order": 1,
            "proposal": 1,
            "consumption": 1,
            "order_reaction": 1,
            "proposal_reaction": 1,
            "change_price": 1,
        }
        assert results["follower_count"] == {
            "Government": {
                parse_time("2025-03-01 09:00:00"): 0,
            },
            "Agent 33": {
                parse_time("2025-03-01 09:00:00"): 0,
                parse_time("2025-03-01 20:56:05"): 0,
            },
        }
        assert results["sales"] == {
            "PrimeDiner4": {parse_time("2025-03-05 08:28:41"): 15001.0}
        }
