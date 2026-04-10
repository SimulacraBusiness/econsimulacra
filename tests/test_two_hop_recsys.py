import random

from econsimulacra.envs.social_networks import SocialNetwork


class TestTwoHopRecommenderSystem:
    """
    0 -> 1 <-> 2
         |     ^
         v     |
         3 <-> 4 <-------|
               ^         |
               |         v
               5 -> 7 -> 8
               |         ^
               v         |
               6 -> 9 -> 10
    """

    config = {
        "type": "SocialNetwork",
        "followCap": 4,
        "recSys": {
            "type": "TwoHopRecommenderSystem",
            "maxRecommendations": 2,
            "isRandomized": False,
        },
    }
    sn = SocialNetwork(config=config, prng=random.Random(0), registered_classes=[])
    recsys = sn.rec_sys
    for i in range(11):
        sn.add_agent(agent_id=i, agent_name=f"agent_{i}")
    sn.follow_agent(0, 1)
    sn.follow_agent(1, 2)
    sn.follow_agent(2, 1)
    sn.follow_agent(1, 3)
    sn.follow_agent(3, 4)
    sn.follow_agent(4, 3)
    sn.follow_agent(4, 2)
    sn.follow_agent(5, 4)
    sn.follow_agent(5, 2)
    sn.follow_agent(5, 6)
    sn.follow_agent(5, 7)
    sn.follow_agent(7, 8)
    sn.follow_agent(6, 9)
    sn.follow_agent(9, 10)
    sn.follow_agent(10, 8)
    sn.follow_agent(4, 8)
    sn.follow_agent(8, 4)
    sn.unfollow_agent(5, 2)

    def test_two_hop_recsys(self) -> None:
        assert self.recsys.agent_id2follows == {
            0: {1},
            1: {2, 3},
            2: {1},
            3: {4},
            4: {2, 3, 8},
            5: {4, 6, 7},
            6: {9},
            7: {8},
            8: {4},
            9: {10},
            10: {8},
        }
        assert self.recsys.agent_id2followers == {
            0: set(),
            1: {0, 2},
            2: {1, 4},
            3: {1, 4},
            4: {3, 5, 8},
            5: set(),
            6: {5},
            7: {5},
            8: {4, 7, 10},
            9: {6},
            10: {9},
        }
        assert self.recsys.agent_id2num_followers == {
            0: 0,
            1: 2,
            2: 2,
            3: 2,
            4: 3,
            5: 0,
            6: 1,
            7: 1,
            8: 3,
            9: 1,
            10: 1,
        }
        agent_id2two_hop_follows = self.recsys.agent_id2two_hop_follows
        for k, v in agent_id2two_hop_follows.items():
            agent_id2two_hop_follows[k] = dict(v)
        assert self.recsys.agent_id2two_hop_follows == {
            0: {2: 1, 3: 1},
            1: {4: 1},
            2: {3: 1},
            3: {2: 1, 8: 1},
            4: {1: 1},
            5: {2: 1, 3: 1, 8: 2, 9: 1},
            6: {10: 1},
            7: {4: 1},
            8: {2: 1, 3: 1},
            9: {8: 1},
            10: {4: 1},
        }

    def test_get_recommendations(self) -> None:
        def get_ids_from_recs(recs: list[dict[str, int | str]]) -> list[int]:
            return [rec["agent_id"] for rec in recs]

        assert get_ids_from_recs(self.recsys.get_recommendations(0)) == [2, 3]
        assert get_ids_from_recs(self.recsys.get_recommendations(1)) == [4, 8]
        assert get_ids_from_recs(self.recsys.get_recommendations(2)) == [3, 4]
        assert get_ids_from_recs(self.recsys.get_recommendations(3)) == [8, 2]
        assert get_ids_from_recs(self.recsys.get_recommendations(4)) == [1, 6]
        assert get_ids_from_recs(self.recsys.get_recommendations(5)) == [8, 2]
        assert get_ids_from_recs(self.recsys.get_recommendations(6)) == [10, 4]
        assert get_ids_from_recs(self.recsys.get_recommendations(7)) == [4, 1]
        assert get_ids_from_recs(self.recsys.get_recommendations(8)) == [2, 3]
        assert get_ids_from_recs(self.recsys.get_recommendations(9)) == [8, 4]
        assert get_ids_from_recs(self.recsys.get_recommendations(10)) == [4, 1]
