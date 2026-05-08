from typing import Any, Callable, Optional

import pytest

from econsimulacra.agents import Agent
from econsimulacra.envs import Environment, Order
from econsimulacra.events import Event, EventTrigger
from econsimulacra.items import Item
from econsimulacra.logs import DictLogger
from econsimulacra.memory import MemoryHandler

Provider = Callable[[int], Any]


class DummyHousehold(Agent):
    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        action_dic: dict[str, Any] = {}
        is_moving: bool = obs["self_is_moving"]
        if is_moving:
            action_dic["move"] = obs["self_destination"]
        else:
            pos: tuple[int, int] = obs["self_pos"]
            retailer_pos: tuple[int, int] = obs["others_pos"][0]["pos"]
            if pos == obs["self_init_pos"]:
                if self.inventory_dic["Rice"] >= 75:
                    action_dic["consumptions"] = [
                        {"item_name": "Rice", "item_amount": 10}
                    ]
                else:
                    action_dic["move"] = retailer_pos
            elif pos == retailer_pos:
                if self.inventory_dic["Rice"] < 75:
                    info4co_located_agents: list[dict[str, Any]] = obs[
                        "others_inventory"
                    ]
                    retailer_inventory_dic: dict[str, Any] = info4co_located_agents[0]
                    retailer_id: int = retailer_inventory_dic["agent_id"]
                    action_dic["orders"] = [
                        {
                            "item_name": "Rice",
                            "item_amount": 10,
                            "counterparty_id": retailer_id,
                        }
                    ]
                else:
                    action_dic["move"] = obs["self_pos"]
            else:
                action_dic["move"] = obs["self_pos"]
        action_dic["tweet"] = "Hello, world!"
        follow_id: Optional[int] = None
        unfollow_id: Optional[int] = None
        visible_tl: list[dict[str, Any]] = obs["visible_tl"]
        follows: set[int] = [tl_dic["agent_id"] for tl_dic in visible_tl]
        unfollow_id = follows[0] if len(follows) > 0 else None
        recommended_follows: list[int] = [
            rec["agent_id"] for rec in obs["recommended_follows"]
        ]
        follow_id = recommended_follows[0] if len(recommended_follows) > 0 else None
        action_dic["follow"] = follow_id
        action_dic["unfollow"] = unfollow_id
        return action_dic


class DummyRetailer(Agent):
    def self_assign_name(self, config: dict[str, Any]) -> None:
        self.agent_name = "DummyRetailer"

    def _initialize_inventory(self, config) -> dict[str, float | int]:
        return {"Yen": 500, "Rice": 10000}

    def act(self, obs):
        action_dic: dict[str, list[dict[str, Any]]] = {
            "reactions": [],
            "set_prices": [],
        }
        item_name2prices: list[dict[str, Any]] = obs["item_name2price"]
        for d in item_name2prices:
            item_name: str = d["item_name"]
            if item_name == "Rice":
                action_dic["set_prices"].append(
                    {
                        "item_name": item_name,
                        "price": d["price"] * self.prng.uniform(0.99, 1.11),
                    }
                )
        incoming_orders: list[Order] = obs["incoming_orders"]
        for order_info in incoming_orders:
            action_dic["reactions"].append(
                {
                    "kind": "order",
                    "id": order_info["order_id"],
                    "accept_amount": order_info["item_amount"],
                }
            )
        return action_dic


class DummyMemoryHandler(MemoryHandler):
    def get_memory(self, agent_id: int) -> Optional[dict[str, Any]]:
        return {}


class DummyEvent(Event):
    def __init__(
        self,
        trigger: EventTrigger,
        config: dict[str, Any],
    ) -> None:
        super().__init__(trigger=trigger, config=config)
        self.num_executions = 0

    def execute(self, env, log=None):
        self.num_executions += 1


class TestEnvironment:
    config = {
        "simulation": {
            "numSteps": 100,
            "events": ["DummyEvent1", "DummyEvent3"],
        },
        "environment": {
            "space": "gridSpace",
            "socialNetwork": "socialNetwork",
            "cashName": "Yen",
            "agents": ["DummyHousehold", "DummyRetailer"],
            "items": ["Yen", "Rice"],
            "service": ["timeTranslator", "memoryHandler"],
        },
        "gridSpace": {
            "type": "GridSpace",
            "gridSize": [10, 10],
        },
        "socialNetwork": {
            "type": "SocialNetwork",
            "followCap": 2,
            "recSys": {
                "type": "TwoHopRecommenderSystem",
                "maxRecommendations": 2,
            },
        },
        "DummyHousehold": {
            "isHousehold": True,
            "numAgents": 5,
            "inventory": {
                "Yen": [100000, 200000],
                "Rice": [50, 100],
            },
        },
        "DummyRetailer": {
            "isRichInfoAllowed": True,
            "isHousehold": False,
            "numAgents": 1,
            "initialCoords": (9, 9),
            "inventory": {
                "Yen": 100000,
                "Rice": 1000,
            },
            "provideInfo4CoLocatedAgents": ["inventory"],
            "provideInfo4AllAgents": ["self_pos"],
        },
        "Yen": {
            "type": "Item",
            "initialPrice": 1.0,
            "weightInBasket": 0,
        },
        "Rice": {
            "type": "Item",
            "initialPrice": 1000.0,
            "weightInBasket": 1,
        },
        "timeTranslator": {
            "type": "TimeTranslator",
            "numSteps": 100,
            "startDatetime": "2025-01-01 00:00:00",
            "endDatetime": "2025-01-01 00:10:00",
        },
        "memoryHandler": {
            "type": "DummyMemoryHandler",
            "memoryLength": 2,
            "memorySummarizer": {
                "type": "StressAwareSummarizer",
                "stressCalculator": {
                    "type": "StressCalculator",
                    "item2Weight": {
                        "Yen": 0,
                        "Rice": 1,
                    },
                },
            },
        },
        "DummyEvent1": {
            "trigger": {
                "at": (1, 3, 5),
            },
            "type": "DummyEvent",
        },
        "DummyEvent3": {
            "trigger": {
                "between": (2, 4),
            },
            "type": "DummyEvent",
        },
    }

    def test___init__(self) -> None:
        env = Environment(config=self.config)
        assert env.config == self.config
        assert env.cash_name == "Yen"
        assert isinstance(env.prng, type(pytest.importorskip("random").Random()))

    def test_register_classes(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        assert DummyHousehold in env.registered_classes
        assert DummyRetailer in env.registered_classes
        assert DummyMemoryHandler in env.registered_classes
        assert DummyEvent in env.registered_classes

    def test_reset(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        assert len(env.agent_ids) == 6
        assert len(env.household_ids) == 5
        assert len(env.others_ids) == 1
        assert len(env.item_name2item) == 2
        assert env.social_network.follow_cap == 2
        assert "timeTranslator" in env.service_dic
        assert env.get_time_translator() is not None
        assert "memoryHandler" in env.service_dic
        assert env.get_memory_handler() is not None
        for item_name, item in env.item_name2item.items():
            assert isinstance(item, Item)
            if item_name == "Yen":
                assert item.get_price() == 1.0
                assert item.weight_in_basket == 0
            elif item_name == "Rice":
                assert item.get_price() == 1000.0
                assert item.weight_in_basket == 1
        for agent_id in env.agent_ids:
            assert agent_id in env.grid_space.agent_id2pos
            assert isinstance(env.grid_space.get_pos(agent_id), tuple)
            assert agent_id in env.agent_id2initial_coords
            assert env.agent_id2initial_coords[agent_id] == env.grid_space.get_pos(
                agent_id
            )
            assert agent_id in env.agent_id2is_moving
            assert env.agent_id2is_moving[agent_id] is False
            assert agent_id in env.agent_id2destination
            assert env.agent_id2destination[agent_id] is None
            assert agent_id in env.social_network.nodes
            assert env.social_network.agent_id2tweet[agent_id] == ""
            assert env.social_network.agent_id2followers[agent_id] == set()
            assert env.social_network.agent_id2follows[agent_id] == set()
            assert env.social_network.get_followers(agent_id) == set()
            assert env.social_network.get_follows(agent_id) == set()
            agent: Agent = env.agent_id2agent[agent_id]
            assert isinstance(agent, (DummyHousehold, DummyRetailer))
            if isinstance(agent, DummyHousehold):
                assert agent.agent_name == f"DummyHousehold{agent_id}"
                inventory_dic = agent.inventory_dic
                assert "Yen" in inventory_dic
                assert "Rice" in inventory_dic
                assert 100000 <= inventory_dic["Yen"] <= 200000
                assert 50 <= inventory_dic["Rice"] <= 100
            else:
                assert agent.inventory_dic == {"Yen": 500, "Rice": 10000}
                assert agent.agent_name == "DummyRetailer"
        env = Environment(config=self.config, logger=DictLogger())
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        logger = env.logger
        assert isinstance(logger, DictLogger)
        assert len(logger.logs) == 12

    def test_move(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        household_id: int = env.household_ids[0]
        initial_pos: tuple[int, int] = env.grid_space.get_pos(household_id)
        space_size: tuple[int, int] = env.grid_space.get_space_size()
        destination_pos: tuple[int, int] = (
            min(initial_pos[0] + 2, space_size[0] - 1),
            min(initial_pos[1] + 3, space_size[1] - 1),
        )
        env._move(agent_id=household_id, where_to_move=destination_pos)
        new_pos: tuple[int, int] = env.grid_space.get_pos(household_id)
        assert new_pos == (
            min(initial_pos[0] + 1, space_size[0] - 1),
            min(initial_pos[1] + 1, space_size[1] - 1),
        )
        assert env.agent_id2is_moving[household_id] is True
        assert env.agent_id2destination[household_id] == destination_pos
        destination_pos: tuple[int, int] = (
            min(initial_pos[0] + 1, space_size[0] - 1),
            min(initial_pos[1] + 1, space_size[1] - 1),
        )
        env._move(agent_id=household_id, where_to_move=destination_pos)
        new_pos = env.grid_space.get_pos(household_id)
        assert new_pos == destination_pos
        assert env.agent_id2is_moving[household_id] is False
        assert env.agent_id2destination[household_id] is None
        destination_pos: str = "DummyRetailer"
        retailer_id: int = env.agent_name2agent_id["DummyRetailer"]
        env._move(agent_id=household_id, where_to_move=destination_pos)
        retailer_pos: tuple[int, int] = env.grid_space.get_pos(retailer_id)
        assert env.agent_id2is_moving[household_id] is True
        assert env.agent_id2destination[household_id] == retailer_pos

    def test_consume_items(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        household_id: int = env.household_ids[0]
        initial_rice_amount: int = env.agent_id2agent[household_id].inventory_dic[
            "Rice"
        ]
        consumptions: list[dict[str, Any]] = [{"item_name": "Rice", "item_amount": 5}]
        env._consume_items(agent_id=household_id, consumptions=consumptions)
        new_rice_amount: int = env.agent_id2agent[household_id].inventory_dic["Rice"]
        assert new_rice_amount == initial_rice_amount - 5

    def test_add_new_orders_and_proposals(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        orders: list[dict[str, Any]] = [
            {
                "item_name": "Rice",
                "item_amount": 10,
                "counterparty_id": 5,
            },
            {
                "item_name": "Rice",
                "item_amount": 20,
                "price": None,
                "counterparty_id": 5,
            },
        ]
        env._add_new_orders_and_proposals(agent_id=0, orders=orders, proposals=[])
        assert len(env.pending_orders) == 2
        order0, order1 = env.pending_orders
        assert order0.order_id == 0
        assert order0.agent_id == 0
        assert order0.item_name == "Rice"
        assert order0.item_amount == 10
        assert order0.counterparty_id == 5
        assert order1.order_id == 1
        assert order1.agent_id == 0
        assert order1.item_name == "Rice"
        assert order1.item_amount == 20
        assert order1.counterparty_id == 5
        env.latest_order_id = 2
        proposals: list[dict[str, Any]] = [
            {
                "responder_agent_id": 1,
                "give_item_name": "Rice",
                "give_item_amount": 15,
                "get_item_name": "Yen",
                "get_item_amount": 15000,
            }
        ]
        env._add_new_orders_and_proposals(agent_id=0, orders=[], proposals=proposals)
        assert len(env.pending_swap_proposals) == 1
        proposal0 = env.pending_swap_proposals[0]
        assert proposal0.proposal_id == 0
        assert proposal0.proposer_agent_id == 0
        assert proposal0.responder_agent_id == 1
        assert proposal0.give_item_name == "Rice"
        assert proposal0.give_item_amount == 15
        assert proposal0.get_item_name == "Yen"
        assert proposal0.get_item_amount == 15000
        assert proposal0.expire_in == 2
        assert proposal0.accept is None
        env.latest_proposal_id = 1

    def test_apply_action_to_env_and_process_orders_and_proposals(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        household_id0: int = env.household_ids[0]
        household_id1: int = env.household_ids[1]
        household_id2: int = env.household_ids[2]
        household_id3: int = env.household_ids[3]
        retailer_id: int = env.others_ids[0]
        household0: Agent = env.agent_id2agent[household_id0]
        rice0_amount: int = household0.inventory_dic["Rice"]
        yen0_amount: int = household0.inventory_dic["Yen"]
        rice1_amount: int = env.agent_id2agent[household_id1].inventory_dic["Rice"]
        yen1_amount: int = env.agent_id2agent[household_id1].inventory_dic["Yen"]
        retailer_rice_amount: int = env.agent_id2agent[retailer_id].inventory_dic[
            "Rice"
        ]
        retailer_yen_amount: int = env.agent_id2agent[retailer_id].inventory_dic["Yen"]
        household1: Agent = env.agent_id2agent[household_id1]
        retailer: Agent = env.agent_id2agent[retailer_id]
        action_dic0: dict[str, Any] = {
            "move": "DummyRetailer",
            "consumptions": [{"item_name": "Rice", "item_amount": 2}],
            "orders": [
                {"item_name": "Rice", "item_amount": 2, "counterparty_id": retailer_id}
            ],
            "tweet": "Hello, world!",
            "follow": household_id1,
        }
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic0)
        action_dic1: dict[str, Any] = {
            "move": "DummyRetailer",
            "consumptions": [{"item_name": "Rice", "item_amount": 2}],
            "orders": [
                {"item_name": "Rice", "item_amount": 2, "counterparty_id": retailer_id}
            ],
            "proposals": [
                {
                    "responder_agent_id": household_id0,
                    "give_item_name": "Rice",
                    "give_item_amount": 5,
                    "get_item_name": "Yen",
                    "get_item_amount": 5000,
                }
            ],
            "tweet": "Hello, world!",
            "follow": household_id0,
        }
        env.apply_action_to_env(agent_id=household_id1, action_dic=action_dic1)
        assert len(env.pending_orders) == 2
        assert len(env.pending_swap_proposals) == 1
        assert env.social_network.agent_id2tweet[household_id0] == "Hello, world!"
        assert env.social_network.agent_id2tweet[household_id1] == "Hello, world!"
        assert household_id1 in env.social_network.get_follows(household_id0)
        assert household_id0 in env.social_network.get_follows(household_id1)
        action_dic00: dict[str, Any] = {"follow": household_id2}
        action_dic01: dict[str, Any] = {"follow": household_id3}
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic00)
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic01)
        assert household_id2 in env.social_network.get_follows(household_id0)
        assert household_id3 not in env.social_network.get_follows(household_id0)
        action_dic02: dict[str, Any] = {"unfollow": household_id2}
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic02)
        assert household_id2 not in env.social_network.get_follows(household_id0)
        action_dic03: dict[str, Any] = {"follow": household_id3}
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic03)
        assert household_id3 in env.social_network.get_follows(household_id0)
        action_dic_retailer: dict[str, Any] = {
            "reactions": [
                {
                    "kind": "order",
                    "id": 0,
                    "accept_amount": 2,
                },
                {
                    "kind": "order",
                    "id": 1,
                    "accept_amount": 2,
                },
            ]
        }
        env.apply_action_to_env(agent_id=retailer_id, action_dic=action_dic_retailer)
        order0, order1 = env.pending_orders
        assert order0.accepted_amount == 2
        assert order1.accepted_amount == 2
        action_dic0 = {
            "reactions": [
                {
                    "kind": "proposal",
                    "id": 0,
                    "accept": True,
                }
            ]
        }
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic0)
        proposal0 = env.pending_swap_proposals[0]
        assert proposal0.accept is True
        env._process_orders_and_proposals()
        env._remove_expired_orders_and_proposals()
        assert len(env.pending_orders) == 0
        assert len(env.pending_swap_proposals) == 0
        assert household0.inventory_dic["Rice"] == rice0_amount - 2 + 2 + 5
        assert household0.inventory_dic["Yen"] == yen0_amount - 2000 - 5000
        assert household1.inventory_dic["Rice"] == rice1_amount - 2 + 2 - 5
        assert household1.inventory_dic["Yen"] == yen1_amount - 2000 + 5000
        assert retailer.inventory_dic["Rice"] == retailer_rice_amount - 2 - 2
        assert retailer.inventory_dic["Yen"] == retailer_yen_amount + 2000 + 2000

    def test_get_observations(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        for agent_id in env.agent_ids:
            obs: dict[str, Any] = env.get_observations(agent_id=agent_id)
            assert "time" in obs
            assert obs["time"] == "2025-01-01 00:00:00"
            assert "timedelta" in obs
            assert obs["timedelta"] == "0:00:06"
            assert "self_agent_id" in obs
            assert obs["self_agent_id"] == agent_id
            assert "self_name" in obs
            assert obs["self_name"] == env.agent_id2agent[agent_id].get_self_name()
            assert "memory" in obs
            assert obs["memory"] == {}
            assert "self_pos" in obs
            assert obs["self_pos"] == env.grid_space.get_pos(agent_id)
            assert "self_init_pos" in obs
            assert obs["self_init_pos"] == env.agent_id2initial_coords[agent_id]
            assert "self_is_moving" in obs
            assert obs["self_is_moving"] is env.agent_id2is_moving[agent_id]
            assert "self_destination" in obs
            assert obs["self_destination"] == env.agent_id2destination[agent_id]
            assert "others_pos" in obs
            if agent_id in env.household_ids:
                assert obs["others_pos"] == [
                    {
                        "agent_id": env.agent_name2agent_id["DummyRetailer"],
                        "agent_name": "DummyRetailer",
                        "pos": (9, 9),
                    }
                ]
            else:
                assert obs["others_pos"] == []
            assert "follow_cap" in obs
            assert obs["follow_cap"] == 2
            assert "num_followers" in obs
            assert obs["num_followers"] == 0
            assert "num_follows" in obs
            assert obs["num_follows"] == 0
            assert "self_tweet" in obs
            assert obs["self_tweet"] == ""
            assert "visible_tl" in obs
            assert obs["visible_tl"] == []
            assert "incoming_orders" in obs
            assert obs["incoming_orders"] == []
            assert "incoming_proposals" in obs
            assert obs["incoming_proposals"] == []
            assert "recommended_follows" in obs
            assert len(obs["recommended_follows"]) > 0
            if agent_id not in env.household_ids:
                assert "item_name2price" in obs
            else:
                assert "item_name2price" not in obs
        household_id: int = env.household_ids[0]
        while True:
            env._move(agent_id=household_id, where_to_move="DummyRetailer")
            if not env.agent_id2is_moving[household_id]:
                break
            else:
                obs = env.get_observations(agent_id=household_id)
                co_located_agents: set[int] = env.grid_space.get_colocated_agents(
                    agent_id=household_id
                )
                if len(co_located_agents) == 0:
                    assert "others_inventory" not in obs
                else:
                    assert "others_inventory" in obs
        obs = env.get_observations(agent_id=household_id)
        assert "others_inventory" in obs
        assert obs["others_inventory"] == [
            {
                "agent_id": env.agent_name2agent_id["DummyRetailer"],
                "agent_name": "DummyRetailer",
                "Rice": {"price": 1000.0, "amount": 10000},
            }
        ]

    def test_step(self) -> None:
        env = Environment(config=self.config)
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        all_actions_dic: dict[int, dict[str, Any]] = {}
        for agent_id in env.agent_ids:
            agent: Agent = env.agent_id2agent[agent_id]
            obs: dict[str, Any] = env.get_observations(agent_id=agent_id)
            action_dic: dict[str, Any] = agent.act(obs=obs)
            all_actions_dic[agent_id] = action_dic
        env.step(all_actions_dic=all_actions_dic)
        env = Environment(config=self.config, logger=DictLogger())
        env.register_classes(
            [DummyHousehold, DummyRetailer, DummyMemoryHandler, DummyEvent]
        )
        env.reset(seed=42)
        for _ in range(self.config["simulation"]["numSteps"]):
            all_actions_dic = {}
            for agent_id in env.agent_ids:
                agent = env.agent_id2agent[agent_id]
                obs = env.get_observations(agent_id=agent_id)
                if agent_id in env.household_ids:
                    assert "item_name2price" not in obs
                else:
                    assert "item_name2price" in obs
                action_dic = agent.act(obs=obs)
                all_actions_dic[agent_id] = action_dic
            env.step(all_actions_dic=all_actions_dic)
        assert env.get_time() == self.config["timeTranslator"]["endDatetime"]
        event_manager = env.event_manager
        assert event_manager.events[0].num_executions == 3
        assert event_manager.events[1].num_executions == 3
        logger = env.logger
        assert isinstance(logger, DictLogger)
        assert len(logger.logs) > 12
        memory_handler = env.get_memory_handler()
        for agent_id in env.agent_ids:
            memory = memory_handler.agent_id2memory[agent_id]
            if agent_id in env.household_ids:
                assert len(memory.move_history) == 2
                assert len(memory.purchase_history) == 2
                assert len(memory.social_history) == 2
            else:
                assert len(memory.sale_history) == 2
                assert len(memory.set_price_history) == 2
