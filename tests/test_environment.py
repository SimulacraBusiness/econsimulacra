from econsimulacra.agents import Agent
from econsimulacra.envs import Environment
from econsimulacra.items import Item
import pytest
from typing import Any, Optional


class DummyHousehold(Agent):
    def _initialize_inventory(self) -> dict[str, float | int]:
        return {"Yen": 100000, "Rice": 50}

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        action_dic: dict[str, Any] = {}
        is_moving: bool = obs["is_moving"]
        if is_moving:
            action_dic["move"] = obs["destination"]
        else:
            if obs["pos"] == obs["initial_coords"]:
                if self.inventory_dic["Rice"] > 50:
                    action_dic["consumptions"] = {
                        "item_name": "Rice", "item_amount": 1
                    }
                else:
                    action_dic["move"] = obs["retailer_pos"]
            elif obs["pos"] == obs["retailer_pos"]:
                if self.inventory_dic["Rice"] < 50:
                    action_dic["purchase"] = {
                        "item_name": "Rice", "item_amount": 1, "rice": None,
                        "counterparty_id": obs["retailer_id"]
                    }
                else:
                    action_dic["move"] = obs["initial_coords"]
            else:
                action_dic["move"] = obs["initial_coords"]
        action_dic["tweet"] = "Hello, world!"
        follow_id: int
        unfollow_id: Optional[int] = None
        follows: set[int] = obs["follows"]
        for id in obs["recommended_follows"]:
            if id not in follows:
                follow_id = id
                break
            else:
                unfollow_id = id
        action_dic["follow"] = follow_id
        action_dic["unfollow"] = unfollow_id
        return action_dic


class DummyRetailer(Agent):
    def _initialize_inventory(self) -> dict[str, float | int]:
        return {"Yen": 500, "Rice": 10000}

    def act(self, obs):
        item: Item
        for item_name, item in obs["item_name2item"].items():
            if item_name == "Rice":
                item.set_price(1000)
        action_dic: dict[str, Any] = {
            "reactions": []
        }
        for order in obs["orders"]:
            action_dic["reactions"].append(
                {
                    "order_id": order["order_id"],
                    "accepted_amount": order["item_amount"],
                }
            )
        return action_dic


class Yen(Item):
    def __init__(self, item_id: int, item_name: str = "Yen") -> None:
        super().__init__(item_id=item_id, item_name=item_name)


class Rice(Item):
    def __init__(self, item_id: int, item_name: str = "Rice") -> None:
        super().__init__(item_id=item_id, item_name=item_name)
        self.price = 1000


class DummyEnvironment(Environment):
    def get_observations(self, agent_id: int) -> dict[str, Any]:
        retailer_id: int = self.agent_name2agent_id["DummyRetailer5"]
        if agent_id not in self.agent_id2agent:
            raise ValueError(f"Agent ID {agent_id} not found in the environment.")
        if agent_id in self.household_ids:
            return {
                "space_size": self.space_size,
                "pos": self.grid_space.get_pos(agent_id),
                "initial_coords": self.agent_id2initial_coords[agent_id],
                "retailer_id": retailer_id,
                "retailer_pos": self.grid_space.get_pos(retailer_id),
                "is_moving": self.agent_id2is_moving[agent_id],
                "destination": self.agent_id2destination[agent_id],
                "follows": self.social_network.get_follows(agent_id),
                "recommended_follows": [id for id in self.household_ids if id != agent_id],
            }
        else:
            return {
                "orders": self.pending_orders,
                "item_name2item": self.item_name2item,
            }


class TestEnvironment:
    config = { # <- ここ変えた！
        "gridSpace": (10, 10),
        "simulation": {
            "numSteps": 10,
        },
        "environment": {
            "gridSpace": (10, 10),
            "cashName": "Yen",
            "agents": ["DummyHousehold", "DummyRetailer"],
            "items": ["Yen", "Rice"]
        },
        "DummyHousehold": {
            "isHousehold": True,
            "numAgents": 5,
        },
        "DummyRetailer": {
            "isHousehold": False,
            "numAgents": 1,
            "initialCoords": (5, 5),
        },
    }

    def test___init__(self) -> None:
        env = DummyEnvironment(config=self.config)
        assert env.space_size == (10, 10)
        assert env.config == self.config
        assert env.cash_name == "Yen"
        assert isinstance(env.prng, type(pytest.importorskip("random").Random()))

    def test_register_classes(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
        )
        assert DummyHousehold in env.registered_classes
        assert DummyRetailer in env.registered_classes
        assert Yen in env.registered_classes
        assert Rice in env.registered_classes

    def test_reset(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
        )
        env.reset(seed=42)
        assert len(env.agent_ids) == 6
        assert len(env.household_ids) == 5
        assert len(env.others_ids) == 1
        assert len(env.item_name2item) == 2
        for agent_id in env.agent_ids:
            assert agent_id in env.grid_space.agent_id2pos
            assert isinstance(env.grid_space.get_pos(agent_id), tuple)
            assert agent_id in env.agent_id2initial_coords
            assert env.agent_id2initial_coords[agent_id] == env.grid_space.get_pos(agent_id)
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
                assert agent.inventory_dic == {"Yen": 100000, "Rice": 50}
                assert agent.agent_name == f"DummyHousehold{agent_id}"
            else:
                assert agent.inventory_dic == {"Yen": 500, "Rice": 10000}
                assert agent.agent_name == f"DummyRetailer{agent_id}"

    def test_move(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
        )
        env.reset(seed=42)
        household_id: int = env.household_ids[0]
        initial_pos: tuple[int, int] = env.grid_space.get_pos(household_id)
        destination_pos: tuple[int, int] = (
            min(initial_pos[0] + 2, env.space_size[0] - 1),
            min(initial_pos[1] + 3, env.space_size[1] - 1)
        )
        env._move(
            agent_id=household_id, where_to_move=destination_pos
        )
        new_pos: tuple[int, int] = env.grid_space.get_pos(household_id)
        assert new_pos == (
            min(initial_pos[0] + 1, env.space_size[0] - 1),
            min(initial_pos[1] + 1, env.space_size[1] - 1)
        )
        assert env.agent_id2is_moving[household_id] is True
        assert env.agent_id2destination[household_id] == destination_pos
        destination_pos: tuple[int, int] = (
            min(initial_pos[0] + 1, env.space_size[0] - 1),
            min(initial_pos[1] + 1, env.space_size[1] - 1)
        )
        env._move(
            agent_id=household_id, where_to_move=destination_pos
        )
        new_pos = env.grid_space.get_pos(household_id)
        assert new_pos == destination_pos
        assert env.agent_id2is_moving[household_id] is False
        assert env.agent_id2destination[household_id] is None
        destination_pos: str = "DummyRetailer5"
        env._move(
            agent_id=household_id, where_to_move=destination_pos
        )
        retailer_id: int = env.agent_name2agent_id["DummyRetailer5"]
        retailer_pos: tuple[int, int] = env.grid_space.get_pos(retailer_id)
        assert env.agent_id2is_moving[household_id] is True
        assert env.agent_id2destination[household_id] == retailer_pos

    def test_consume_items(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
        )
        env.reset(seed=42)
        household_id: int = env.household_ids[0]
        initial_Rice_amount: int = env.agent_id2agent[household_id].inventory_dic["Rice"]
        consumptions: list[dict[str, Any]] = [
            {"item_name": "Rice", "item_amount": 5}
        ]
        env._consume_items(
            agent_id=household_id,
            consumptions=consumptions
        )
        new_Rice_amount: int = env.agent_id2agent[household_id].inventory_dic["Rice"]
        assert new_Rice_amount == initial_Rice_amount - 5

    def test_add_new_orders_and_proposals(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
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
        assert proposal0.expire_in == 1
        assert proposal0.accept is None
        env.latest_proposal_id = 1

    def test_apply_action_to_env_and_process_orders_and_proposals(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
        )
        env.reset(seed=42)
        household_id0: int = env.household_ids[0]
        household_id1: int = env.household_ids[1]
        retailer_id: int = env.others_ids[0]
        action_dic0: dict[str, Any] = {
            "move": "DummyRetailer5",
            "consumptions": [{"item_name": "Rice", "item_amount": 2}],
            "orders": [{"item_name": "Rice", "item_amount": 2, "counterparty_id": retailer_id}],
            "tweet": "Hello, world!",
            "follow": household_id1,
        }
        env.apply_action_to_env(agent_id=household_id0, action_dic=action_dic0)
        action_dic1: dict[str, Any] = {
            "move": "DummyRetailer5",
            "consumptions": [{"item_name": "Rice", "item_amount": 2}],
            "orders": [{"item_name": "Rice", "item_amount": 2, "counterparty_id": retailer_id}],
            "proposals": [{"responder_agent_id": household_id0, "give_item_name": "Rice", "give_item_amount": 5, "get_item_name": "Yen", "get_item_amount": 5000}],
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
        household0: Agent = env.agent_id2agent[household_id0]
        household1: Agent = env.agent_id2agent[household_id1]
        retailer: Agent = env.agent_id2agent[retailer_id]
        assert household0.inventory_dic["Rice"] == 50 - 2 + 2 + 5
        assert household0.inventory_dic["Yen"] == 100000 - 2000 - 5000
        assert household1.inventory_dic["Rice"] == 50 - 2 + 2 - 5
        assert household1.inventory_dic["Yen"] == 100000 - 2000 + 5000
        assert retailer.inventory_dic["Rice"] == 10000 - 2 - 2
        assert retailer.inventory_dic["Yen"] == 500 + 2000 + 2000

    def test_step(self) -> None:
        env = DummyEnvironment(config=self.config)
        env.register_classes(
            [
                DummyHousehold, DummyRetailer, Yen, Rice
            ]
        )
        env.reset(seed=42)
        all_actions_dic: dict[int, dict[str, Any]] = {}
        for agent_id in env.agent_ids:
            agent: Agent = env.agent_id2agent[agent_id]
            obs: dict[str, Any] = env.get_observations(agent_id=agent_id)
            action_dic: dict[str, Any] = agent.act(obs=obs)
            all_actions_dic[agent_id] = action_dic
        env.step(all_actions_dic=all_actions_dic)