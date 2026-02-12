from econsimulacra.agents import Agent
from econsimulacra.envs import Environment
from econsimulacra.items import Item
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import Simulator
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
                if self.inventory_dic["Rice"] >= 50:
                    action_dic["consumptions"] = [
                        {"item_name": "Rice", "item_amount": 1}
                    ]
                else:
                    action_dic["move"] = obs["retailer_pos"]
            elif obs["pos"] == obs["retailer_pos"]:
                if self.inventory_dic["Rice"] < 50:
                    action_dic["orders"] = [
                        {
                            "item_name": "Rice",
                            "item_amount": 1,
                            "rice": None,
                            "counterparty_id": obs["retailer_id"],
                        }
                    ]
                else:
                    action_dic["move"] = obs["initial_coords"]
            else:
                action_dic["move"] = obs["initial_coords"]
        action_dic["tweet"] = "Hello, world!"
        follow_id: Optional[int] = None
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
        if len(obs["orders"]) == 0:
            return {}
        action_dic: dict[str, Any] = {"reactions": []}
        for order in obs["orders"]:
            action_dic["reactions"].append(
                {
                    "kind": "order",
                    "id": order.order_id,
                    "accept_amount": order.item_amount,
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
                "recommended_follows": [
                    id for id in self.household_ids if id != agent_id
                ],
            }
        else:
            return {
                "orders": self.pending_orders,
                "item_name2item": self.item_name2item,
            }

class TestSimulator:
    config = {
        "gridSpace": (10, 10),
        "simulation": {
            "numSteps": 10,
        },
        "environment": {
            "gridSpace": (10, 10),
            "cashName": "Yen",
            "agents": ["DummyHousehold", "DummyRetailer"],
            "items": ["Yen", "Rice"],
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
    def test_init(self):
        simulator = Simulator(
            config=self.config,
            env_class=DummyEnvironment,
            logger=DictLogger(),
        )
        assert isinstance(simulator.env, DummyEnvironment)

    def test_convert_list_to_tuple(self):
        simulator = Simulator(
            config=self.config,
            env_class=DummyEnvironment,
            logger=DictLogger(),
        )
        input_obj = {
            "a": [1, 2, 3],
            "b": {"c": [4, 5], "d": (6, 7)},
            "e": {
                "f": {"g": [8, 9]},
                "h": 10,
            }
        }
        converted_obj = simulator._convert_list_to_tuple(input_obj)
        assert isinstance(converted_obj["a"], tuple)
        assert isinstance(converted_obj["b"]["c"], tuple)
        assert isinstance(converted_obj["b"]["d"], tuple)
        assert isinstance(converted_obj["e"]["f"]["g"], tuple)

    def test_simulate(self):
        simulator = Simulator(
            config=self.config,
            env_class=DummyEnvironment,
            logger=DictLogger(),
        )
        simulator.register_classes(
            [
                DummyHousehold,
                DummyRetailer,
                Yen,
                Rice,
            ]
        )
        simulator.simulate(seed=42)
        logger: DictLogger = simulator.env.logger
        assert logger is not None
        assert len(logger.logs) > 0