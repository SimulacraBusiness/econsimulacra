import asyncio
from econsimulacra.agents import Agent
from econsimulacra.envs import Environment
from econsimulacra.envs import Order
from econsimulacra.logs import DictLogger
from econsimulacra.simulator import Simulator
from typing import Any
from typing import Callable
from typing import Optional

Provider = Callable[[int], Any]


class DummyHousehold(Agent):
    async def act(self, obs: dict[str, Any]) -> dict[str, Any]:
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
        recommended_follows: list[int] = obs["recommended_follows"]
        follow_id = recommended_follows[0] if len(recommended_follows) > 0 else None
        action_dic["follow"] = follow_id
        action_dic["unfollow"] = unfollow_id
        return action_dic


class DummyRetailer(Agent):
    def self_assign_name(self, config: dict[str, Any]) -> None:
        self.agent_name = "DummyRetailer"

    def _initialize_inventory(self, config) -> dict[str, float | int]:
        return {"Yen": 500, "Rice": 10000}

    async def act(self, obs):
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


class TestSimulator:
    config = {
        "simulation": {
            "numSteps": 10,
        },
        "environment": {
            "space": "gridSpace",
            "socialNetwork": "socialNetwork",
            "cashName": "Yen",
            "agents": ["DummyHousehold", "DummyRetailer"],
            "items": ["Yen", "Rice"],
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
        },
        "Rice": {
            "type": "Item",
            "initialPrice": 1000.0,
        },
    }

    def test_init(self):
        simulator = Simulator(
            config=self.config,
            env_class=Environment,
            logger=DictLogger(),
        )
        assert isinstance(simulator.env, Environment)

    def test_convert_list_to_tuple(self):
        simulator = Simulator(
            config=self.config,
            env_class=Environment,
            logger=DictLogger(),
        )
        input_obj = {
            "a": [1, 2, 3],
            "b": {"c": [4, 5], "d": (6, 7)},
            "e": {
                "f": {"g": [8, 9]},
                "h": 10,
            },
        }
        converted_obj = simulator._convert_list_to_tuple(input_obj)
        assert isinstance(converted_obj["a"], tuple)
        assert isinstance(converted_obj["b"]["c"], tuple)
        assert isinstance(converted_obj["b"]["d"], tuple)
        assert isinstance(converted_obj["e"]["f"]["g"], tuple)

    def test_simulate(self):
        simulator = Simulator(
            config=self.config,
            env_class=Environment,
            logger=DictLogger(),
        )
        simulator.register_classes(
            [
                DummyHousehold,
                DummyRetailer,
            ]
        )
        asyncio.run(simulator.simulate(seed=42, parallel_batch_size=4))
        logger: DictLogger = simulator.env.logger
        assert logger is not None
        assert len(logger.logs) > 0
