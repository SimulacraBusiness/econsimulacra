from typing import Any

DEFAULT_ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "move": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                },
                {"type": "string"},
            ],
            "default": None,
        },
        "consumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_name": {"type": "string"},
                    "item_amount": {"type": "number"},
                },
                "required": ["item_name", "item_amount"],
            },
            "default": [],
        },
        "orders": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "counterparty_id": {"type": "integer"},
                    "item_name": {"type": "string"},
                    "item_amount": {"type": "number"},
                    "ttl": {"type": "integer"},
                },
                "required": ["counterparty_id", "item_name", "item_amount", "ttl"],
            },
            "default": [],
        },
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "responder_agent_id": {"type": "integer"},
                    "give_item_name": {"type": "string"},
                    "give_item_amount": {"type": "number"},
                    "get_item_name": {"type": "string"},
                    "get_item_amount": {"type": "number"},
                    "ttl": {"type": "integer"},
                },
                "required": [
                    "responder_agent_id",
                    "give_item_name",
                    "give_item_amount",
                    "get_item_name",
                    "get_item_amount",
                    "ttl",
                ],
            },
            "default": [],
        },
        "reactions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "kind": {"type": "string", "enum": ["order", "proposal"]},
                    "id": {"type": "integer"},
                    "accept_amount": {"type": "number"},
                    "accept": {"type": "boolean"},
                },
                "required": ["kind", "id", "accept_amount", "accept"],
            },
            "default": [],
        },
        "tweet": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "follow": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "unfollow": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    },
    "required": [
        "move",
        "consumptions",
        "orders",
        "proposals",
        "reactions",
        "tweet",
        "follow",
        "unfollow",
    ],
}

DEFAULT_OBS_DESCRIPTION: dict[str, str] = {
    "time": "The current time step in the simulation. Provided as either an integer or a string in ISO datetime format.",
    "timedelta": "The time delta for each simulation step, represented as either an integer or a string in ISO format (e.g., '0:00:01' for 1 second).",
    "self_agent_id": "Your unique identifier.",
    "self_name": "Your name.",
    "self_pos": "Your current coordinates in the grid space, represented as a list of integers [x, y].",
    "self_init_pos": "Your home in the grid space at the beginning of the simulation, represented as a list of integers [x, y].",
    "self_is_moving": "A boolean indicating whether you are currently moving. If True, you will be moved to the target coordinates in the next time step. If False, you can choose whether to move elsewhere or stay in the same place in the next time step.",
    "self_destination": "If you are currently moving, this is the target you will be moved to in the next time step. If you are not currently moving, this will be null.",
    "others_pos": "A list of dictionaries representing the positions of other agents in the grid space. Each dictionary has 'agent_id', 'agent_name', and 'pos' (coordinates) of the other agent.",
    "self_inventory": "A dictionary representing your inventory, where the keys are item names and the values are the amounts of each item you have.",
    "self_tweet": "Your most recent tweet. If you have not tweeted yet, this will be null.",
    "visible_tl": "A list of tweets that you can see in your timeline. Each tweet is represented as a dictionary with 'agent_id' (the id of the agent who tweeted), 'agent_name' (the name of the agent who tweeted), and 'tweet' (the content of the tweet).",
    "recommended_follows": "A list of agent ids that you are not currently following. You may consider following these agents to receive more information from them.",
    "incoming_orders": "A list of orders that other agents have made to you. Each order is represented as a dictionary with 'order_id', 'counterparty_id', 'counterparty_name', 'item_name', 'item_amount', and 'ttl' (time to live, i.e., how many more time steps this order will be valid).",
    "incoming_proposals": "A list of proposals that other agents have made to you. Each proposal is represented as a dictionary with 'proposal_id', 'proposer_id', 'proposer_name', 'give_item_name', 'give_item_amount', 'get_item_name', 'get_item_amount', and 'ttl' (time to live, i.e., how many more time steps this proposal will be valid).",
    "item_name2price": "A dictionary mapping item names to their current prices in the market. This can help you make informed decisions about trading.",
    "others_inventory": "A list of dictionaries representing the inventories of other agents that are co-located with you in the same grid space. Each dictionary has 'agent_id', 'agent_name', and 'inventory' (which is itself a dictionary mapping item names to amounts) of the other agent.",
}

DEFAULT_ACTION_DESCRIPTION: dict[str, str] = {
    "move": "The move action is a list of integers representing the coordinates to move to. You can only move to adjacent grid spaces or stay in the same place in the next time step.",
    "consumptions": "The consumptions action is a list of items that you want to consume. Each item is represented as an object with 'item_name' and 'item_amount'. If the agent does not want to consume anything, it can set this to an empty list.",
    "orders": "The orders action is a list of orders that you want to make. Order is used for person to coorporation trade. Each order is represented as an object with 'counterparty_id', 'item_name', 'item_amount', and 'ttl'. To make an order, you must go to the same grid space as the counterparty agent.",
    "proposals": "The proposals action is a list of proposals that you want to make. Proposal is used for person to person trade. Each proposal is represented as an object with 'responder_agent_id', 'give_item_name', 'give_item_amount', 'get_item_name', 'get_item_amount', and 'ttl'. If the agent does not want to make any proposals, it can set this to an empty list.",
    "reactions": "The reactions action is a list of reactions that you want to make in response to incoming orders or proposals. Each reaction is represented as an object with 'kind' (either 'order' or 'proposal'), 'id' (the id of the order or proposal), 'accept_amount' (the amount of item accepted, or null if not applicable), and 'accept' (a boolean indicating whether to accept the order/proposal, or null if not applicable). If the agent does not want to make any reactions, it can set this to an empty list.",
    "tweet": "The tweet action is a string representing what you want to tweet.",
    "follow": "The follow action is an integer representing the id of the agent that you want to follow.",
    "unfollow": "The unfollow action is an integer representing the id of the agent that you want to unfollow.",
}
