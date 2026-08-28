from typing import Any

DEFAULT_ACTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "sleep_duration": {
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "null"},
            ]
        },
        "move": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                {"type": "null"},
            ]
        },
        "consumptions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_name": {"type": "string"},
                    "item_amount": {"type": "number", "minimum": 1},
                },
                "required": ["item_name", "item_amount"],
            },
        },
        "orders": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "counterparty_id": {"type": "integer"},
                    "item_name": {"type": "string"},
                    "item_amount": {"type": "number", "minimum": 1},
                    "ttl": {"type": "integer", "minimum": 2},
                },
                "required": ["counterparty_id", "item_name", "item_amount", "ttl"],
            },
        },
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "responder_agent_id": {"type": "integer"},
                    "give_item_name": {"type": "string"},
                    "give_item_amount": {"type": "number", "minimum": 1},
                    "get_item_name": {"type": "string"},
                    "get_item_amount": {"type": "number", "minimum": 1},
                    "ttl": {"type": "integer", "minimum": 2},
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
        },
        "set_prices": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_name": {"type": "string"},
                    "price": {"type": "number", "minimum": 0},
                },
                "required": ["item_name", "price"],
            },
        },
        "inner_thought": {"type": "string"},
        "tweet": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "follow": {
            "anyOf": [
                {"type": "integer", "minimum": 0},
                {"type": "null"},
            ]
        },
        "unfollow": {
            "anyOf": [
                {"type": "integer", "minimum": 0},
                {"type": "null"},
            ]
        },
    },
    "required": [
        "sleep_duration",
        "move",
        "consumptions",
        "orders",
        "proposals",
        "reactions",
        "set_prices",
        "inner_thought",
        "tweet",
        "follow",
        "unfollow",
    ],
}

DEFAULT_SIMULATION_DESCRIPTION: str = "You are a member of the society. Based on the following observation, decide the action to take."

DEFAULT_OBS_DESCRIPTION: dict[str, str] = {
    "time": "The current time step in the simulation. Provided as either an integer or a string in ISO datetime format.",
    "timedelta": "The time delta for each simulation step, represented as either an integer or a string in ISO format (e.g., '0:00:01' for 1 second).",
    "self_agent_id": "Your unique identifier.",
    "self_name": "Your name.",
    "self_is_household": "A boolean indicating whether you are a household agent. If True, you are a household; if False, you are a firm agent (e.g., retailer or restaurant).",
    "self_is_sleeping": "A boolean indicating whether you are currently sleeping. If True, you are sleeping and will not be able to take any actions until you wake up. If False, you are awake and can take actions.",
    "memory": "A dictionary representing your memory, where the keys are memory keys and the values are the corresponding memory values."
    + 'Some memory entries may include an associated "stress" value. '
    + "A high stress value indicates that your needs or desires are not being satisfied and you are in a negative state."
    + "You should take these stress signals into account when deciding your behavior. "
    + "Use stress as an internal signal of dissatisfaction when interpreting your situation and deciding what to do.",
    "self_pos": "Your current coordinates in the grid space, represented as a list of integers [x, y].",
    "self_init_pos": "Your home in the grid space, represented as a list of integers [x, y]. "
    + "You may feel relaxed when you are at home, "
    + "while staying away from home for a long time may gradually cause stress to build up.",
    "self_is_moving": "A boolean indicating whether you are currently moving. "
    + "If True, you will automatically be moved to the target coordinates in the next time step. "
    + "If False, you can choose whether to move elsewhere or stay in the same place in the next time step.",
    "self_destination": "If you are currently moving (if self_is_moving is True), "
    + "this is the target you will be moved to in the next time step. "
    + "If you are not currently moving, this will be null.",
    "others_pos": "A list of dictionaries representing the positions of other agents in the grid space. "
    + "Each dictionary has 'agent_id', 'agent_name', 'is_household', and 'pos' (coordinates) of the other agent.",
    "nearby_agents": "A list of dictionaries representing the nearby agents in the neighboring grid cells. "
    + "Each dictionary has 'agent_id', 'agent_name', and 'pos' (coordinates) of the nearby agent. "
    + "You may feel crowded and stressed when there are many nearby agents. ",
    "self_salary": "Your salary that is supposed to be paid.",
    "self_inventory": "A dictionary representing your inventory, where the keys are item names and the values are the amounts of each item you have. "
    + "The items included in this inventory constitute the complete set of goods that exist in this world.",
    "self_tweet": "Your most recent tweet. If you have not tweeted yet, this will be null.",
    "follow_cap": "The maximum number of agents you can follow. If there is no limit, this will be null.",
    "num_followers": "The number of agents that are currently following you.",
    "num_follows": "The number of agents you are currently following.",
    "visible_tl": "A list of tweets that you can see in your timeline. "
    + "Each tweet is represented as a dictionary with 'agent_id' (the id of the agent who tweeted), "
    + "'agent_name' (the name of the agent who tweeted), and 'tweet' (the content of the tweet).",
    "recommended_follows": "A list of agents that you are not currently following but may consider following to connect with.",
    "incoming_orders": "A list of orders that other agents have made to you. "
    + "Each order is represented as a dictionary with 'order_id', 'counterparty_id', 'counterparty_name', "
    + "'item_name', 'item_amount', and 'ttl' (time to live, i.e., how many more time steps this order will be valid).",
    "incoming_proposals": "A list of proposals that other agents have made to you. "
    + "Each proposal is represented as a dictionary with 'proposal_id', 'proposer_id', 'proposer_name', "
    + "'give_item_name', 'give_item_amount', 'get_item_name', 'get_item_amount', and 'ttl' (time to live, i.e., how many more time steps this proposal will be valid).",
    "item_name2price": "A dictionary mapping item names to their current prices in the market. "
    + "This can help you make informed decisions about trading.",
    "others_inventory": "A list of dictionaries representing the inventories of other agents "
    + "that are co-located with you in the same grid space. "
    + "Each dictionary has 'agent_id', 'agent_name', 'is_household', and one entry per offered item. "
    + "Each item entry contains its posted price and available amount.",
}

DEFAULT_ACTION_DESCRIPTION: dict[str, str] = {
    "sleep_duration": "The sleep_duration action represents your intention to sleep for a certain duration. "
    + "If you want to sleep, specify the duration of sleep as either an integer (representing the number of time steps to sleep) or a string. "
    + "If you specify a string, it should be 'float' + 'unit', where 'float' is a positive floating-point number and 'unit' is one of 'm', 'h'. "
    + "For example, '5h' means sleeping for 5 hours. '30m' means sleeping for 30 minutes. "
    + "If you do not want to sleep, set this to null. "
    + "When you sleep, you will not be able to take any actions until you wake up. "
    + "Sleeping can help you recover from stress and feel better. "
    + "If you want to sleep, you must be located at your home (self_init_pos) to execute the sleep action.",
    "move": "The move action is a list of integers representing the destination coordinates you want to head toward. "
    + "Even if the destination is far away, you may specify any coordinates on the grid as your target. "
    + "The environment will automatically move you one step toward that destination at each time step "
    + "(i.e., movement per step is limited to adjacent grid spaces or staying in place). "
    + "Therefore, you do not need to manually specify intermediate positions; simply choose the location you ultimately want to reach. "
    + "You can refer to the others_pos field in the observation to decide where to go if you want to visit a store. "
    + "You can also refer to the self_init_pos field in the observation to go back home.",
    "consumptions": "The consumptions action is a list of items that you want to consume. "
    + "Each item is represented as an object with 'item_name' and 'item_amount'. "
    + "If you do not want to consume anything, it can set this to an empty list. "
    + "Don't specify item_name that you don't have in your inventory. "
    + "The specified item_amount must not exceed the available quantity of that item in your inventory. "
    + "You will be happy if you consume items that you like.",
    "orders": "The orders action represents a purchase at a posted price offered by a corporation (e.g., retailer or restaurant). "
    + "When you order, you accept the seller's price without negotiation. "
    + "Order is used for person-to-corporation trades under a posted-price mechanism. "
    + "Each order is represented as an object with 'counterparty_id', 'item_name', 'item_amount', and 'ttl' (must be more than 1). "
    + "You must be located in the same grid space as the corporation to execute the order. "
    + "You must have sufficient cash in your inventory to execute an order. "
    + "The total cost of the order (posted price * item_amount) must not exceed your available cash balance. "
    + "For the available items and their posted prices, refer to the others_inventory field in the observation. "
    + "Do not place an order for any item that does not exist in this world (i.e., any item not listed in self_inventory) "
    + "The item_amount must represent the quantity you actually want to buy. "
    + "Do not buy more than you can reasonably consume, store, or use.",
    "proposals": "The proposals action represents a bilateral trade proposal made to another individual agent. "
    + "When you propose a trade, you explicitly specify both what you give and what you request in return. "
    + "This action is used for person-to-person negotiation and may include money as one of the exchanged items. "
    + "Each proposal is represented as an object with 'responder_agent_id', 'give_item_name', 'give_item_amount', 'get_item_name', 'get_item_amount', and 'ttl' (must be more than 1). "
    + "If you do not want to make any proposals, it can set this to an empty list. "
    + "You must only specify items that exist in your own inventory (self_inventory) as give_item_name. "
    + "Furthermore, the specified give_item_amount must not exceed the available quantity of that item in your inventory. "
    + "Do not propose a trade for any item that does not exist in this world (i.e., any item not listed in self_inventory).",
    "reactions": "The reactions action is a list of reactions that you want to make in response to incoming orders or proposals. "
    + "If you are a household, ignore any orders. "
    + "Each reaction is represented as an object with 'kind' (either 'order' or 'proposal'), 'id' (the id of the order or proposal, not the agent id), 'accept_amount' (the amount of item accepted, or null if not applicable), and 'accept' (a boolean indicating whether to accept the order/proposal, or null if not applicable). "
    + "If you do not want to make any reactions, it can set this to an empty list. "
    + "You must reject the order/proposal if you do not have sufficient amount in your inventory to fulfill the accepted amount in the order/proposal. "
    + "For orders, you can only accept or reject the entire order.",
    "set_prices": "The set_prices action is a list of items for which you want to set the price. "
    + "Each item is represented as an object with 'item_name' and 'price'. "
    + "This action is only available to agents that are corporations (e.g., retailers or restaurants) "
    + "and is used to set the posted price for items that other agents can purchase from the corporation. "
    + "Do not specify any item_name that you don't have in your inventory. "
    + "The specified price must be a positive value.",
    "inner_thought": "A private internal thought that is never shown to other agents. "
    + "This field represents what you are currently thinking or feeling internally, "
    + "including your honest impressions and opinions about the people you follows. "
    + "You may express thoughts that you would never say publicly. "
    + "Unlike tweets, inner thoughts do not need to be polite, socially acceptable, or strategically filtered. "
    + "Write naturally as an internal monologue rather than as a public statement.",
    "tweet": "The tweet action is a string representing what you want to tweet. "
    + "Your tweet will be visible to agents that are following you and may also be visible to other agents. "
    + "You are not required to tweet at every step. "
    + 'If you do not have a strong opinion, emotional reaction, or relevant information, return an empty string "". '
    + "Only tweet when you genuinely feel it is worth sharing."
    + "Occasionally introduce new topics on your own, as real social media users often do. "
    + "Your tweet does not have to be limited to topics explicitly mentioned in the prompt "
    + "Avoid making every tweet about the simulation mechanics or your economic decisions.",
    "follow": "The follow action is an integer representing the id of the agent that you want to follow. "
    + "You may refer to the recommended_follows field in the observation for agents that you are not currently following "
    + "but may consider following to connect. "
    + "You cannot follow more agents than the follow_cap specified in the observation. "
    + "Refer to the num_follows field in the observation to see how many agents you are currently following. "
    + "Refer to the visible_tl field in the observation to see the tweets from agents you are currently following. "
    + "Do not follow an agent that you are already following.",
    "unfollow": "The unfollow action is an integer representing the id of the agent that you want to unfollow. "
    + "You cannot unfollow an agent that you are not currently following. "
    + "You must unfollow an agent if you want to follow a new agent but have already reached the follow_cap limit. "
    + "According to your inner thought, you may want to unfollow agents that you have negative feelings toward or that you think are not worth following.",
}
