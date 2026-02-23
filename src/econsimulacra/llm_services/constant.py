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
                {"type": "null"},
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
                    "ttl": {"type": "integer"}#, "minimum": 1},
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
                    "ttl": {"type": "integer"}#, "minimum": 1},
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
                    "accept_amount": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                    "accept": {"anyOf": [{"type": "boolean"}, {"type": "null"}]},
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
