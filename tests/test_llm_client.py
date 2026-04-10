import json

from econsimulacra.llm_services import DEFAULT_ACTION_JSON_SCHEMA, LLMClient

MODIFIED_ACTION_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "move": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                    },
                    "minItems": 2,
                    "maxItems": 2,
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
                    "item_name": {"type": "string", "enum": ["item1", "item2"]},
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
                    "counterparty_id": {"type": "integer", "minimum": 0, "maximum": 2},
                    "item_name": {"type": "string", "enum": ["item1", "item2"]},
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
                    "responder_agent_id": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2,
                    },
                    "give_item_name": {"type": "string", "enum": ["item1", "item2"]},
                    "give_item_amount": {"type": "number", "minimum": 1},
                    "get_item_name": {"type": "string", "enum": ["item1", "item2"]},
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
        "set_price": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_name": {"type": "string", "enum": ["item1", "item2"]},
                    "price": {"type": "number", "minimum": 0},
                },
                "required": ["item_name", "price"],
            },
        },
        "tweet": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "follow": {
            "anyOf": [
                {"type": "integer", "minimum": 0, "maximum": 2},
                {"type": "null"},
            ]
        },
        "unfollow": {
            "anyOf": [
                {"type": "integer", "minimum": 0, "maximum": 2},
                {"type": "null"},
            ]
        },
    },
    "required": [
        "move",
        "consumptions",
        "orders",
        "proposals",
        "reactions",
        "set_price",
        "tweet",
        "follow",
        "unfollow",
    ],
}


class DummyClient(LLMClient):
    async def generate_response(self, prompt: str) -> dict[str, str]:
        return {"response": f"Echo: {prompt}"}


class TestLLMClient:
    def test_init(self) -> None:
        client = DummyClient({"modelName": "dummy"})
        assert client.config == {"modelName": "dummy"}
        assert client._get_json_schema() == json.dumps(DEFAULT_ACTION_JSON_SCHEMA)
        client = DummyClient(
            config={
                "modelName": "dummy",
                "jsonSchemaPath": "tests/dummy_action_schema.json",
            }
        )
        assert client._get_json_schema() != json.dumps(DEFAULT_ACTION_JSON_SCHEMA)
        obtained_schema = json.loads(client._get_json_schema())
        assert obtained_schema == json.load(
            open("tests/dummy_action_schema.json", "r", encoding="utf-8")
        )
        del obtained_schema["dummy"]
        assert obtained_schema == DEFAULT_ACTION_JSON_SCHEMA

    def test_modify_json_schema(self) -> None:
        config = {
            "modelName": "dummy",
            "modifySchema": True,
            "gridSpace": [9, 10],
            "items": ["item1", "item2"],
            "numAgents": 3,
        }
        client = DummyClient(config)
        modified_schema_str = client._get_json_schema()
        assert modified_schema_str != json.dumps(DEFAULT_ACTION_JSON_SCHEMA)
        modified_schema = json.loads(modified_schema_str)
        assert modified_schema == MODIFIED_ACTION_JSON_SCHEMA
