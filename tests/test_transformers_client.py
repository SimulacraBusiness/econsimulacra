# from econsimulacra.llm_services import TransformersClient


# class TestTransformersClient:
#     config: dict[str, str] = {
#         "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
#         "device": "cuda",
#         "dtype": "float32",
#     }
#     def test_init(self) -> None:
#         client = TransformersClient(config=self.config)
#         assert client.config == self.config
#         assert client.json_generator is not None
#         test_out = client.json_generator("{}")
#         assert isinstance(test_out, dict)
