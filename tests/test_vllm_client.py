# import asyncio

# from econsimulacra.llm_services import VLLMClient


# class TestVLLMClient:
#     config = {
#         "modelName": "meta-llama/Meta-Llama-3-8B-Instruct",
#         "vllmPython": "/home/m2023rhashimoto/econsimulacra/.venv-vllm/bin/python",
#         "useGpu": True,
#         "gpuIds": [0, 1],
#         "isDataParallel": True,
#         "host": "127.0.0.1",
#         "port": 8000,
#         "dtype": "auto",
#         "timeOut": 60,
#         "maxRetries": 3,
#         "serverStartTimeout": 300,
#         "maxConcurrentGenerations": 32,
#         "trustRemoteCode": False,
#         "gpuMemoryUtilization": 0.9,
#         "serverLogPath": "logs/vllm_server.log"
#     }

#     def test_init(self) -> None:
#         client = VLLMClient(config=self.config)
#         assert client.config == self.config
#         out = asyncio.run(client.generate_response("{}"))
#         assert isinstance(out, dict)
