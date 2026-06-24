from __future__ import annotations

import asyncio
import atexit
import copy
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import unicodedata
import urllib.request
from typing import Any, Optional, Type, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from .base import LLMClient
from .llm_client_utils import save_response_record_from_chat_completion


class VLLMClient(LLMClient):
    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[Any] = None,
        registered_classes: list[Type] = [],
    ) -> None:
        """Initialization.

        Args:
            config: Configuration dictionary.
                This dictionary controls both the vLLM server launch and the client-side
                request behavior. Required fields:
                - modelName (str):
                    Name or path of the model to be served by vLLM.
                    Example: "meta-llama/Meta-Llama-3-8B-Instruct"
                - vllmPython (str):
                    Absolute path to the Python executable in the dedicated vLLM
                    environment. This should point to an environment where vLLM
                    is installed.
                    Example: "/path/to/.venv-vllm/bin/python"
                Recommended fields (strongly suggested for stable usage):
                - gpuIds (list[int]):
                    List of GPU device IDs to be used by the vLLM server.
                    Example: [0], [0, 1]
                - timeOut (float, default=120.0):
                    Timeout (in seconds) for each API request.
                    Increase this for large models or long prompts.
                - maxRetries (int, default=5):
                    Number of retry attempts when API calls fail due to timeout,
                    connection errors, or rate limits.
                - ignoreServerErrors (bool, default=False):
                    If True, the client will ignore server-side errors and return an empty response.
                    If False, the client will raise exceptions on server errors.
                Optional fields (advanced tuning):
                # Server configuration
                - host (str, default="127.0.0.1"):
                    Host address for the OpenAI-compatible API server.
                - port (int):
                    Port number for the server. If not provided, a free port is chosen.
                    For stability, specifying a fixed port is recommended.
                - apiKey (str, default="EMPTY"):
                    API key used for authentication (dummy value is fine for local use).
                - serverStartTimeout (float, default=180.0):
                    Maximum time (seconds) to wait for the vLLM server to become ready.
                - serverLogPath (str):
                    Path to a file where vLLM server logs will be written.
                # Parallelism configuration
                - isDataParallel (bool, default=False):
                    If True, use data parallelism across GPUs.
                    If False, use tensor parallelism.
                # Model / inference behavior
                - dtype (str, default="auto"):
                    Data type for model weights (e.g., "float16", "bfloat16", "auto").
                - maxModelLen (int):
                    Maximum sequence length supported by the model.
                - gpuMemoryUtilization (float):
                    Fraction of GPU memory to utilize (e.g., 0.9).
                - enforceEager (bool, default=False):
                    If True, disables CUDA graph optimizations for improved stability
                    at the cost of performance.
                - trustRemoteCode (bool, default=False):
                    Whether to allow execution of remote model code.
                - servedModelName (str):
                    Name exposed via the API (useful when aliasing models).
                - quantization (str):
                    Quantization method for the model (e.g., "awq", "fp8").
                # Low-level vLLM arguments
                - vllmModule (str, default="vllm.entrypoints.openai.api_server"):
                    Python module used to launch the vLLM server.
                - vllmArgs (list[str]):
                    Additional command-line arguments passed directly to vLLM.
                # Client-side configuration
                - "llmRecordSavePath": path to save the generated prompts (optional, for debugging purposes).
                - "saveNumTokens": whether to save the number of tokens in the generated response (optional, default is False).
                - "savePromptResponsePair": whether to save the prompt-response pair (optional, default is False).
            prng: Optional pseudo-random number generator (not used in this client).
        """
        super().__init__(config, prng, registered_classes)
        self.use_gpu: bool = config.get("useGpu", True)
        if not self.use_gpu:
            raise ValueError("VLLMClient currently supports only useGpu=True.")
        self.gpu_ids: list[int] = list(config.get("gpuIds", []))
        if not self.gpu_ids:
            raise ValueError("VLLMClient requires non-empty 'gpuIds' when useGpu=True.")
        self.is_data_parallel: bool = config.get("isDataParallel", False)
        self.host: str = config.get("host", "127.0.0.1")
        self.port: int = int(config.get("port", self._find_free_port()))
        self.api_key: str = config.get("apiKey", "EMPTY")
        self.time_out: float = float(config.get("timeOut", 120.0))
        self.max_retries: int = int(config.get("maxRetries", 5))
        self.server_start_timeout: float = float(
            config.get("serverStartTimeout", 180.0)
        )
        self.ignore_server_errors: bool = bool(
            config.get("ignoreServerErrors", False)
        )
        self.dtype: str = config.get("dtype", "auto")
        self.trust_remote_code: bool = config.get("trustRemoteCode", False)
        self.max_model_len: Optional[int] = config.get("maxModelLen")
        self.gpu_memory_utilization: Optional[float] = config.get(
            "gpuMemoryUtilization"
        )
        self.enforce_eager: bool = config.get("enforceEager", False)
        self.served_model_name: Optional[str] = config.get("servedModelName")
        self.quantization: Optional[str] = config.get("quantization")
        self.vllm_python: str = config.get("vllmPython", sys.executable)
        self.server_log_path: Optional[str] = config.get("serverLogPath")
        if self.is_data_parallel:
            self.tensor_parallel_size = 1
            self.data_parallel_size = len(self.gpu_ids)
        else:
            self.tensor_parallel_size = len(self.gpu_ids)
            self.data_parallel_size = 1
        self.base_url: str = f"http://{self.host}:{self.port}/v1"
        self.json_schema: dict[str, Any] = json.loads(self._get_json_schema())
        self._server_proc: Optional[subprocess.Popen[str]] = None
        self._owns_server: bool = False
        self._server_log_fp: Optional[Any] = None
        self._server_lock = asyncio.Lock()
        self._start_server()
        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.time_out,
            max_retries=0,
        )
        atexit.register(self.close)

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return cast(int, sock.getsockname()[1])

    def _build_server_command(self) -> tuple[list[str], dict[str, str]]:
        if not os.path.exists(self.vllm_python):
            raise FileNotFoundError(f"vllmPython not found: {self.vllm_python!r}")
        env: dict[str, str] = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gid) for gid in self.gpu_ids)
        cmd: list[str] = [
            self.vllm_python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_name,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--api-key",
            self.api_key,
            "--dtype",
            self.dtype,
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--data-parallel-size",
            str(self.data_parallel_size),
        ]

        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])
        if self.gpu_memory_utilization is not None:
            cmd.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.served_model_name is not None:
            cmd.extend(["--served-model-name", self.served_model_name])
        if self.quantization is not None:
            cmd.extend(["--quantization", self.quantization])

        cmd.extend(list(self.config.get("vllmArgs", [])))
        return cmd, env

    def _probe_server(self) -> bool:
        url = f"{self.base_url}/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False

    def _open_server_log(self) -> Any:
        if self.server_log_path is None:
            return subprocess.DEVNULL
        log_dir = os.path.dirname(os.path.abspath(self.server_log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._server_log_fp = open(self.server_log_path, "a", encoding="utf-8")
        return self._server_log_fp

    def _start_server(self) -> None:
        if self._probe_server():
            self._owns_server = False
            return
        cmd, env = self._build_server_command()
        stdout_target = self._open_server_log()
        stderr_target = (
            self._server_log_fp if self._server_log_fp is not None else subprocess.PIPE
        )
        self._server_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            start_new_session=True,
        )
        self._owns_server = True
        deadline = time.time() + self.server_start_timeout
        while time.time() < deadline:
            if self._probe_server():
                return
            assert self._server_proc is not None
            retcode = self._server_proc.poll()
            if retcode is not None:
                stderr_text = ""
                if self._server_proc.stderr is not None:
                    try:
                        stderr_text = self._server_proc.stderr.read()
                    except Exception:
                        pass
                self.close()
                raise RuntimeError(
                    "Failed to start vLLM server.\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Return code: {retcode}\n"
                    f"stderr:\n{stderr_text}"
                )
            time.sleep(1.0)

        self.close()
        raise TimeoutError("Timed out waiting for vLLM server to become ready.")

    def _restart_server(self) -> None:
        self.close()
        time.sleep(1.0)
        self._start_server()

    async def _ensure_server_ready(self) -> None:
        if not self._probe_server():
            async with self._server_lock:
                if not self._probe_server():
                    await asyncio.to_thread(self._restart_server)

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        prompt = re.sub(r"<\|[^|>]*\|>", "", prompt)
        prompt = re.sub(r"<<[^>]*>>", "", prompt)
        prompt = re.sub(r"\[/?INST\]", "", prompt)
        prompt = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", prompt)
        prompt = unicodedata.normalize("NFKC", prompt)
        prompt = prompt.encode("utf-8", errors="replace").decode("utf-8")
        schema = copy.deepcopy(self.json_schema)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                await self._ensure_server_ready()
                async with self._sem:
                    response: ChatCompletion = (
                        await self.client.chat.completions.create(
                            model=self.served_model_name or self.model_name,
                            messages=[{"role": "user", "content": prompt}],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {
                                    "name": "agent_action",
                                    "strict": True,
                                    "schema": schema,
                                },
                            },
                        )
                    )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError(
                        "VLLMClient: Received empty response from vLLM server."
                    )
                parsed: Any = json.loads(content)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"VLLMClient: Expected JSON object in response, got: {parsed}"
                    )
                save_response_record_from_chat_completion(
                    response=response,
                    prompt=prompt,
                    record_config=self._get_llm_record_config(),
                )
                return cast(dict[str, Any], parsed)

            except BadRequestError as e:
                raise ValueError(
                    f"Bad request to vLLM/OpenAI-compatible server: {e}"
                ) from e

            except (
                APITimeoutError,
                APIConnectionError,
                InternalServerError,
                RateLimitError,
                json.JSONDecodeError,
                ValueError,
            ) as e:
                last_error = e
                await asyncio.sleep(min(2**attempt, 8))
                if isinstance(e, (APITimeoutError, APIConnectionError)):
                    async with self._server_lock:
                        if not self._probe_server():
                            await asyncio.to_thread(self._restart_server)
        
        if self.ignore_server_errors:
            return {}
        else:
            raise RuntimeError(
                f"VLLMClient: Failed after {self.max_retries} retries. "
                f"Last error: {last_error!r}"
            )

    def close(self) -> None:
        if not self._owns_server or self._server_proc is None:
            self._close_log_file()
            return
        proc = self._server_proc
        self._server_proc = None
        self._owns_server = False
        try:
            if proc.poll() is None:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                    proc.wait(timeout=5)
        finally:
            self._close_log_file()

    def _close_log_file(self) -> None:
        if self._server_log_fp is not None:
            try:
                self._server_log_fp.close()
            except Exception:
                pass
            self._server_log_fp = None

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    def __del__(self) -> None:
        pass
