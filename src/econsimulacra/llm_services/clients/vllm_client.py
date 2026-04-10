from __future__ import annotations

import asyncio
import atexit
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional, cast

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion

from .base import LLMClient


class VLLMClient(LLMClient):
    """vLLM client backed by an internally managed OpenAI-compatible server.

    This client starts a vLLM server in a separate Python environment specified
    by `vllmPython`, so that the main environment can keep its own dependency
    set (e.g. outlines 0.1.11) without conflicting with vLLM's dependencies.

    Required config:
        - modelName: str

    Recommended config:
        - vllmPython: str
            Absolute path to the Python executable in the dedicated vLLM env.
            Example: "/path/to/.venv-vllm/bin/python"
        - useGpu: bool = True
        - gpuIds: list[int]
        - isDataParallel: bool = False

    Optional config:
        - host: str = "127.0.0.1"
        - port: int | None = None
        - apiKey: str = "EMPTY"
        - timeOut: float = 30.0
        - maxRetries: int = 3
        - serverStartTimeout: float = 180.0
        - dtype: str = "auto"
        - trustRemoteCode: bool = False
        - maxModelLen: int | None = None
        - gpuMemoryUtilization: float | None = None
        - enforceEager: bool = False
        - servedModelName: str | None = None
        - serverLogPath: str | None = None
        - vllmModule: str = "vllm.entrypoints.openai.api_server"
        - vllmArgs: list[str] = []

    Parallelism policy:
        - isDataParallel = False:
            tensor_parallel_size = len(gpuIds), data_parallel_size = 1
        - isDataParallel = True:
            tensor_parallel_size = 1, data_parallel_size = len(gpuIds)
    """

    def __init__(
        self,
        config: dict[str, Any],
        prng: Optional[Any] = None,
    ) -> None:
        super().__init__(config, prng)

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
        self.time_out: float = float(config.get("timeOut", 30.0))
        self.max_retries: int = int(config.get("maxRetries", 3))
        self.server_start_timeout: float = float(
            config.get("serverStartTimeout", 180.0)
        )

        self.dtype: str = config.get("dtype", "auto")
        self.trust_remote_code: bool = config.get("trustRemoteCode", False)
        self.max_model_len: Optional[int] = config.get("maxModelLen")
        self.gpu_memory_utilization: Optional[float] = config.get(
            "gpuMemoryUtilization"
        )
        self.enforce_eager: bool = config.get("enforceEager", False)
        self.served_model_name: Optional[str] = config.get("servedModelName")

        self.vllm_python: str = config.get("vllmPython", sys.executable)
        self.vllm_module: str = config.get(
            "vllmModule", "vllm.entrypoints.openai.api_server"
        )
        self.server_log_path: Optional[str] = config.get("serverLogPath")

        self.tensor_parallel_size: int
        self.data_parallel_size: int
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

        self._start_server()

        self.client: AsyncOpenAI = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.time_out,
            max_retries=self.max_retries,
        )

        atexit.register(self.close)

    @staticmethod
    def _find_free_port() -> int:
        """Find a free localhost TCP port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = cast(int, sock.getsockname()[1])
        return port

    def _build_server_command(self) -> tuple[list[str], dict[str, str]]:
        """Build the vLLM server command and environment."""
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

        extra_args: list[str] = list(self.config.get("vllmArgs", []))
        cmd.extend(extra_args)

        return cmd, env

    def _probe_server(self) -> bool:
        """Return True if the OpenAI-compatible server is responding."""
        url: str = f"{self.base_url}/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return False

    def _open_server_log(self) -> Any:
        """Open server log destination."""
        if self.server_log_path is None:
            return subprocess.DEVNULL

        log_dir: str = os.path.dirname(os.path.abspath(self.server_log_path))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._server_log_fp = open(self.server_log_path, "a", encoding="utf-8")
        return self._server_log_fp

    def _start_server(self) -> None:
        """Start the vLLM server and wait until it becomes ready."""
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

        deadline: float = time.time() + self.server_start_timeout

        while time.time() < deadline:
            if self._probe_server():
                return

            assert self._server_proc is not None
            retcode = self._server_proc.poll()
            if retcode is not None:
                if self._server_proc.stderr is not None:
                    try:
                        stderr_text = self._server_proc.stderr.read()
                    except Exception:
                        stderr_text = ""

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

    async def generate_response(self, prompt: str) -> dict[str, Any]:
        """Generate a structured JSON response from the vLLM server."""
        schema: dict[str, Any] = copy.deepcopy(self.json_schema)
        response: Optional[ChatCompletion] = None

        for _ in range(self.max_retries):
            try:
                async with self._sem:
                    response = await self.client.chat.completions.create(
                        model=self.served_model_name or self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "agent_action",
                                "strict": True,
                                "schema": schema,
                            },
                        },
                    )
                break

            except BadRequestError as e:
                print(f"[BadRequestError] {e}")
                return {}

            except RateLimitError as e:
                print(f"[RateLimitError] {e}")
                await asyncio.sleep(1)

            except APITimeoutError as e:
                print(f"[APITimeoutError] {e}")
                await asyncio.sleep(1)

            except APIConnectionError as e:
                print(f"[APIConnectionError] {e}")
                await asyncio.sleep(1)

        if response is None:
            raise ValueError("VLLMClient: Failed to get response after retries.")

        content: Optional[str] = response.choices[0].message.content
        if content is None:
            raise ValueError("VLLMClient: Received empty response from vLLM server.")

        try:
            parsed: Any = json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"VLLMClient: Failed to parse JSON response: {content}"
            ) from e

        if not isinstance(parsed, dict):
            raise ValueError(
                f"VLLMClient: Expected JSON object in response, got: {parsed}"
            )

        return cast(dict[str, Any], parsed)

    def close(self) -> None:
        """Terminate the internally managed vLLM server if owned by this client."""
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
        try:
            self.close()
        except Exception:
            pass
