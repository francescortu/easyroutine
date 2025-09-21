from easyroutine.inference.base_model_interface import (
    BaseInferenceModel,
    BaseInferenceModelConfig,
)
from easyroutine.console import progress
from vllm import LLM, SamplingParams
from typing import Union, List, Literal, Optional
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from dataclasses import dataclass
import subprocess
import time
import os

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # torch might not be available in CPU-only envs


@dataclass
class VLLMInferenceModelConfig(BaseInferenceModelConfig):
    """just a placeholder for now, as we don't have any specific config for VLLM."""

    gpu_memory_utilization: float = 0.8
    max_model_len: Optional[int] = None


class VLLMInferenceModel(BaseInferenceModel):
    """
    VLLM inference model interface.
    This class extends the BaseInferenceModel to provide specific functionality for VLLM.
    """

    def __init__(self, config: VLLMInferenceModelConfig):
        super().__init__(config)
        # Determine visible GPUs and cap tensor parallel size to avoid NCCL init failures
        requested_tp = getattr(config, "n_gpus", 1) or 1
        visible_gpus = None
        if torch is not None and torch.cuda.is_available():
            try:
                visible_gpus = torch.cuda.device_count()
            except Exception:
                visible_gpus = None

        if visible_gpus is None:
            # Fallback to CUDA_VISIBLE_DEVICES env if torch couldn't tell us
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if cuda_visible:
                # count comma-separated ids excluding blanks
                visible_gpus = len([x for x in cuda_visible.split(",") if x != ""])

        if visible_gpus is None:
            # Unknown visibility, assume at least 1 if CUDA might be present
            visible_gpus = 0

        if visible_gpus == 0 and requested_tp > 0:
            # No GPUs visible: fail fast with a clear message
            raise RuntimeError(
                "No CUDA GPUs are visible to the process, but tensor_parallel_size>0 was requested. "
                "Ensure you are on a GPU node and CUDA_VISIBLE_DEVICES is set."
            )

        if requested_tp > visible_gpus:
            # Cap TP and inform the user to prevent NCCL TCPStore worker crashes
            print(
                f"[VLLM] Requested tensor_parallel_size={requested_tp} but only {visible_gpus} GPU(s) are visible. "
                f"Capping tensor_parallel_size to {visible_gpus} to avoid NCCL init failure."
            )
            requested_tp = max(1, visible_gpus)

        try:
            self.model = LLM(
                model=config.model_name,
                tensor_parallel_size=requested_tp,
                dtype=config.dtype,  # type: ignore[arg-type]
                gpu_memory_utilization=getattr(config, "gpu_memory_utilization", 0.8),
                # max_seq_len_to_capture=1000,
                max_model_len=getattr(config, "max_model_len", None),
            )
        except Exception as e:
            # Provide actionable hints for common startup failures
            hints = [
                f"model={getattr(config, 'model_name', 'unknown')}",
                f"dtype={getattr(config, 'dtype', 'bfloat16')}",
                f"requested_tp={getattr(config, 'n_gpus', 1)}",
                f"effective_tp={requested_tp}",
                f"visible_gpus={visible_gpus}",
            ]
            raise RuntimeError(
                "Failed to initialize vLLM LLM engine. "
                "This often happens when tensor_parallel_size exceeds the number of visible GPUs, "
                "when NCCL cannot initialize between ranks, or due to GPU OOM. "
                "Details: " + ", ".join(hints) + f". Original error: {e}"
            )

    def convert_chat_messages_to_custom_format(
        self, chat_messages: List[dict[str, str]]
    ) -> List[dict[str, str]]:
        """
        For now, VLLM is compatible with the chat template format we use.
        """
        return chat_messages

    def chat(
        self,
        chat_messages: Union[List[dict[str, str]], List[List[dict[str, str]]]],
        use_tqdm=False,
        **kwargs,
    ) -> list:
        """
        Generate a response based on the provided chat messages.

        Arguments:
            chat_messages (List[dict[str, str]]): List of chat messages to process.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The generated response from the model.
        """
        # Normalize/convert messages for single or batched input
        if (
            isinstance(chat_messages, list)
            and chat_messages
            and isinstance(chat_messages[0], dict)
        ):
            # Single conversation (list[dict])
            chat_messages = self.convert_chat_messages_to_custom_format(  # type: ignore[assignment]
                chat_messages  # type: ignore[arg-type]
            )
        elif (
            isinstance(chat_messages, list)
            and chat_messages
            and isinstance(chat_messages[0], list)
        ):
            # Batch of conversations (list[list[dict]])
            chat_messages = [
                self.convert_chat_messages_to_custom_format(  # type: ignore[misc]
                    msgs  # type: ignore[arg-type]
                )
                for msgs in chat_messages
            ]

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )

        # Generate response using VLLM
        response = self.model.chat(
            chat_messages, sampling_params=sampling_params, use_tqdm=use_tqdm
        )  # type: ignore[arg-type]

        return response

    def batch_chat(
        self, chat_messages: List[List[dict[str, str]]], use_tqdm=False, **kwargs
    ) -> list:
        return self.chat(chat_messages=chat_messages, use_tqdm=use_tqdm, **kwargs)

    def get_last_text_from_response(self, response) -> str:
        return response.outputs[-1].text


class VLLMServer:
    """
    Class to manage the VLLM server. It starts the server with the specified model and configuration.
    It runs a subprocess to start the server and waits for it to be ready.
    """

    def __init__(
        self,
        model: str,
        port: int = 8000,
        host: str = "localhost",
        n_gpus: int = 1,
        dtype: str = "bfloat16",
    ):
        """
        Initialize and start the VLLM server.

        Args:
            model: Name/path of the model to serve
            port: Port to serve on
            host: Host to serve on
            n_gpus: Number of GPUs to use
            dtype: Data type for the model
        """
        self.host = host
        self.port = port
        self.model = model

        self.cmd = [
            "vllm",
            "serve",
            model,
            "--tensor-parallel-size",
            str(n_gpus),
            "--dtype",
            dtype,
            "--port",
            str(port),
            "--host",
            host,
        ]

        print(f"Starting VLLM server with command: {' '.join(self.cmd)}")

        # Start the server process
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,  # This makes output strings instead of bytes
            )

            # Wait for the server to start by checking output
            server_started = False
            error_message = ""

            # redirect the output for the following 1min
            for _ in progress(
                range(50), desc="Starting VLLM Server", force_batch_mode=True
            ):
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process ended prematurely - get error output
                    out, err = self.process.communicate()
                    error_message = err
                    raise RuntimeError(f"VLLM server failed to start: {err}")

                # Check for specific output indicating the server is ready
                stdout = self.process.stdout
                line = stdout.readline() if stdout is not None else ""
                if "Server started" in line:
                    server_started = True
                    break

                # time.sleep(1)

            if not server_started:
                if error_message:
                    raise RuntimeError(
                        f"VLLM server didn't start properly. Errors: {error_message}"
                    )
                else:
                    raise TimeoutError("VLLM server didn't start in the expected time")

            print(f"VLLM server started successfully at {self.return_url()}")

        except Exception as e:
            raise RuntimeError(f"Error starting VLLM server: {str(e)}")

    def return_url(self):
        """
        Returns the URL of the VLLM server.
        """
        return f"http://{self.host}:{self.port}"

    def stop(self):
        """
        Stop the VLLM server.
        """
        if hasattr(self, "process") and self.process:
            print("Stopping VLLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                print("VLLM server stopped.")
            except subprocess.TimeoutExpired:
                print("VLLM server didn't terminate, killing it.")
                self.process.kill()
                self.process.wait()
