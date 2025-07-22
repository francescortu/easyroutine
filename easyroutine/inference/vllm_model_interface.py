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

    def __init__(self, config: BaseInferenceModelConfig):
        super().__init__(config)
        self.model = LLM(
            model=config.model_name,
            tensor_parallel_size=config.n_gpus,
            dtype=config.dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            # max_seq_len_to_capture=1000,
            max_model_len=config.max_model_len,
        )

    def convert_chat_messages_to_custom_format(
        self, chat_messages: List[dict[str, str]]
    ) -> List[dict[str, str]]:
        """
        For now, VLLM is compatible with the chat template format we use.
        """
        return chat_messages

    def chat(
        self, chat_messages: List[dict[str, str]], use_tqdm=False, **kwargs
    ) -> list:
        """
        Generate a response based on the provided chat messages.

        Arguments:
            chat_messages (List[dict[str, str]]): List of chat messages to process.
            **kwargs: Additional parameters for the model.

        Returns:
            str: The generated response from the model.
        """
        chat_messages = self.convert_chat_messages_to_custom_format(chat_messages)

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
        )

        # Generate response using VLLM
        response = self.model.chat(
            chat_messages, sampling_params=sampling_params, use_tqdm=use_tqdm
        )  # type: ignore

        return response
    
    def batch_chat(
        self, chat_messages: List[List[dict[str, str]]], use_tqdm=False, **kwargs
    ) -> List[str]:
        return self.chat(
            chat_messages=chat_messages, use_tqdm=use_tqdm, **kwargs
        )

    def get_last_text_from_response(self, response)-> str:
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
            for _ in progress(range(50), desc="Starting VLLM Server", force_batch_mode=True):
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process ended prematurely - get error output
                    out, err = self.process.communicate()
                    error_message = err
                    raise RuntimeError(f"VLLM server failed to start: {err}")

                # Check for specific output indicating the server is ready
                line = self.process.stdout.readline()
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
