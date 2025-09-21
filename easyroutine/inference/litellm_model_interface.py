from easyroutine.inference.base_model_interface import (
    BaseInferenceModel,
    BaseInferenceModelConfig,
)

from easyroutine.inference.vllm_model_interface import VLLMServer
from vllm import LLM, SamplingParams
from typing import Union, List, Literal
from dataclasses import dataclass
from litellm import completion, batch_completion


@dataclass
class LiteLLMInferenceModelConfig(BaseInferenceModelConfig):
    """just a placeholder for now, as we don't have any specific config for VLLM."""

    model_name: str

    n_gpus: int = 0
    dtype: str = "bfloat16"
    temperature: float = 0
    top_p: float = 0.95
    max_new_tokens: int = 5000

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    xai_api_key: str = ""
    openrouter_api_key: str = ""


class LiteLLMInferenceModel(BaseInferenceModel):
    def __init__(self, config: LiteLLMInferenceModelConfig):
        """
        LiteLLM inference model interface.
        This class extends the BaseInferenceModel to provide specific functionality for LiteLLM.
        """
        self.config = config
        self.set_os_env()

        # Check if the model name indicates we should start a local VLLM server
        if self.config.model_name.startswith("hosted_vllm/"):
            print(
                f"Initializing hosted VLLM server for model: {self.config.model_name}"
            )
            assert config.n_gpus > 0, "Hosted VLLM models require at least one GPU."
            # Extract the actual model name by removing the "hosted_vllm/" prefix
            actual_model_name = self.config.model_name.replace("hosted_vllm/", "", 1)
            print(f"Using model: {actual_model_name}")

            try:
                self.vllm_server = VLLMServer(
                    model=actual_model_name,
                    n_gpus=self.config.n_gpus,
                    dtype=self.config.dtype,
                )

                server_url = self.vllm_server.return_url()
                print(f"VLLM server started at: {server_url}")

                self.additional_args = {
                    "api_base": server_url,
                }
            except Exception as e:
                print(f"Error starting VLLM server: {e}")
                raise e
        else:
            print(
                f"Using external LiteLLM provider for model: {self.config.model_name}"
            )
            self.vllm_server = None
            self.additional_args = {}

    def __del__(self):
        """
        Cleanup resources when the object is deleted.
        This ensures the VLLM server is properly stopped.
        """
        if hasattr(self, "vllm_server") and self.vllm_server:
            try:
                print("Cleaning up VLLM server...")
                self.vllm_server.stop()
            except Exception as e:
                print(f"Error stopping VLLM server: {e}")

    def set_os_env(self):
        """
        This method sets the environment variables for the API keys.
        """
        import os

        os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
        os.environ["ANTHROPIC_API_KEY"] = self.config.anthropic_api_key
        os.environ["XAI_API_KEY"] = self.config.xai_api_key
        os.environ["OPENROUTER_API_KEY"] = self.config.openrouter_api_key
        
    def convert_chat_messages_to_custom_format(
        self, chat_messages: List[dict[str, str]]
    ) -> List[dict[str, str]]:
        """
        For now, LiteLLM is compatible with the chat template format we use.
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

        response = completion(
            model=self.config.model_name,
            messages=chat_messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            **self.additional_args,
        )
        return response["choices"]

    def batch_chat(
        self, chat_messages: List[List[dict[str, str]]], use_tqdm=False, **kwargs
    ) -> List[list]:
        """
        Generate responses for a batch of chat messages.

        Arguments:
            chat_messages (List[List[dict[str, str]]]): List of chat messages to process.
            **kwargs: Additional parameters for the model.

        Returns:
            List[list]: List of generated responses from the model.
        """
        chat_messages = [
            self.convert_chat_messages_to_custom_format(msg) for msg in chat_messages
        ]

        responses = batch_completion(
            model=self.config.model_name,
            messages=chat_messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_new_tokens,
            **self.additional_args,
        )
        return responses

    def get_last_text_from_response(self, response)-> str:
        return response["choices"][-1]["message"]["content"]