from easyroutine.inference.base_model_interface import BaseInferenceModel, BaseInferenceModelConfig
from vllm import LLM, SamplingParams
from typing import Union, List, Literal
from vllm.entrypoints.chat_utils import ChatCompletionMessageParam
from dataclasses import dataclass

@dataclass
class VLLMInferenceModelConfig(BaseInferenceModelConfig):
    """just a placeholder for now, as we don't have any specific config for VLLM."""
    
class VLLMInferenceModel(BaseInferenceModel):
    """
    VLLM inference model interface.
    This class extends the BaseInferenceModel to provide specific functionality for VLLM.
    """

    def __init__(self, config: BaseInferenceModelConfig):
        super().__init__(config)
        self.model = LLM(model=config.model_name, tensor_parallel_size=config.n_gpus, dtype=config.dtype)
        
        
    def convert_chat_messages_to_custom_format(self, chat_messages: List[dict[str, str]]) -> List[dict[str, str]]:
        """
        For now, VLLM is compatible with the chat template format we use.
        """
        return chat_messages

    def chat(self, chat_messages: List[dict[str, str]], use_tqdm=False, **kwargs) -> list:
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
            max_tokens=self.config.max_new_tokens
        )

        
        # Generate response using VLLM
        response = self.model.chat(chat_messages, sampling_params=sampling_params, use_tqdm=use_tqdm) # type: ignore
        
        return response