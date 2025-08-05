from easyroutine.inference.base_model_interface import BaseInferenceModel, BaseInferenceModelConfig
from vllm import LLM, SamplingParams
from typing import Union, List, Literal
from dataclasses import dataclass
from litellm import completion, batch_completion

@dataclass
class LiteLLMInferenceModelConfig(BaseInferenceModelConfig):
    """just a placeholder for now, as we don't have any specific config for VLLM."""
    model_name: str

    n_gpus: int = 0
    dtype: str = 'bfloat16'
    temperature: float = 0
    top_p: float = 0.95
    max_new_tokens: int = 5000
    
    openai_api_key: str = ''
    anthropic_api_key: str = ''
    xai_api_key: str = ''
    
class LiteLLMInferenceModel(BaseInferenceModel):
    
    def __init__(self, config: LiteLLMInferenceModelConfig):
        self.config = config
        self.set_os_env()
        
    def set_os_env(self):
        import os
        os.environ['OPENAI_API_KEY'] = self.config.openai_api_key
        os.environ['ANTHROPIC_API_KEY'] = self.config.anthropic_api_key
        os.environ['XAI_API_KEY'] = self.config.xai_api_key
        
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
        
        
        response = completion(
            model = self.config.model_name,
            messages = chat_messages,
            temperature = self.config.temperature,
            top_p = self.config.top_p,
            max_tokens = self.config.max_new_tokens,
        )
        return response['choices']
    
    def batch_chat(self, chat_messages: List[List[dict[str, str]]], use_tqdm=False, **kwargs) -> List[list]:
        """
        Generate responses for a batch of chat messages.
        
        Arguments:
            chat_messages (List[List[dict[str, str]]]): List of chat messages to process.
            **kwargs: Additional parameters for the model.
        
        Returns:
            List[list]: List of generated responses from the model.
        """
        chat_messages = [self.convert_chat_messages_to_custom_format(msg) for msg in chat_messages]
        
        responses = batch_completion(
            model = self.config.model_name,
            messages = chat_messages,
            temperature = self.config.temperature,
            top_p = self.config.top_p,
            max_tokens = self.config.max_new_tokens,
        )
        return responses