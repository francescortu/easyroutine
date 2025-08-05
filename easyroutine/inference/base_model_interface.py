from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Literal, Union


@dataclass
class BaseInferenceModelConfig:
    """
    Configuration for the model interface.
    """
    model_name: str
    n_gpus: int = 1
    dtype: str = 'bfloat16'
    temperature: float = 0
    top_p: float = 0.95
    max_new_tokens: int = 5000
    
    

    
class BaseInferenceModel(ABC):
    """
    Base class for inference models.
    This class should be extended by specific model implementations.
    """
    
    def __init__(self, config: BaseInferenceModelConfig):
        self.config = config
    
    @classmethod
    def init_model(cls, model_name: str, n_gpus: int = 1, dtype: str = 'bfloat16') -> 'BaseInferenceModel':
        """
        Initialize the model with the given configuration.
        
        Arguments:
            model_name (str): Name of the model to initialize.
            n_gpus (int): Number of GPUs to use.
            dtype (str): Data type for the model.
        Returns:

            InferenceModel: An instance of the model.
        """
        config = BaseInferenceModelConfig(model_name=model_name, n_gpus=n_gpus, dtype=dtype)
        return cls(config)
    
    def append_with_chat_template(self, message:str, role:Literal['user', 'assistant', 'system'] = 'user', chat_history:List[dict[str,str]] = []) -> List[dict[str, str]]:
        """
        Apply chat template to the message.
        """
        # assert the chat_history
        if len(chat_history) > 0:
            assert all('role' in msg and 'content' in msg for msg in chat_history), "Chat history must contain 'role' and 'content' keys."
        # Append the new message to the chat history
        return chat_history + [{'role': role, 'content': message}]
    
    @abstractmethod
    def convert_chat_messages_to_custom_format(self, chat_messages: List[dict[str, str]]) -> Union[List[dict[str, str]], str]:
        """
        Convert chat messages to a custom format required by the model.
        
        Arguments:
            chat_messages (List[dict[str, str]]): List of chat messages to convert.
        
        Returns:
            Union[List[dict[str, str]], str]: Converted chat messages in the required format.
        """
        pass
    
    @abstractmethod
    def chat(self, chat_messages: list, **kwargs) -> list:
        """
        Generate a response based on the provided chat messages.
        
        Arguments:
            chat_messages (list): List of chat messages to process.
            **kwargs: Additional parameters for the model.
        
        Returns:
            str: The generated response from the model.
        """
        pass
    
    
    

    