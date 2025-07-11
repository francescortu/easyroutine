from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import List, Literal, Union


@dataclass
class BaseInferenceModelConfig:
    """
    Base configuration class for inference model implementations.
    
    This configuration class provides common parameters needed for model
    inference across different backends and implementations. It establishes
    a standard interface for configuring model loading, generation parameters,
    and hardware utilization.
    
    Attributes:
        model_name (str): Identifier of the model to load. Can be:
            - Hugging Face model repository name (e.g., "gpt-3.5-turbo")
            - Local path to a model directory
            - Model name supported by the specific inference backend
            
        n_gpus (int, optional): Number of GPUs to utilize for model inference.
            Determines parallel processing capability and memory distribution.
            Defaults to 1.
            
        dtype (str, optional): Data type precision for model parameters.
            Common options include:
            - "bfloat16": Good balance of speed and accuracy
            - "float16": Faster inference, potential accuracy loss
            - "float32": Full precision, slower inference
            Defaults to "bfloat16".
            
        temperature (float, optional): Sampling temperature for text generation.
            Controls randomness in generation:
            - 0.0: Deterministic, always select most likely token
            - 0.1-0.7: Low randomness, more focused responses
            - 0.8-1.2: Higher randomness, more creative responses
            Defaults to 0 (deterministic).
            
        top_p (float, optional): Nucleus sampling parameter. Only tokens with
            cumulative probability <= top_p are considered for sampling.
            Range: [0.0, 1.0]. Lower values make output more focused.
            Defaults to 0.95.
            
        max_new_tokens (int, optional): Maximum number of new tokens to generate
            in a single inference call. Controls output length and prevents
            runaway generation. Defaults to 5000.
    
    Example:
        >>> config = BaseInferenceModelConfig(
        ...     model_name="microsoft/DialoGPT-large",
        ...     n_gpus=2,
        ...     dtype="bfloat16",
        ...     temperature=0.7,
        ...     max_new_tokens=1000
        ... )
        >>> model = SomeInferenceModel(config)
    
    Note:
        This is a base configuration class. Specific inference implementations
        may extend this with additional parameters relevant to their backend.
    """
    model_name: str
    n_gpus: int = 1
    dtype: str = 'bfloat16'
    temperature: float = 0
    top_p: float = 0.95
    max_new_tokens: int = 5000
    
    

    
class BaseInferenceModel(ABC):
    """
    Abstract base class for inference model implementations.
    
    This class defines the standard interface that all inference model implementations
    should follow, ensuring consistency across different backends (VLLM, Hugging Face,
    custom implementations, etc.). It provides common functionality and enforces
    implementation of essential methods through abstract methods.
    
    The base class handles configuration management and provides utility methods
    for common inference tasks like chat template application and message formatting.
    Subclasses should implement the specific inference logic for their backend.
    
    Key Design Principles:
        - Uniform interface across different inference backends
        - Configuration-driven initialization and behavior
        - Support for both single-turn and multi-turn conversations
        - Extensible for backend-specific optimizations
        - Thread-safe inference operations
    
    Attributes:
        config (BaseInferenceModelConfig): Configuration object containing
            model parameters, generation settings, and hardware specifications.
    
    Abstract Methods:
        Subclasses must implement backend-specific methods for:
        - Model loading and initialization
        - Text generation and inference
        - Resource management and cleanup
    
    Example:
        >>> class MyInferenceModel(BaseInferenceModel):
        ...     def __init__(self, config):
        ...         super().__init__(config)
        ...         # Initialize specific backend
        ...     
        ...     def generate(self, prompt):
        ...         # Implement generation logic
        ...         pass
        >>> 
        >>> model = MyInferenceModel.init_model("gpt2", n_gpus=1)
        >>> response = model.generate("Hello, world!")
    
    Note:
        This class uses the ABC (Abstract Base Class) pattern to ensure
        all subclasses implement required methods. Direct instantiation
        of this class will raise a TypeError.
    """
    
    def __init__(self, config: BaseInferenceModelConfig):
        self.config = config
    
    @classmethod
    def init_model(cls, model_name: str, n_gpus: int = 1, dtype: str = 'bfloat16') -> 'BaseInferenceModel':
        """
        Class method for convenient model initialization with minimal configuration.
        
        This factory method provides a streamlined way to create model instances
        with common default settings, automatically constructing the configuration
        object and initializing the model. It's designed for quick setup scenarios
        where detailed configuration isn't needed.
        
        Args:
            model_name (str): Identifier of the model to initialize. Accepts:
                - Hugging Face model repository names
                - Local model paths
                - Any model identifier supported by the implementation
                
            n_gpus (int, optional): Number of GPUs to allocate for the model.
                Determines parallelization and memory distribution strategy.
                Must be > 0. Defaults to 1.
                
            dtype (str, optional): Precision/data type for model parameters.
                Affects inference speed and memory usage:
                - "bfloat16": Recommended for most use cases
                - "float16": Faster but may affect accuracy
                - "float32": Full precision, slower
                Defaults to "bfloat16".
        
        Returns:
            BaseInferenceModel: An initialized instance of the implementing class
                ready for inference operations.
        
        Example:
            >>> # Quick initialization with defaults
            >>> model = MyInferenceModel.init_model("gpt2")
            >>> 
            >>> # Custom GPU and precision settings
            >>> model = MyInferenceModel.init_model(
            ...     model_name="microsoft/DialoGPT-large",
            ...     n_gpus=2,
            ...     dtype="float16"
            ... )
            >>> 
            >>> # Ready for inference
            >>> response = model.generate("Hello!")
        
        Note:
            This method creates a BaseInferenceModelConfig with default values
            for unspecified parameters. For more detailed configuration, create
            a custom config object and use the regular constructor.
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
    
    
    

    