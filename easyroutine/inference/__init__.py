"""
Model inference interfaces for various backends and deployment scenarios.

This package provides standardized interfaces for running inference across different
model backends and deployment strategies. It abstracts the complexities of different
inference engines while providing a consistent API for text generation and model
interaction.

Key Features:
    - Unified interface across different inference backends
    - Support for both single-GPU and multi-GPU deployments
    - Configurable generation parameters (temperature, top_p, etc.)
    - Chat template management for conversational models
    - Optimized inference with VLLM backend support
    - Extensible architecture for custom inference implementations

Supported Backends:
    - VLLM: High-performance inference engine with PagedAttention
    - Hugging Face Transformers: Direct model inference
    - Custom implementations: Extensible base classes

Main Components:
    - BaseInferenceModel: Abstract base class for all inference implementations
    - BaseInferenceModelConfig: Common configuration for inference parameters
    - VLLMInferenceModel: VLLM-based high-performance inference
    - VLLMInferenceModelConfig: VLLM-specific configuration options

Example Usage:
    >>> from easyroutine.inference import VLLMInferenceModel
    >>> 
    >>> # Initialize with VLLM backend
    >>> model = VLLMInferenceModel.init_model(
    ...     model_name="microsoft/DialoGPT-large",
    ...     n_gpus=2,
    ...     dtype="bfloat16"
    ... )
    >>> 
    >>> # Generate responses
    >>> response = model.generate("Hello, how are you?")
    >>> print(response)
    
    >>> # Multi-turn conversation
    >>> chat_history = []
    >>> chat_history = model.append_with_chat_template(
    ...     "What is machine learning?",
    ...     role="user",
    ...     chat_history=chat_history
    ... )
    >>> response = model.generate_with_chat_template(chat_history)

The package is designed to be backend-agnostic, allowing easy switching between
different inference engines based on performance requirements, memory constraints,
or deployment scenarios.
"""

from easyroutine.inference.base_model_interface import BaseInferenceModelConfig

# Optional VLLM imports - only available if vllm is installed
try:
    from easyroutine.inference.vllm_model_interface import VLLMInferenceModel, VLLMInferenceModelConfig
except ImportError:
    # VLLM is not available - this is optional
    pass