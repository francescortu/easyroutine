"""
EasyRoutine: A comprehensive toolkit for AI model interpretability and analysis.

EasyRoutine is a powerful Python package that provides a collection of utilities
and tools for working with machine learning models, with a particular focus on
mechanistic interpretability of transformer models. It offers a unified interface
for model analysis, activation extraction, intervention studies, and more.

Key Features:
    - 🔍 Mechanistic Interpretability: Comprehensive tools for analyzing transformer
      model internals, including activation extraction, attention pattern analysis,
      and intervention studies
    - 🚀 High-Performance Inference: Optimized inference backends with support for
      VLLM and custom implementations
    - 📊 Progress Tracking: Adaptive progress bars that work in both interactive
      and batch environments
    - 🛠️ Utility Functions: Common utilities for file system navigation and
      robust logging
    - 🔧 Extensible Architecture: Modular design allowing easy extension and
      customization

Main Modules:
    interpretability: Core functionality for transformer model analysis
        - HookedModel: Wrapper for extracting activations from any transformer
        - ExtractionConfig: Fine-grained control over activation extraction
        - ActivationCache: Efficient storage and manipulation of activations
        - Intervention tools: Ablation studies and activation patching
        
    inference: Model inference interfaces for various backends
        - BaseInferenceModel: Abstract interface for inference implementations  
        - VLLMInferenceModel: High-performance inference with VLLM backend
        - Configuration classes for flexible inference setup
        
    console: Progress tracking and console utilities
        - progress(): tqdm-style progress bars with environment detection
        - LoggingProgress: Text-based progress for batch environments
        
    logger: Robust logging functionality
        - Structured logging with multiple output options
        - Level control and formatting customization
        - Integration with rich for enhanced console output
        
    utils: Common utility functions
        - File system navigation helpers
        - Path manipulation utilities

Quick Start:
    >>> import easyroutine
    >>> from easyroutine.interpretability import HookedModel, ExtractionConfig
    >>> from easyroutine.console import progress
    >>> 
    >>> # Load a model for interpretability analysis
    >>> model = HookedModel.from_pretrained("gpt2")
    >>> 
    >>> # Configure what to extract
    >>> config = ExtractionConfig(
    ...     extract_resid_out=True,
    ...     extract_attn_pattern=True
    ... )
    >>> 
    >>> # Extract activations with progress tracking
    >>> data = [{"input_ids": tokenized_inputs}]
    >>> cache = model.extract_cache(
    ...     progress(data, description="Extracting activations"),
    ...     target_token_positions=["last"],
    ...     extraction_config=config
    ... )

Installation:
    The package can be installed via pip:
    ```bash
    pip install easyroutine
    ```

For more detailed documentation and examples, visit the project repository.
"""

from .utils import path_to_parents
from .logger import logger