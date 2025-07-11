# EasyRoutine

A comprehensive Python toolkit for AI model interpretability, analysis, and efficient inference. EasyRoutine provides researchers and practitioners with powerful tools for understanding transformer models, extracting internal activations, and performing mechanistic interpretability studies.

## 🚀 Key Features

- **🔍 Mechanistic Interpretability**: Deep analysis of transformer model internals
  - Extract activations from any model component (residual streams, attention layers, MLPs)
  - Support for attention pattern analysis and head-specific extractions
  - Intervention capabilities for ablation studies and activation patching
  - Works with both language models and vision-language models

- **⚡ High-Performance Inference**: Optimized model inference with multiple backends
  - VLLM integration for high-throughput inference
  - Multi-GPU support and memory optimization
  - Configurable generation parameters and chat templates

- **📊 Smart Progress Tracking**: Adaptive progress bars for any environment
  - Rich progress bars for interactive use
  - Clean text logging for batch jobs (SLURM, PBS, etc.)
  - Automatic environment detection

- **🛠️ Essential Utilities**: Common tools for ML workflows
  - Robust logging with multiple output formats
  - File system navigation helpers
  - Memory management utilities

## 📦 Installation

```bash
pip install easyroutine
```

For development installation:
```bash
git clone https://github.com/francescortu/easyroutine.git
cd easyroutine
pip install -e .
```

## 🔍 Interpretability - Quick Start

### Basic Activation Extraction

```python
from easyroutine.interpretability import HookedModel, ExtractionConfig
from easyroutine.console import progress

# Load any Hugging Face transformer model
model = HookedModel.from_pretrained(
    model_name="gpt2",  # or any HF model
    device_map="auto"
)

# Prepare your data
texts = ["Hello, world!", "How are you today?"]
tokenizer = model.get_tokenizer()
dataset = [tokenizer(text, return_tensors="pt") for text in texts]

# Configure what activations to extract
config = ExtractionConfig(
    extract_resid_out=True,        # Residual stream outputs
    extract_attn_pattern=True,     # Attention patterns  
    extract_mlp_out=True,          # MLP layer outputs
    save_input_ids=True            # Keep track of tokens
)

# Extract activations with progress tracking
cache = model.extract_cache(
    progress(dataset, description="Extracting activations"),
    target_token_positions=["last"],  # Focus on final token
    extraction_config=config
)

# Access extracted data
residual_activations = cache["resid_out_0"]  # Layer 0 residual outputs
attention_patterns = cache["attn_pattern_0_5"]  # Layer 0, Head 5 attention
print(f"Extracted activations: {list(cache.keys())}")
```

### Advanced: Intervention Studies

```python
from easyroutine.interpretability import Intervention

# Define interventions for causal analysis
interventions = [
    Intervention(
        component="resid_out_5",  # Target layer 5 residual stream
        intervention_type="ablation",  # Zero out activations
        positions=["last"]  # Only affect last token
    )
]

# Run model with interventions
cache_with_intervention = model.extract_cache(
    dataset,
    target_token_positions=["last"],
    extraction_config=config,
    interventions=interventions
)

# Compare original vs. intervened activations
original_logits = cache["logits"]
intervened_logits = cache_with_intervention["logits"] 
effect = original_logits - intervened_logits
```

### Vision-Language Models

```python
# Works seamlessly with VLMs like LLaVA, Pixtral, etc.
vlm = HookedModel.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")

# Process multimodal inputs
processor = vlm.get_processor()
inputs = processor(images=image, text="What do you see?", return_tensors="pt")

# Extract activations from multimodal processing
cache = vlm.extract_cache(
    [inputs],
    target_token_positions=["last-image", "end-image"],
    extraction_config=ExtractionConfig(extract_resid_out=True)
)
```

## ⚡ High-Performance Inference

```python
from easyroutine.inference import VLLMInferenceModel

# Initialize high-performance inference engine
model = VLLMInferenceModel.init_model(
    model_name="microsoft/DialoGPT-large",
    n_gpus=2,
    dtype="bfloat16"
)

# Generate responses
response = model.generate("Hello, how can I help you today?")

# Multi-turn conversations with chat templates
chat_history = []
chat_history = model.append_with_chat_template(
    "What is machine learning?",
    role="user",
    chat_history=chat_history
)
response = model.generate_with_chat_template(chat_history)
```

## 📊 Smart Progress Tracking

EasyRoutine automatically adapts progress display to your environment:

```python
from easyroutine.console import progress

# Works great in Jupyter notebooks, terminals, and batch jobs
for item in progress(large_dataset, description="Processing data"):
    result = expensive_computation(item)

# In interactive environments: rich progress bar with ETA
# In batch jobs (SLURM, etc.): clean timestamped logging
# [2024-01-15 10:30:15] Processing data: 1500/10000 (15.0%) - Elapsed: 2.3m, Remaining: 13.1m
```

## 🛠️ Utilities & Logging

```python
from easyroutine.logger import setup_logging, logger
from easyroutine import path_to_parents

# Flexible logging setup
setup_logging(
    level="INFO",
    file="experiment.log", 
    console=True,
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger.info("Starting experiment...")

# Convenient navigation helpers
path_to_parents(2)  # Go up 2 directory levels
```

## 🎯 Use Cases

**Mechanistic Interpretability Research**
- Analyze attention patterns in transformer models
- Study information flow through residual streams
- Perform activation patching experiments
- Investigate emergent capabilities in language models

**Model Analysis & Debugging** 
- Extract internal representations for analysis
- Compare activations across different model variants
- Debug model behavior on specific inputs
- Understand failure modes and edge cases

**Educational & Research**
- Teaching transformer internals with hands-on exploration
- Rapid prototyping of interpretability experiments  
- Reproducible research with comprehensive activation logging

## 📖 Documentation

- **API Reference**: Comprehensive docstrings for all functions and classes
- **Examples**: Jupyter notebooks with common use cases
- **Tutorials**: Step-by-step guides for interpretability workflows

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details on:
- Code style and testing requirements
- Documentation standards  
- How to submit issues and pull requests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔬 Citation

If you use EasyRoutine in your research, please cite:

```bibtex
@software{easyroutine2024,
  title={EasyRoutine: A Toolkit for AI Model Interpretability},
  author={Francesco Ortu},
  year={2024},
  url={https://github.com/francescortu/easyroutine}
}
```

---

### Development

For publishing new versions, use semantic version tags in commit messages:
- `[patch]`: Bug fixes (x.x.7 → x.x.8) 
- `[minor]`: New features (x.7.x → x.8.0)
- `[major]`: Breaking changes (2.x.x → 3.0.0)

Example: `git commit -m "Add support for Gemma models [minor]"`