# Interpretability Tools

This section covers the tools provided by `easyroutine` for model interpretability beyond basic activation extraction.

## LogitLens

`LogitLens` is a powerful tool that enables you to interpret intermediate representations in transformer models by projecting them through the model's output head to the vocabulary space. This technique, often called "logit lens" in the mechanistic interpretability literature, allows researchers to understand what token predictions would be made at each layer, providing insights into how representations evolve through the network.

### Key Features

- Project any intermediate activation to token probability space
- Apply proper layer normalization based on model architecture (supports both LayerNorm and RMSNorm)
- Compute metrics like logit differences between specific token directions
- Support for various model architectures with different normalization schemes

### Usage

```python
from easyroutine.interpretability import HookedModel
from easyroutine.interpretability.tools import LogitLens

# Create a model
model = HookedModel.from_pretrained("mistralai/Mistral-7B-v0.1")

# Create LogitLens from the model
logit_lens = LogitLens.from_model(model)

# Run forward pass and collect activations
cache = model.forward(
    inputs, 
    target_token_positions=["last"],
    extraction_config=ExtractionConfig(
        extract_resid_out=True,
        extract_last_layernorm=True  # Required for proper normalization
    )
)

# Analyze residual streams from different layers using LogitLens
layer = 12  # Specify which layer to analyze
results = logit_lens.compute(
    activations=cache, 
    target_key=f"resid_out_{layer}", 
    apply_norm=True,
    apply_softmax=True  # Set to True if you want probabilities instead of logits
)

# To compute logit difference between two specific tokens
token_diff_results = logit_lens.compute(
    activations=cache,
    target_key=f"resid_out_{layer}",
    token_directions=[(model.hf_tokenizer.encode(" yes")[0], model.hf_tokenizer.encode(" no")[0])],
    metric="logit_diff"
)
```

### API Reference

#### LogitLens Constructor

```python
def __init__(self, unembedding_matrix, last_layer_norm, model_name, model_config):
```

**Arguments:**
- `unembedding_matrix`: The weight matrix used in the model's output embedding layer
- `last_layer_norm`: The model's final layer normalization module
- `model_name`: Name of the model
- `model_config`: Model configuration object containing architecture details

#### Class Methods

```python
@classmethod
def from_model(cls, model: HookedModel) -> 'LogitLens'
```
Create a LogitLens instance from a HookedModel.

```python
@classmethod
def from_model_name(cls, model_name: str) -> 'LogitLens'
```
Create a LogitLens instance directly from a model name (will load the model).

#### Core Methods

```python
def compute(
    self,
    activations: ActivationCache,
    target_key: str,
    token_directions: Optional[Union[List[int], List[Tuple[int, int]]]] = None,
    apply_norm: bool = True,
    apply_softmax: bool = False,
    metric: Optional[Literal["logit_diff", "accuracy"]] = "logit_diff"
) -> dict:
```

**Arguments:**
- `activations`: ActivationCache containing model activations
- `target_key`: The key/pattern for activations to analyze (e.g., "resid_out_{i}")
- `token_directions`: Optional list of token IDs or pairs to compute specific direction metrics
- `apply_norm`: Whether to apply layer normalization to the activations
- `apply_softmax`: Whether to apply softmax to get probabilities instead of raw logits
- `metric`: Metric to compute if token_directions are provided ("logit_diff" or "accuracy")

**Returns:**
- Dictionary with computed logit lens results

### Example: Analyzing Mid-layer Predictions

```python
import torch
from easyroutine.interpretability import HookedModel, ExtractionConfig
from easyroutine.interpretability.tools import LogitLens

model = HookedModel.from_pretrained("gpt2")
logit_lens = LogitLens.from_model(model)

# Create a prompt
tokenizer = model.get_tokenizer()
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Extract activations from all layers
cache = model.forward(
    inputs, 
    target_token_positions=["last"],
    extraction_config=ExtractionConfig(
        extract_resid_out=True,
        extract_last_layernorm=True
    )
)

# Analyze how the prediction for the last token evolves through layers
results = {}
for layer in range(model.model_config.num_hidden_layers):
    layer_results = logit_lens.compute(
        activations=cache, 
        target_key=f"resid_out_{layer}", 
        apply_norm=True,
        apply_softmax=True
    )
    
    # Get top 5 tokens for the last position
    logits = layer_results[f"logit_lens_resid_out_{layer}"][0, -1]
    top_tokens = torch.topk(logits, 5)
    
    results[layer] = [
        (tokenizer.decode(idx.item()), prob.item()) 
        for idx, prob in zip(top_tokens.indices, top_tokens.values)
    ]

# Print results to see how predictions evolve
for layer, tokens in results.items():
    print(f"Layer {layer}: {tokens}")
```

This example shows how to track the evolution of predictions through the layers of the model, providing insights into how the model builds up its final prediction.

## ActivationSaver

`ActivationSaver` is a utility that provides a structured way to save and load model activations with rich metadata. This is particularly useful for large-scale experiments where you want to save activations for later analysis.

### Key Features

- Structured filesystem organization for saved activations
- Automatic metadata tracking (model, configuration, extraction details)
- Tagging system for easier identification of specific runs
- Query functionality through companion `ActivationLoader` class
- Supports renaming experiments and organizing workspace

### Usage

```python
from easyroutine.interpretability import HookedModel, ExtractionConfig
from easyroutine.interpretability.activation_saver import ActivationSaver

# Create a model and extract activations
model = HookedModel.from_pretrained("gpt2")
inputs = tokenizer("Hello, world!", return_tensors="pt")

cache = model.forward(
    inputs, 
    target_token_positions=["last"],
    extraction_config=ExtractionConfig(extract_resid_out=True)
)

# Save activations with metadata
saver = ActivationSaver(base_dir="./saved_activations", experiment_name="hello_world_test")
save_dir = saver.save(
    activations=cache,
    model=model,
    target_token_positions=["last"],
    interventions=None,
    extraction_config=ExtractionConfig(extract_resid_out=True),
    tag="first_test"
)

# Or use the simplified method if the cache already contains metadata
# This is the case with caches created using model.extract_cache()
dataset_cache = model.extract_cache(
    dataloader,
    target_token_positions=["last"],
    extraction_config=ExtractionConfig(extract_resid_out=True)
)
saver.save_cache(dataset_cache, tag="dataset_run")
```

### API Reference

#### ActivationSaver

```python
def __init__(self, base_dir: Union[Path, str], experiment_name: str = "default")
```

**Arguments:**
- `base_dir`: Base directory where activations will be saved
- `experiment_name`: Name of the experiment (creates a subdirectory)

```python
@classmethod
def from_env(cls, experiment_name: str = "default")
```
Create a saver using the base directory specified in the environment variable `ACTIVATION_BASE_DIR`

```python
def save(
    self,
    activations: Union[torch.Tensor, dict, ActivationCache],
    model: HookedModel,
    target_token_positions,
    interventions: Optional[List[Intervention]],
    extraction_config: ExtractionConfig,
    other_metadata: dict = {},
    tag: Optional[str] = None,
)
```

**Arguments:**
- `activations`: The activation cache or tensor to save
- `model`: The HookedModel used to generate the activations
- `target_token_positions`: The token positions used during extraction
- `interventions`: Any interventions applied during extraction
- `extraction_config`: The extraction configuration used
- `other_metadata`: Additional metadata to store
- `tag`: Optional tag for easier identification

**Returns:**
- The directory path where activations were saved

```python
def save_cache(
    self,
    cache: ActivationCache,
    other_metadata: dict = {},
    tag: Optional[str] = None,
)
```

A simplified save method that expects the cache to already contain metadata.

```python
def rename_experiment(self, new_experiment_name: str)
```
Renames the experiment directory and updates all metadata files within it.

#### ActivationLoader

```python
def __init__(self, base_dir: Path, experiment_name: str = "default")
```

**Arguments:**
- `base_dir`: Base directory where activations are saved
- `experiment_name`: Name of the experiment to load from

```python
@classmethod
def from_env(cls, experiment_name: str = "default")
```
Create a loader using the base directory specified in the environment variable `ACTIVATION_BASE_DIR`

```python
@classmethod
def from_saver(cls, saver: ActivationSaver)
```
Create a loader from an existing ActivationSaver

```python
def query(
    self,
    experiment_name: Optional[str] = None,
    model_name: Optional[str] = None,
    target_token_positions: Optional[List[Union[str, int]]] = None,
    pivot_positions: Optional[List[int]] = None,
    save_time: Optional[str] = None,
    custom_keys: Optional[dict] = None,
    extraction_config: Optional[ExtractionConfig] = None,
    interventions: Optional[List[Intervention]] = None,
    tag: Optional[str] = None,
) -> QueryResult
```

Search for saved activations based on various criteria.

**Arguments:**
- Various search criteria to filter saved activations

**Returns:**
- A QueryResult object containing matching runs

### Query Results

The `QueryResult` class provides a convenient interface for managing the results of queries:

```python
# List all matching runs
print(query_result)

# Get detailed paths
print(query_result.get_paths())

# Load a specific run by index (e.g., most recent with -1)
activations, metadata = query_result.load(-1)

# Load by specific time folder
activations, metadata = query_result.load("2023-04-15_14-30-25")

# Remove a specific run
query_result.remove(-1)  # Remove the most recent run

# Update experiment name for a run
query_result.update_run_experiment(-1, "new_experiment_name")
```

### Example: Comprehensive Workflow

```python
import torch
from easyroutine.interpretability import HookedModel, ExtractionConfig
from easyroutine.interpretability.activation_saver import ActivationSaver, ActivationLoader

# 1. Create model and dataset
model = HookedModel.from_pretrained("gpt2")
tokenizer = model.get_tokenizer()

# Create a simple dataset
texts = ["Example text 1", "Example text 2", "Example text 3"]
dataset = [tokenizer(text, return_tensors="pt") for text in texts]

# 2. Extract and save activations
saver = ActivationSaver(base_dir="./saved_activations", experiment_name="documentation_example")

for i, inputs in enumerate(dataset):
    # Extract activations for this batch
    cache = model.forward(
        inputs, 
        target_token_positions=["last"],
        extraction_config=ExtractionConfig(
            extract_resid_out=True,
            extract_attn_pattern=True
        )
    )
    
    # Save with batch-specific tag
    saver.save(
        activations=cache,
        model=model,
        target_token_positions=["last"],
        interventions=None,
        extraction_config=ExtractionConfig(
            extract_resid_out=True,
            extract_attn_pattern=True
        ),
        other_metadata={"batch_id": i},
        tag=f"batch_{i}"
    )

# 3. Load and analyze results later
loader = ActivationLoader.from_saver(saver)

# Query for specific batch
batch_0_results = loader.query(tag="batch_0")
activations, metadata = batch_0_results.load(-1)  # Load the most recent match

# Query by custom metadata
specific_batch = loader.query(custom_keys={"batch_id": 1})
activations, metadata = specific_batch.load(-1)

# Query by model name and extraction configuration
all_gpt2_runs = loader.query(
    model_name="gpt2",
    extraction_config=ExtractionConfig(extract_attn_pattern=True)
)

print(f"Found {len(all_gpt2_runs.results)} matching runs")
```

This example demonstrates a complete workflow of extracting activations from multiple inputs, saving them with metadata, and later querying and loading them for analysis.