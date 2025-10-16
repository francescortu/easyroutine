# Dashboard

Research Dashboard for tracking experiments in real-time with multiple display modes.

::: easyroutine.console.Dashboard

## Overview

The Dashboard provides a unified interface for tracking experiments with three display modes:

- **CLI Mode**: Rich terminal dashboard with live updates
- **Streamlit Mode**: Web-based dashboard accessible in browser
- **None Mode**: Minimal output with standard print/progress

## Basic Usage

### Quick Start

```python
from easyroutine.console import Dashboard

# Create dashboard (defaults to CLI mode)
dash = Dashboard(title="My Experiment")

# Log configuration
dash.log_arguments({"epochs": 100, "lr": 0.001})
dash.log_metadata({"model": "ResNet-18"})

# Track progress automatically
for epoch in dash.progress(range(100), description="Training"):
    # Your training code
    loss = train_epoch()
    dash.track_variable("loss", loss)
    dash.print(f"Epoch {epoch}: loss={loss:.4f}")
```

### With Different Modes

```python
# CLI Mode (default) - Rich terminal UI
dash = Dashboard(title="Experiment", mode="cli")
dash.print("Use dash.print() for output")

# Streamlit Mode - Web UI with automatic stdout capture
dash = Dashboard(title="Experiment", mode="streamlit")
print("Regular print works!")  # Automatically captured

# None Mode - Standard output, minimal overhead
dash = Dashboard(title="Experiment", mode="none")
print("Uses rich.print()")
```

## Initialization

### Parameters

- **title** (str): Experiment title displayed in dashboard header
- **description** (str, optional): Detailed experiment description
- **mode** (str, default="auto"): Display mode - "cli", "streamlit", "none", or "auto"
- **save_dir** (str, optional): Directory to save dashboard state (None = no save)
- **auto_save** (bool, default=False): Whether to auto-save state on exit
- **auto_refresh** (bool, default=True): Whether to auto-refresh display
- **refresh_rate** (float, default=0.5): Display refresh rate in seconds
- **capture_output** (bool, default=True): Whether to capture stdout/stderr
- **streamlit_port** (int, default=8501): Port for Streamlit server (mode="streamlit" only)
- **streamlit_open_browser** (bool, default=True): Auto-open browser for Streamlit

### Examples

```python
# Basic CLI dashboard
dash = Dashboard(title="Training ResNet-18")

# Web dashboard on custom port
dash = Dashboard(
    title="Experiment",
    mode="streamlit",
    streamlit_port=8080,
    streamlit_open_browser=False,
)

# Auto-save to directory
dash = Dashboard(
    title="Training",
    save_dir="./results",
    auto_save=True,
)

# Custom refresh rate
dash = Dashboard(
    title="Fast Updates",
    refresh_rate=0.1,  # Update every 100ms
)
```

## Logging Configuration

### log_arguments()

Log experiment arguments/parameters.

```python
dash.log_arguments(args)
```

**Parameters:**
- `args`: Argparse namespace, dict, or any object with `__dict__`

**Examples:**

```python
# From argparse
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
dash.log_arguments(args)

# From dict
dash.log_arguments({
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "Adam",
})

# From custom object
class Config:
    epochs = 100
    lr = 0.001

config = Config()
dash.log_arguments(config)
```

### log_metadata()

Log or update experiment metadata.

```python
dash.log_metadata(metadata)
```

**Parameters:**
- `metadata`: Dictionary of metadata to log

**Examples:**

```python
# Log model information
dash.log_metadata({
    "model": "ResNet-18",
    "dataset": "CIFAR-10",
    "num_parameters": "11M",
    "device": "cuda:0",
})

# Update metadata during training
dash.log_metadata({"best_accuracy": 0.95})
```

## Progress Tracking

### progress()

Track progress through an iterable (like tqdm). Automatically creates nested steps.

```python
for item in dash.progress(iterable, description="Processing"):
    process(item)
```

**Parameters:**
- `iterable`: The iterable to track
- `description` (str): Description of this step
- `total` (int, optional): Total items (auto-detected if possible)

**Examples:**

```python
# Simple progress
for epoch in dash.progress(range(100), description="Training"):
    train()

# Nested progress (automatic hierarchy)
for epoch in dash.progress(range(10), description="Epochs"):
    for batch in dash.progress(data_loader, description="Batches"):
        train_batch(batch)

# With explicit total
for item in dash.progress(generator(), description="Processing", total=1000):
    process(item)

# Multiple nested levels
for epoch in dash.progress(range(5), description="Training"):
    for phase in dash.progress(["train", "val"], description="Phase"):
        for batch in dash.progress(range(100), description="Batches"):
            process(batch)
```

### Manual Step Control

For more control over steps:

```python
# Start a step
step_id = dash.start_step("Training", total=100)

# Update progress
for i in range(100):
    train()
    dash.update_step(step_id, advance=1)

# Complete step
dash.stop_step(step_id)
```

**Or use context manager:**

```python
with dash.step("Training", total=100) as step:
    for i in range(100):
        train()
        step.update()
```

## Variable Tracking

### track_variable()

Track a variable over time.

```python
dash.track_variable(name, value, step=None)
```

**Parameters:**
- `name` (str): Variable name
- `value` (float | int | str): Variable value
- `step` (int, optional): Step/epoch number for this value

**Examples:**

```python
# Track loss during training
for epoch in range(100):
    loss = train_epoch()
    dash.track_variable("loss", loss, step=epoch)
    dash.track_variable("learning_rate", get_lr(), step=epoch)

# Track multiple metrics
dash.track_variable("train_loss", train_loss)
dash.track_variable("val_loss", val_loss)
dash.track_variable("accuracy", accuracy)
dash.track_variable("f1_score", f1)

# Track without explicit step
for batch in range(1000):
    loss = train_batch()
    dash.track_variable("batch_loss", loss)
```

## Output

### print()

Print to dashboard output (CLI mode) or captured output (Streamlit mode).

```python
dash.print(*args, **kwargs)
```

**Mode Behavior:**
- **CLI mode**: Prints to terminal output panel in dashboard
- **Streamlit mode**: Regular `print()` is captured automatically
- **None mode**: Falls back to `rich.print()`

**Examples:**

```python
# CLI mode - use dash.print()
dash = Dashboard(mode="cli")
dash.print("Training started")
dash.print(f"Epoch {epoch}: loss={loss:.4f}")

# Streamlit mode - regular print works
dash = Dashboard(mode="streamlit")
print("This appears in web UI!")
print(f"Batch {batch} complete")

# None mode - rich.print()
dash = Dashboard(mode="none")
dash.print("[green]Success![/green]")
```

## Persistence

### save()

Save dashboard state to JSON file.

```python
dash.save(filepath="dashboard_state.json")
```

**Example:**

```python
dash = Dashboard(title="Training")
# ... run experiment ...
dash.save("results/experiment_001.json")
```

### load()

Load dashboard state from JSON file.

```python
dash = Dashboard.load("dashboard_state.json")
```

**Example:**

```python
# Load previous experiment
dash = Dashboard.load("results/experiment_001.json")

# Continue from checkpoint
dash = Dashboard.load("checkpoint.json")
for epoch in dash.progress(range(10, 100), description="Resume Training"):
    train()
```

## Summary

### print_summary()

Print final experiment summary.

```python
dash.print_summary()
```

**Example:**

```python
dash = Dashboard(title="Training")
# ... run experiment ...
dash.print_summary()
# Output:
# ============================================================
# Experiment: Training
# Steps completed: 10/10
# Variables tracked: 3
# ============================================================
```

## Complete Examples

### Example 1: Simple Training Loop

```python
from easyroutine.console import Dashboard
import time

# Create dashboard
dash = Dashboard(title="ResNet-18 Training")

# Log configuration
dash.log_arguments({
    "epochs": 10,
    "batch_size": 32,
    "lr": 0.001,
})

dash.log_metadata({
    "model": "ResNet-18",
    "dataset": "CIFAR-10",
})

# Training loop
for epoch in dash.progress(range(10), description="Training"):
    # Data loading
    for batch in dash.progress(range(100), description="Loading"):
        time.sleep(0.01)
    
    # Training
    epoch_loss = 0
    for batch in dash.progress(range(100), description="Training"):
        loss = train_batch()
        epoch_loss += loss
        time.sleep(0.01)
    
    # Track metrics
    avg_loss = epoch_loss / 100
    dash.track_variable("loss", avg_loss, step=epoch)
    dash.print(f"Epoch {epoch}: loss={avg_loss:.4f}")

dash.print_summary()
```

### Example 2: Multi-phase Training

```python
from easyroutine.console import Dashboard

dash = Dashboard(title="Multi-phase Training", mode="cli")

# Configuration
dash.log_arguments({"total_epochs": 150})
dash.log_metadata({
    "model": "ViT-B/16",
    "phases": ["warmup", "training", "finetuning"],
})

# Phase 1: Warmup
for epoch in dash.progress(range(10), description="Warmup"):
    train_with_small_lr()
    dash.track_variable("lr", 1e-5, step=epoch)

# Phase 2: Training
for epoch in dash.progress(range(100), description="Training"):
    train_with_normal_lr()
    dash.track_variable("lr", 1e-3, step=epoch + 10)

# Phase 3: Finetuning
for epoch in dash.progress(range(40), description="Finetuning"):
    train_with_decay_lr()
    dash.track_variable("lr", 1e-4, step=epoch + 110)

dash.save("results/multiphase_training.json")
```

### Example 3: Streamlit Remote Monitoring

```python
from easyroutine.console import Dashboard

# Web dashboard for remote monitoring
dash = Dashboard(
    title="Distributed Training",
    mode="streamlit",
    streamlit_port=8501,
    auto_save=True,
    save_dir="./results",
)

dash.log_arguments({
    "nodes": 4,
    "gpus_per_node": 8,
    "total_gpus": 32,
})

dash.log_metadata({
    "model": "GPT-3",
    "dataset": "OpenWebText",
    "tokens": "300B",
})

print("Starting distributed training...")
print("Monitor at: http://your-server:8501")

for epoch in dash.progress(range(1000), description="Training"):
    # Regular print statements work!
    print(f"Epoch {epoch} starting...")
    
    loss = distributed_train()
    dash.track_variable("loss", loss)
    dash.track_variable("throughput", get_throughput())
    
    print(f"Epoch {epoch} complete: loss={loss:.4f}")

print("Training complete!")
dash.print_summary()
```

### Example 4: Hyperparameter Search

```python
from easyroutine.console import Dashboard
import itertools

# Grid search with dashboard
learning_rates = [1e-4, 1e-3, 1e-2]
batch_sizes = [16, 32, 64]

best_acc = 0
best_params = {}

# Use none mode for clean output during search
dash = Dashboard(title="Hyperparameter Search", mode="none")

configs = list(itertools.product(learning_rates, batch_sizes))
for lr, bs in dash.progress(configs, description="Configurations"):
    print(f"\nTesting lr={lr}, batch_size={bs}")
    
    # Train with this config
    acc = train(lr=lr, batch_size=bs)
    
    dash.track_variable(f"acc_lr{lr}_bs{bs}", acc)
    
    if acc > best_acc:
        best_acc = acc
        best_params = {"lr": lr, "batch_size": bs}
        print(f"  New best: {acc:.4f}")

dash.log_metadata({
    "best_accuracy": best_acc,
    "best_params": best_params,
})

dash.print_summary()
dash.save("results/hyperparameter_search.json")
```

### Example 5: Context Manager for Steps

```python
from easyroutine.console import Dashboard

dash = Dashboard(title="Pipeline Execution")

# Use context manager for automatic step completion
with dash.step("Data Preprocessing", total=1000) as step:
    for item in data:
        preprocess(item)
        step.update()

with dash.step("Model Training", total=100) as step:
    for epoch in range(100):
        train()
        step.update()
        dash.track_variable("loss", get_loss())

with dash.step("Evaluation", total=500) as step:
    for sample in test_data:
        evaluate(sample)
        step.update()

dash.print_summary()
```

## Mode Comparison

| Feature | CLI | Streamlit | None |
|---------|-----|-----------|------|
| Rich terminal UI | ✅ | ❌ | ❌ |
| Web UI | ❌ | ✅ | ❌ |
| Live updates | ✅ | ✅ | ❌ |
| Auto stdout capture | ❌* | ✅ | ❌ |
| Remote access | ❌ | ✅ | ❌ |
| Zero overhead | ❌ | ❌ | ✅ |
| State persistence | ✅ | ✅ | ✅ |
| Best for | Local dev | Remote jobs | Quick scripts |

*Use `dash.print()` in CLI mode

## See Also

- [Dashboard Modes Documentation](../../dashboard_modes.md)
- [Streamlit Setup Guide](../../streamlit_mode.md)
- [Implementation Summary](../../dashboard_implementation_summary.md)
