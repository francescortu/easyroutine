# Dashboard Documentation

## Overview

The Dashboard is a simple, intuitive tool for tracking research experiments with real-time visualization in CLI or web interfaces.

## Quick Start

```python
from easyroutine.console import Dashboard

# Create dashboard
dash = Dashboard(title="My Experiment")

# Log configuration
dash.log_metadata({"model": "gpt2", "lr": 0.001})

# Track progress automatically
for epoch in dash.progress(range(10), description="Training"):
    loss = train()
    dash.track_variable("loss", loss, step=epoch)

# Print summary
dash.print_summary()
```

## Features

### ✨ Automatic Progress Tracking

The simplest way to track progress - just wrap your iterables:

```python
# Automatically tracks and displays progress
for item in dash.progress(data, description="Processing"):
    process(item)
```

### 🔄 Nested Steps

Progress bars automatically nest when used inside each other:

```python
for epoch in dash.progress(range(5), description="Epochs"):
    for batch in dash.progress(dataloader, description="  Batches"):
        train()  # Inner progress shows nested under outer
```

### 📊 Variable Tracking

Track metrics over time:

```python
for epoch in range(10):
    loss = train()
    accuracy = evaluate()
    
    dash.track_variable("loss", loss, step=epoch)
    dash.track_variable("accuracy", accuracy, step=epoch)

# Plot variables (requires matplotlib)
dash.plot_variable("loss")
```

### 🎛️ Manual Step Control

For more control, manually manage steps:

```python
# Method 1: Start/Update/Stop
step_id = dash.start_step("Data Loading", total=100)
for i in range(100):
    load_data(i)
    dash.update_step(step_id)
dash.stop_step(step_id)

# Method 2: Context Manager
with dash.step("Training", total=1000) as step:
    for i in range(1000):
        train()
        step.update()
```

### 📟 Output Capture

Capture print statements to display in dashboard:

```python
with dash.capture_output():
    print("This output will be shown in the dashboard")
    model.summary()
```

### 💾 Persistence

Save and load dashboard state:

```python
# Auto-save on exit
dash = Dashboard(
    title="Experiment",
    save_dir="./runs",
    auto_save=True
)

# Manual save
dash.save("./experiment_state.json")

# Load previous run
dash = Dashboard.load("./experiment_state.json")
```

### 🌐 Web Interface

Launch a web UI (Streamlit - coming soon):

```python
# Start web server
dash.serve(port=8080)
```

## Complete Example

```python
from easyroutine.console import Dashboard
import time
import random

# Use as context manager for automatic cleanup
with Dashboard(
    title="Complete ML Workflow",
    backend="cli",  # or "streamlit" (coming soon)
    save_dir="./dashboard_output",
    auto_save=True
) as dash:
    
    # Log experiment config
    class Config:
        model = "transformer"
        lr = 0.001
        epochs = 5
        batch_size = 64
    
    dash.log_arguments(Config())
    dash.log_metadata({"dataset": "WikiText", "gpu": "A100"})
    
    # Stage 1: Data loading
    for item in dash.progress(range(100), description="Loading Data"):
        time.sleep(0.01)
    
    # Stage 2: Training with nested loops
    for epoch in dash.progress(range(5), description="Training"):
        epoch_losses = []
        
        # Training batches (automatically nested)
        for batch in dash.progress(range(10), description="  Batches"):
            time.sleep(0.05)
            loss = random.uniform(2.0 - epoch * 0.3, 2.5 - epoch * 0.3)
            epoch_losses.append(loss)
        
        # Track metrics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        dash.track_variable("train_loss", avg_loss, step=epoch)
        dash.track_variable("learning_rate", 0.001 * (0.9 ** epoch), step=epoch)
        
        # Capture some output
        with dash.capture_output():
            print(f"Epoch {epoch}: loss = {avg_loss:.4f}")
    
    # Stage 3: Evaluation
    with dash.step("Evaluation", total=50) as step:
        for i in range(50):
            time.sleep(0.02)
            step.update()
    
    # Final metrics
    dash.track_variable("final_accuracy", 0.92)
    
    print("✅ Experiment complete!")
```

## API Reference

### Dashboard Class

```python
Dashboard(
    title: str = "Experiment",
    description: str = "",
    backend: str = "auto",  # "auto", "cli", "streamlit", "none"
    save_dir: Optional[str] = None,
    auto_save: bool = False,
    auto_refresh: bool = True,
    refresh_rate: float = 0.5,
)
```

### Core Methods

- **`log_arguments(args)`** - Log experiment arguments
- **`log_metadata(metadata: dict)`** - Log metadata
- **`progress(iterable, description="")`** - Track progress (automatic)
- **`start_step(description, total=None)`** - Start manual step
- **`update_step(step_id, advance=1)`** - Update step progress
- **`stop_step(step_id)`** - Complete step
- **`step(description, total=None)`** - Context manager for steps
- **`track_variable(name, value, step=None)`** - Track variable
- **`plot_variable(name)`** - Plot variable (requires matplotlib)
- **`capture_output()`** - Context manager for output capture
- **`save(filepath=None)`** - Save state to JSON
- **`serve(port=8080)`** - Launch web UI
- **`print_summary()`** - Print final summary

## CLI Display

The CLI backend uses Rich library for beautiful terminal output with:

- 📊 Real-time progress bars with nesting
- 📈 Variable tracking table
- 📟 Output capture display
- ⏱️ Time tracking and estimates
- ✓ Status indicators (running, completed, failed)

## Architecture

```
Dashboard/
├── __init__.py         # Main Dashboard class
├── state.py            # Data models & state management
├── cli_backend.py      # Rich-based terminal display
├── streamlit_backend.py  # Web UI (coming soon)
└── utils.py            # Utilities (output capture, etc.)
```

## Design Philosophy

1. **Simple by default** - `dash.progress()` should "just work"
2. **Nested support** - Automatically handle nested loops
3. **Implicit completion** - Steps auto-complete when done
4. **Real-time updates** - See progress as it happens
5. **Minimal boilerplate** - Maximum functionality, minimum code

## Tips & Best Practices

1. **Use automatic progress** for most cases - it's simpler
2. **Manual steps** are useful when you need precise control
3. **Track key metrics** that matter for your research
4. **Use output capture** to keep important logs in the dashboard
5. **Enable auto-save** for long-running experiments
6. **Context manager** ensures proper cleanup

## Requirements

- Python 3.8+
- rich (for CLI display)
- streamlit (optional, for web UI)
- matplotlib (optional, for plotting)

## Examples

See the `examples/` directory:

- `dashboard_quickstart.py` - Minimal quick start
- `dashboard_example.py` - Comprehensive examples showing all features
