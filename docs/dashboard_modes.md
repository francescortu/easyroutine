# Dashboard Modes

The Dashboard supports three different modes for displaying experiment information:

## 1. CLI Mode (default)

Rich-based terminal dashboard with live updates.

```python
from easyroutine.console import Dashboard

dash = Dashboard(title="My Experiment", mode="cli")
```

**Features:**
- Beautiful terminal UI using Rich library
- Real-time updates with Live display
- Nested progress tracking
- Variable monitoring
- Terminal output panel
- Header with arguments and metadata

**Best for:**
- Interactive terminal sessions
- Development and debugging
- Real-time monitoring

**Note:** In CLI mode, use `dash.print()` instead of `print()` to display output in the terminal panel. Regular `print()` won't appear in the dashboard due to Rich Live display conflicts.

**Example:**
```bash
poetry run python examples/dashboard_mode_cli.py
```

## 2. Streamlit Mode

Web-based dashboard using Streamlit.

```python
from easyroutine.console import Dashboard

dash = Dashboard(
    title="My Experiment",
    mode="streamlit",
    streamlit_port=8501,
    streamlit_open_browser=True,
)
```

**Features:**
- Web-based UI accessible in browser
- Real-time updates
- Shareable URL for remote viewing
- **Automatic stdout/stderr capture** (no conflicts!)
- Clean, professional interface

**Installation:**
```bash
poetry add streamlit
# or
poetry install --extras streamlit
```

**Best for:**
- Remote experiments
- Sharing results with team
- Clean, persistent dashboards
- Long-running experiments

**Note:** In Streamlit mode, regular `print()` statements are automatically captured and displayed in the web UI!

**Example:**
```bash
poetry run python examples/dashboard_mode_streamlit.py
```

## 3. None Mode

No dashboard - fallback to standard output.

```python
from easyroutine.console import Dashboard

dash = Dashboard(title="My Experiment", mode="none")
```

**Features:**
- No overhead or UI
- Uses standard rich.print() and progress bars
- All Dashboard methods still work
- State is still tracked (can save/load)

**Best for:**
- Quick scripts
- Minimal overhead needed
- Standard terminal output preferred
- Testing without UI

**Note:** In None mode:
- `dash.print()` → `rich.print()`
- `dash.progress()` → standard rich progress bars
- No live dashboard display
- State is still tracked internally

**Example:**
```bash
poetry run python examples/dashboard_mode_none.py
```

## 4. Auto Mode

Automatically detect the best mode.

```python
from easyroutine.console import Dashboard

dash = Dashboard(title="My Experiment", mode="auto")  # or just omit mode
```

Currently defaults to CLI mode when stdout is a terminal.

## Comparison Table

| Feature | CLI | Streamlit | None |
|---------|-----|-----------|------|
| Rich terminal UI | ✅ | ❌ | ❌ |
| Web UI | ❌ | ✅ | ❌ |
| Live updates | ✅ | ✅ | ❌ |
| Auto stdout capture | ❌* | ✅ | ❌ |
| Nested progress | ✅ | ✅ | ✅ |
| Variable tracking | ✅ | ✅ | ✅ |
| Remote access | ❌ | ✅ | ❌ |
| Zero overhead | ❌ | ❌ | ✅ |
| State persistence | ✅ | ✅ | ✅ |

*Use `dash.print()` instead

## Complete Example

```python
import time
from easyroutine.console import Dashboard

# Create dashboard (choose your mode!)
dash = Dashboard(
    title="Training ResNet-18",
    mode="cli",  # or "streamlit", "none", "auto"
)

# Log configuration
dash.log_arguments({"epochs": 10, "lr": 0.001})
dash.log_metadata({"model": "ResNet-18", "dataset": "CIFAR-10"})

# Track progress
for epoch in dash.progress(range(10), description="Training"):
    for batch in dash.progress(range(100), description="Batches"):
        # Train...
        loss = compute_loss()
        
        # Track metrics
        dash.track_variable("loss", loss)
        
        # Print updates
        dash.print(f"Batch {batch}: loss={loss:.3f}")

# Print summary
dash.print_summary()
```

## Mode Selection Guidelines

**Use CLI mode when:**
- Running experiments locally in terminal
- Need real-time visual feedback
- Debugging and development
- Single-user local experiments

**Use Streamlit mode when:**
- Running experiments on remote servers
- Need to share results with team
- Want persistent, clean dashboards
- Long-running experiments
- Need automatic stdout capture

**Use None mode when:**
- Writing quick scripts
- Don't need dashboard overhead
- Prefer standard terminal output
- Integration with existing tools
- Minimal dependencies

## Configuration Options

All modes support these common options:

```python
dash = Dashboard(
    title="Experiment Name",           # Required
    description="Description",          # Optional
    mode="auto",                        # "auto", "cli", "streamlit", "none"
    save_dir="./results",               # Auto-save state
    auto_save=True,                     # Save on exit
    auto_refresh=True,                  # Auto-refresh display
    refresh_rate=0.5,                   # Refresh interval (seconds)
)
```

Streamlit-specific options:

```python
dash = Dashboard(
    mode="streamlit",
    streamlit_port=8501,                # Web server port
    streamlit_open_browser=True,        # Auto-open browser
)
```
