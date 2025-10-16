"""Quick start example for the research dashboard.

The Dashboard supports multiple modes:
- mode="cli" (default): Rich terminal dashboard with live updates
- mode="streamlit": Web-based dashboard (requires: poetry add streamlit)
- mode="none": No dashboard, fallback to standard output
- mode="auto": Auto-detect best mode

See dashboard_mode_switch.py for examples of all modes.
"""

import time
from easyroutine.console import Dashboard

# Create dashboard with title (uses CLI mode by default)
dash = Dashboard(title="Training ResNet-18")
# Try: dash = Dashboard(title="Training ResNet-18", mode="none") for minimal overhead

# Log experiment configuration
dash.log_arguments({"epochs": 100, "batch_size": 32, "optimizer": "Adam", "lr": 0.001})
dash.log_metadata({"model": "ResNet-18", "dataset": "ImageNet", "gpus": 4})

# Track training progress with nested steps
for epoch in dash.progress(range(3), description="Training"):
    dash.track_variable("epoch", epoch)

    # Nested step: data loading
    for batch in dash.progress(range(5), description="Loading Data"):
        time.sleep(0.5)

    # Nested step: forward/backward pass
    for batch in dash.progress(range(5), description="Training Batches"):
        loss = 2.5 - epoch * 0.5 - batch * 0.1
        dash.track_variable("loss", loss)
        dash.print(f"Batch {batch}: loss={loss:.3f}")
        time.sleep(0.5)

dash.print("Training complete!")

# Keep dashboard open for viewing
input("\nPress Enter to close dashboard...")
