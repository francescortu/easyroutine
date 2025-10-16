"""
Example: Dashboard in CLI mode (default)

This shows the Rich-based terminal dashboard with live updates.
"""

import time
from easyroutine.console import Dashboard

# Create dashboard in CLI mode (default)
dash = Dashboard(
    title="Training ResNet-18",
    mode="cli",  # Rich-based terminal dashboard
)

# Log experiment configuration
dash.log_arguments({"epochs": 3, "batch_size": 32, "optimizer": "Adam", "lr": 0.001})
dash.log_metadata({"model": "ResNet-18", "dataset": "CIFAR-10", "gpus": 1})

# Track training with nested progress
for epoch in dash.progress(range(3), description="Training"):
    dash.track_variable("epoch", epoch)

    # Data loading
    for batch in dash.progress(range(5), description="Loading Data"):
        time.sleep(0.1)

    # Training batches
    for batch in dash.progress(range(5), description="Training Batches"):
        loss = 2.5 - epoch * 0.5 - batch * 0.1
        dash.track_variable("loss", loss)
        dash.print(f"Batch {batch}: loss={loss:.3f}")
        time.sleep(0.1)

dash.print("Training complete!")
dash.print_summary()

# Keep dashboard open
input("\nPress Enter to close...")
