"""
Example: Dashboard in None mode

This shows the passthrough mode - no dashboard UI, just standard output.
Useful for simple scripts or when you don't need the dashboard overhead.
"""

import time
from easyroutine.console import Dashboard

# Create dashboard in None mode - fallback to standard output
dash = Dashboard(
    title="Training ResNet-18",
    mode="none",  # No dashboard, just standard rich output
)

# Log experiment configuration (still tracked in state, but not displayed)
dash.log_arguments({"epochs": 3, "batch_size": 32, "optimizer": "Adam", "lr": 0.001})
dash.log_metadata({"model": "ResNet-18", "dataset": "CIFAR-10", "gpus": 1})

print("\n🚀 Starting training with mode='none' (standard output)\n")

# Track training - will use standard rich.progress() instead of dashboard
for epoch in dash.progress(range(3), description="Training"):
    dash.track_variable("epoch", epoch)

    # Data loading
    for batch in dash.progress(range(5), description="Loading Data"):
        time.sleep(0.1)

    # Training batches
    for batch in dash.progress(range(5), description="Training Batches"):
        loss = 2.5 - epoch * 0.5 - batch * 0.1
        dash.track_variable("loss", loss)
        # dash.print() falls back to rich.print()
        dash.print(f"✅ Batch {batch}: loss={loss:.3f}")
        time.sleep(0.1)

dash.print("\n🎉 Training complete!")
dash.print_summary()

print("\nNote: In 'none' mode, dash.print() uses rich.print() and")
print("      dash.progress() uses standard progress bars.")
