"""
Example: Nested Progress Bars

This example demonstrates the nested progress bar functionality added
to easyroutine.console.progress. You can now nest progress bars without
errors, and they will be automatically indented to show the hierarchy.
"""
from easyroutine.console.progress import progress
import time


def example_nested_loops():
    """
    Basic example of nested progress bars.
    Works in both interactive and batch modes.
    """
    print("\n=== Example: Nested Loops ===\n")
    
    for i in progress(range(3), description="Outer loop"):
        time.sleep(0.1)
        
        for j in progress(range(3), description=f"  Inner loop {i}"):
            time.sleep(0.1)


def example_data_processing():
    """
    Practical example: processing datasets with batches.
    """
    print("\n=== Example: Data Processing ===\n")
    
    datasets = ["Dataset A", "Dataset B"]
    
    for dataset in progress(datasets, description="Processing Datasets"):
        # Simulate loading data
        time.sleep(0.2)
        
        # Process batches
        for batch in progress(range(5), description=f"  Processing batches"):
            time.sleep(0.1)


def example_triple_nested():
    """
    Advanced example: triple-nested loops.
    """
    print("\n=== Example: Triple-Nested Loops ===\n")
    
    for dataset in progress(range(2), description="Datasets", force_batch_mode=True):
        for epoch in progress(range(2), description="  Epochs", force_batch_mode=True):
            for batch in progress(range(3), description="    Batches", force_batch_mode=True):
                time.sleep(0.05)


if __name__ == "__main__":
    print("=" * 70)
    print("NESTED PROGRESS BARS EXAMPLES")
    print("=" * 70)
    
    example_nested_loops()
    example_data_processing()
    example_triple_nested()
    
    print("\n" + "=" * 70)
    print("Examples completed successfully!")
    print("=" * 70)
