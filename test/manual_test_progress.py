"""
Test progress bars functionality including basic and nested progress bars.

This test file demonstrates the progress bar features in both interactive (Rich)
mode and batch mode, including support for nested progress bars with proper
indentation to visualize the nesting structure.
"""

import time
from easyroutine.console.progress import progress


def run_test(description, **kwargs):
    """Helper function to run a progress bar test."""
    print("-" * 50)
    print(f"Testing: {description}")
    print(f"Arguments: {kwargs}")
    print("-" * 50)

    items = range(100)
    for _ in progress(items, description=description, **kwargs):
        time.sleep(0.02)
    print("\n")


def test_nested_progress_interactive():
    """Test nested progress bars in interactive (Rich) mode."""
    print("-" * 50)
    print("Testing: Nested progress bars in interactive mode")
    print("-" * 50)

    for i in progress(range(5), description="Outer loop"):
        time.sleep(0.1)
        for j in progress(range(20), description=f"  Inner loop {i}"):
            time.sleep(0.1)

    print("✓ Nested progress bars work in interactive mode!\n")


def test_nested_progress_batch():
    """Test nested progress bars in batch mode."""
    print("-" * 50)
    print("Testing: Nested progress bars in batch mode")
    print("-" * 50)

    for i in progress(range(5), description="Outer loop", force_batch_mode=True):
        time.sleep(0.1)
        for j in progress(
            range(20), description=f"Inner loop {i}", force_batch_mode=True
        ):
            time.sleep(0.1)

    print("✓ Nested progress bars work in batch mode!\n")


def test_triple_nested():
    """Test triple-nested progress bars to verify indentation."""
    print("-" * 50)
    print("Testing: Triple-nested progress bars")
    print("-" * 50)

    for i in progress(range(3), description="Level 1", force_batch_mode=True):
        time.sleep(0.1)
        for j in progress(range(4), description=f"Level 2.{i}", force_batch_mode=True):
            time.sleep(0.1)
            for k in progress(
                range(10), description=f"Level 3.{i}.{j}", force_batch_mode=True
            ):
                time.sleep(0.1)

    print("✓ Triple-nested progress bars work correctly!\n")


if __name__ == "__main__":
    print("=" * 60)
    print(" VISUAL TEST FOR PROGRESS BARS")
    print("=" * 60)
    print("This script demonstrates the different progress bar modes.\n")

    # --- Interactive Mode (Default) ---
    # This should show a rich, animated progress bar if run in a standard terminal.
    run_test("Default Interactive Progress")

    # --- Forced Batch Mode ---
    # This simulates running in a non-interactive environment like a SLURM job.
    # It should output simple text-based log lines.
    run_test("Forced Batch Mode", force_batch_mode=True)

    # --- Forced Batch Mode with Item-based Updates ---
    # Updates every 20 items instead of based on time.
    run_test(
        "Forced Batch Mode (Update every 20 items)",
        force_batch_mode=True,
        log_interval=10,  # High log interval to ensure item-based updates trigger first
        update_frequency=20,
    )

    # --- Disabled Progress Bar ---
    # This should run without any output at all.
    run_test("Disabled Progress Bar", disable=True)

    # --- Test with no total ---
    # This should still work, but might not show percentage or time remaining.
    print("-" * 50)
    print("Testing: Progress with an iterator (no len())")
    print("-" * 50)

    def generator():
        for i in range(50):
            yield i

    for _ in progress(generator(), description="Iterator Progress (no total)"):
        time.sleep(0.02)
    print("\n")

    # --- Test with no total in batch mode ---
    print("-" * 50)
    print("Testing: Progress with an iterator in Batch Mode")
    print("-" * 50)
    for _ in progress(
        generator(), description="Iterator Progress (Batch Mode)", force_batch_mode=True
    ):
        time.sleep(0.02)
    print("\n")

    print("=" * 60)
    print(" NESTED PROGRESS BAR TESTS")
    print("=" * 60)

    # --- Nested Progress Tests ---
    test_nested_progress_interactive()
    test_nested_progress_batch()
    test_triple_nested()

    print("=" * 60)
    print(" ALL TESTS COMPLETE")
    print("=" * 60)
