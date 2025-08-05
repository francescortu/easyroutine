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


if __name__ == "__main__":
    print("=" * 50)
    print(" VISUAL TEST FOR PROGRESS BARS")
    print("=" * 50)
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

    print("=" * 50)
    print(" VISUAL TEST COMPLETE")
    print("=" * 50)
