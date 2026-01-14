import time
import unittest
from unittest.mock import patch
import sys
import os
from easyroutine.console.progress import progress, is_non_interactive_batch


class TestIsNonInteractiveBatch(unittest.TestCase):
    """Test cases for is_non_interactive_batch function."""

    def test_jupyter_environment_returns_false(self):
        """Test that Jupyter/IPython environments are detected as interactive."""
        # Mock get_ipython to simulate Jupyter environment
        with patch('builtins.get_ipython', return_value=True):
            result = is_non_interactive_batch()
            self.assertFalse(result, "Jupyter environment should be detected as interactive")

    def test_non_jupyter_with_tty_returns_false(self):
        """Test that regular terminal with TTY is detected as interactive."""
        # Ensure we're not in Jupyter
        with patch('builtins.get_ipython', side_effect=NameError):
            # Mock stdout.isatty to return True
            with patch.object(sys.stdout, 'isatty', return_value=True):
                result = is_non_interactive_batch()
                self.assertFalse(result, "Terminal with TTY should be interactive")

    def test_non_jupyter_without_tty_returns_true(self):
        """Test that non-Jupyter environment without TTY is detected as batch."""
        # Ensure we're not in Jupyter
        with patch('builtins.get_ipython', side_effect=NameError):
            # Mock stdout.isatty to return False
            with patch.object(sys.stdout, 'isatty', return_value=False):
                # Clear any special environment variables
                with patch.dict(os.environ, {'SLURM_PTY_PORT': ''}, clear=False):
                    if 'SLURM_PTY_PORT' in os.environ:
                        del os.environ['SLURM_PTY_PORT']
                    result = is_non_interactive_batch()
                    self.assertTrue(result, "Non-TTY without Jupyter should be batch mode")

    def test_slurm_job_without_tty_returns_true(self):
        """Test that SLURM batch jobs are detected correctly."""
        # Ensure we're not in Jupyter
        with patch('builtins.get_ipython', side_effect=NameError):
            # Mock stdout.isatty to return False
            with patch.object(sys.stdout, 'isatty', return_value=False):
                # Set SLURM_JOB_ID environment variable
                with patch.dict(os.environ, {'SLURM_JOB_ID': '12345'}):
                    result = is_non_interactive_batch()
                    self.assertTrue(result, "SLURM job without TTY should be batch mode")

    def test_jupyter_overrides_non_tty(self):
        """Test that Jupyter detection takes precedence over non-TTY status."""
        # Mock get_ipython to simulate Jupyter environment
        with patch('builtins.get_ipython', return_value=True):
            # Even with non-TTY, should still be detected as interactive
            with patch.object(sys.stdout, 'isatty', return_value=False):
                result = is_non_interactive_batch()
                self.assertFalse(result, "Jupyter should override non-TTY detection")

    def test_dumb_terminal_returns_true(self):
        """Test that TERM=dumb is detected as batch mode."""
        # Ensure we're not in Jupyter
        with patch('builtins.get_ipython', side_effect=NameError):
            # Set TERM to dumb
            with patch.dict(os.environ, {'TERM': 'dumb'}):
                result = is_non_interactive_batch()
                self.assertTrue(result, "TERM=dumb should be batch mode")


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
