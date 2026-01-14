"""Tests for is_non_interactive_batch function to ensure proper Jupyter detection."""
import unittest
from unittest.mock import patch
import sys
import os
from easyroutine.console.progress import is_non_interactive_batch


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


if __name__ == '__main__':
    unittest.main()
