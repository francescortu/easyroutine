import unittest
import os
import sys
import time
from io import StringIO
from unittest.mock import patch, MagicMock
from easyroutine.console.progress import (
    LoggingProgress,
    format_time,
    is_non_interactive_batch,
    get_progress_bar,
    progress,
    _NoOpProgress
)


class TestProgressModule(unittest.TestCase):
    """Test suite for easyroutine.console.progress module."""

    def test_format_time_seconds(self):
        """Test format_time with seconds."""
        self.assertEqual(format_time(30.5), "30.5s")
        self.assertEqual(format_time(45.0), "45.0s")
        self.assertEqual(format_time(59.9), "59.9s")

    def test_format_time_minutes(self):
        """Test format_time with minutes."""
        self.assertEqual(format_time(60), "1.0m")
        self.assertEqual(format_time(90), "1.5m")
        self.assertEqual(format_time(3540), "59.0m")  # Just under an hour

    def test_format_time_hours(self):
        """Test format_time with hours."""
        self.assertEqual(format_time(3600), "1.0h")
        self.assertEqual(format_time(5400), "1.5h")
        self.assertEqual(format_time(7200), "2.0h")

    def test_format_time_edge_cases(self):
        """Test format_time with edge cases."""
        self.assertEqual(format_time(0), "0.0s")
        self.assertEqual(format_time(0.1), "0.1s")


class TestLoggingProgress(unittest.TestCase):
    """Test suite for LoggingProgress class."""

    def setUp(self):
        """Set up test environment."""
        self.progress = LoggingProgress(log_interval=0.1, update_frequency=2)

    def test_init(self):
        """Test LoggingProgress initialization."""
        progress = LoggingProgress(log_interval=5, update_frequency=10)
        self.assertEqual(progress.log_interval, 5)
        self.assertEqual(progress.update_frequency, 10)
        self.assertEqual(progress.tasks, {})

    def test_context_manager(self):
        """Test LoggingProgress as context manager."""
        with LoggingProgress() as progress:
            self.assertIsInstance(progress, LoggingProgress)

    @patch('builtins.print')
    def test_add_task(self, mock_print):
        """Test adding a task."""
        task_id = self.progress.add_task("Test task", total=100)
        
        self.assertEqual(task_id, 0)
        self.assertIn(task_id, self.progress.tasks)
        
        task = self.progress.tasks[task_id]
        self.assertEqual(task["description"], "Test task")
        self.assertEqual(task["total"], 100)
        self.assertEqual(task["completed"], 0)
        
        # Check print was called
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("Starting: Test task", args)
        self.assertIn("Total: 100", args)

    @patch('builtins.print')
    def test_add_task_no_total(self, mock_print):
        """Test adding a task without total."""
        task_id = self.progress.add_task("Test task without total")
        
        task = self.progress.tasks[task_id]
        self.assertIsNone(task["total"])
        
        args = mock_print.call_args[0][0]
        self.assertIn("Total: unknown", args)

    @patch('builtins.print')
    @patch('time.time')
    def test_update_by_time_interval(self, mock_time, mock_print):
        """Test update triggering by time interval."""
        # Mock time progression
        mock_time.side_effect = [0, 0, 0.2]  # start, add_task, update
        
        task_id = self.progress.add_task("Test task", total=100)
        mock_print.reset_mock()
        
        self.progress.update(task_id, advance=5)
        
        # Should print because time interval exceeded
        mock_print.assert_called_once()
        args = mock_print.call_args[0][0]
        self.assertIn("Test task: 5/100", args)

    @patch('builtins.print')
    def test_update_by_frequency(self, mock_print):
        """Test update triggering by item frequency."""
        task_id = self.progress.add_task("Test task", total=100)
        mock_print.reset_mock()
        
        # Update with enough items to trigger frequency-based logging
        self.progress.update(task_id, advance=2)
        
        # Should print because we reached update_frequency=2
        mock_print.assert_called_once()

    @patch('builtins.print')
    def test_update_nonexistent_task(self, mock_print):
        """Test updating a nonexistent task."""
        self.progress.update(999, advance=1)
        
        # Should not crash or print anything
        mock_print.assert_not_called()

    @patch('builtins.print')
    @patch('time.time')
    def test_update_with_total_calculates_percentage(self, mock_time, mock_print):
        """Test that update calculates percentage and remaining time correctly."""
        mock_time.side_effect = [0, 0, 1.0]  # start, add_task, update
        
        task_id = self.progress.add_task("Test task", total=100)
        mock_print.reset_mock()
        
        self.progress.update(task_id, advance=25)
        
        args = mock_print.call_args[0][0]
        self.assertIn("25/100", args)
        self.assertIn("(25.0%)", args)
        self.assertIn("Elapsed:", args)
        self.assertIn("Remaining:", args)

    @patch('builtins.print')
    @patch('time.time')
    def test_update_without_total(self, mock_time, mock_print):
        """Test update for task without total."""
        mock_time.side_effect = [0, 0, 1.0]
        
        task_id = self.progress.add_task("Test task")  # No total
        mock_print.reset_mock()
        
        self.progress.update(task_id, advance=25)
        
        args = mock_print.call_args[0][0]
        self.assertIn("25 items", args)
        self.assertIn("Elapsed:", args)
        self.assertNotIn("Remaining:", args)

    @patch('builtins.print')
    def test_track_with_known_length(self, mock_print):
        """Test track method with iterable of known length."""
        items = [1, 2, 3, 4, 5]
        
        result = list(self.progress.track(items, description="Test tracking"))
        
        self.assertEqual(result, items)
        # Should have printed start and completion messages
        self.assertTrue(mock_print.called)

    @patch('builtins.print')
    def test_track_with_unknown_length(self, mock_print):
        """Test track method with generator (unknown length)."""
        def generate_items():
            for i in range(3):
                yield i
        
        result = list(self.progress.track(generate_items(), description="Test generator"))
        
        self.assertEqual(result, [0, 1, 2])
        self.assertTrue(mock_print.called)

    @patch('builtins.print')
    def test_track_with_explicit_total(self, mock_print):
        """Test track method with explicitly provided total."""
        items = [1, 2, 3]
        
        result = list(self.progress.track(items, total=10, description="Test explicit total"))
        
        self.assertEqual(result, items)
        # Check that the explicit total was used
        start_call = mock_print.call_args_list[0][0][0]
        self.assertIn("Total: 10", start_call)


class TestIsNonInteractiveBatch(unittest.TestCase):
    """Test suite for is_non_interactive_batch function."""

    def test_with_slurm_job_id_and_no_tty(self):
        """Test detection with SLURM_JOB_ID and no TTY."""
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345'}):
            with patch('sys.stdout.isatty', return_value=False):
                self.assertTrue(is_non_interactive_batch())

    def test_with_slurm_job_id_and_tty(self):
        """Test with SLURM_JOB_ID but with TTY (interactive session)."""
        with patch.dict(os.environ, {'SLURM_JOB_ID': '12345'}):
            with patch('sys.stdout.isatty', return_value=True):
                self.assertFalse(is_non_interactive_batch())

    def test_with_pbs_job_id(self):
        """Test detection with PBS_JOBID."""
        with patch.dict(os.environ, {'PBS_JOBID': '12345.server'}):
            with patch('sys.stdout.isatty', return_value=False):
                self.assertTrue(is_non_interactive_batch())

    def test_with_dumb_terminal(self):
        """Test detection with TERM=dumb."""
        with patch.dict(os.environ, {'TERM': 'dumb'}, clear=True):
            self.assertTrue(is_non_interactive_batch())

    def test_with_output_redirection(self):
        """Test detection with output redirection (no TTY)."""
        with patch('sys.stdout.isatty', return_value=False):
            # Remove batch environment variables
            env_vars = ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'SGE_TASK_ID']
            env_patches = {var: patch.dict(os.environ, {}, clear=False) for var in env_vars}
            
            # Remove each variable if it exists
            for var in env_vars:
                if var in os.environ:
                    del os.environ[var]
            
            try:
                self.assertTrue(is_non_interactive_batch())
            finally:
                # Restore any variables that might have been removed
                pass

    def test_slurm_pty_exception(self):
        """Test SLURM PTY exception case."""
        with patch.dict(os.environ, {'SLURM_PTY_PORT': '12345'}):
            with patch('sys.stdout.isatty', return_value=False):
                # Should return False due to SLURM_PTY_PORT exception
                self.assertFalse(is_non_interactive_batch())

    def test_interactive_session(self):
        """Test normal interactive session."""
        with patch('sys.stdout.isatty', return_value=True):
            # Remove batch environment variables
            env_vars = ['SLURM_JOB_ID', 'PBS_JOBID', 'LSB_JOBID', 'SGE_TASK_ID']
            original_values = {}
            
            # Store original values and remove variables
            for var in env_vars:
                if var in os.environ:
                    original_values[var] = os.environ[var]
                    del os.environ[var]
            
            try:
                with patch.dict(os.environ, {'TERM': 'xterm'}, clear=False):
                    self.assertFalse(is_non_interactive_batch())
            finally:
                # Restore original values
                for var, value in original_values.items():
                    os.environ[var] = value


class TestGetProgressBar(unittest.TestCase):
    """Test suite for get_progress_bar function."""

    def test_disabled_progress_bar(self):
        """Test that disabled progress bar returns NoOpProgress."""
        progress_bar = get_progress_bar(disable=True)
        self.assertIsInstance(progress_bar, _NoOpProgress)

    @patch('easyroutine.console.progress.is_non_interactive_batch')
    def test_batch_mode_progress_bar(self, mock_batch_check):
        """Test that batch mode returns LoggingProgress."""
        mock_batch_check.return_value = True
        progress_bar = get_progress_bar()
        self.assertIsInstance(progress_bar, LoggingProgress)

    @patch('easyroutine.console.progress.is_non_interactive_batch')
    def test_interactive_mode_progress_bar(self, mock_batch_check):
        """Test that interactive mode returns Rich Progress."""
        mock_batch_check.return_value = False
        progress_bar = get_progress_bar()
        # Rich Progress class name varies, so check it's not our custom classes
        self.assertNotIsInstance(progress_bar, LoggingProgress)
        self.assertNotIsInstance(progress_bar, _NoOpProgress)

    def test_force_batch_mode(self):
        """Test forcing batch mode regardless of environment."""
        progress_bar = get_progress_bar(force_batch_mode=True)
        self.assertIsInstance(progress_bar, LoggingProgress)

    @patch.dict(os.environ, {'SLURM_JOB_ID': '12345'})
    def test_slurm_environment_settings(self):
        """Test that SLURM environment gets special default settings."""
        with patch('easyroutine.console.progress.is_non_interactive_batch', return_value=True):
            progress_bar = get_progress_bar()
            self.assertIsInstance(progress_bar, LoggingProgress)
            # In SLURM environment, update_frequency should be set
            self.assertEqual(progress_bar.update_frequency, 1)


class TestNoOpProgress(unittest.TestCase):
    """Test suite for _NoOpProgress class."""

    def setUp(self):
        self.progress = _NoOpProgress()

    def test_track_yields_items(self):
        """Test that track method yields items without modification."""
        items = [1, 2, 3, 4, 5]
        result = list(self.progress.track(items))
        self.assertEqual(result, items)

    def test_context_manager(self):
        """Test _NoOpProgress as context manager."""
        with _NoOpProgress() as progress:
            self.assertIsInstance(progress, _NoOpProgress)

    def test_add_task_returns_dummy_id(self):
        """Test that add_task returns a dummy task ID."""
        task_id = self.progress.add_task("Test task", total=100)
        self.assertEqual(task_id, 0)

    def test_update_does_nothing(self):
        """Test that update method does nothing."""
        # Should not raise any exceptions
        self.progress.update(0, advance=1)
        self.progress.update(999, advance=100)  # Invalid task ID should also work


class TestProgressFunction(unittest.TestCase):
    """Test suite for the progress function."""

    @patch('easyroutine.console.progress.get_progress_bar')
    def test_progress_function_with_list(self, mock_get_progress_bar):
        """Test progress function with a list."""
        mock_progress_bar = MagicMock()
        mock_get_progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        items = [1, 2, 3]
        list(progress(items, description="Test"))
        
        # Check that get_progress_bar was called with correct parameters
        mock_get_progress_bar.assert_called_once()
        
        # Check that track was called on the progress bar
        mock_progress_bar.track.assert_called_once_with(items, total=3, description="Test")

    @patch('easyroutine.console.progress.get_progress_bar')
    def test_progress_function_with_generator(self, mock_get_progress_bar):
        """Test progress function with a generator."""
        mock_progress_bar = MagicMock()
        mock_get_progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        def gen():
            yield from [1, 2, 3]
        
        list(progress(gen(), description="Test generator"))
        
        # Generator length can't be determined, so total should be None
        mock_progress_bar.track.assert_called_once()
        args, kwargs = mock_progress_bar.track.call_args
        self.assertIsNone(kwargs['total'])

    @patch('easyroutine.console.progress.get_progress_bar')
    def test_progress_function_with_explicit_total(self, mock_get_progress_bar):
        """Test progress function with explicit total."""
        mock_progress_bar = MagicMock()
        mock_get_progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        items = [1, 2, 3]
        list(progress(items, total=10, description="Test explicit"))
        
        # Should use the explicit total
        mock_progress_bar.track.assert_called_once_with(items, total=10, description="Test explicit")

    @patch('easyroutine.console.progress.get_progress_bar')
    def test_progress_function_desc_parameter(self, mock_get_progress_bar):
        """Test progress function with desc parameter (alternative to description)."""
        mock_progress_bar = MagicMock()
        mock_get_progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        items = [1, 2, 3]
        list(progress(items, desc="Test desc"))
        
        # desc should override description
        mock_progress_bar.track.assert_called_once_with(items, total=3, description="Test desc")

    @patch('easyroutine.console.progress.get_progress_bar')
    def test_progress_function_forwards_parameters(self, mock_get_progress_bar):
        """Test that progress function forwards parameters to get_progress_bar."""
        mock_progress_bar = MagicMock()
        mock_get_progress_bar.return_value.__enter__.return_value = mock_progress_bar
        
        items = [1, 2, 3]
        list(progress(items, disable=True, force_batch_mode=True, log_interval=5))
        
        # Check that parameters were forwarded
        mock_get_progress_bar.assert_called_once_with(
            disable=True,
            force_batch_mode=True,
            log_interval=5,
            update_frequency=0
        )


if __name__ == "__main__":
    unittest.main()