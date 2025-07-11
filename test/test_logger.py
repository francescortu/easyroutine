import unittest
import logging
import tempfile
import os
from io import StringIO
from unittest.mock import patch, MagicMock
from easyroutine.logger import (
    logger,
    warning_once,
    setup_logging,
    enable_debug_logging,
    enable_info_logging,
    enable_warning_logging,
    disable_logging,
    setup_default_logging
)


class TestLogger(unittest.TestCase):
    """Test suite for easyroutine.logger module."""

    def setUp(self):
        """Set up test environment."""
        # Store original logger state
        self.original_level = logger.level
        self.original_handlers = logger.handlers[:]
        self.original_propagate = logger.propagate
        
        # Clear any existing warning_once messages
        from easyroutine.logger import _logged_once_messages
        _logged_once_messages.clear()

    def tearDown(self):
        """Restore original logger state."""
        logger.setLevel(self.original_level)
        logger.handlers.clear()
        logger.handlers.extend(self.original_handlers)
        logger.propagate = self.original_propagate
        
        # Clear warning_once messages
        from easyroutine.logger import _logged_once_messages
        _logged_once_messages.clear()

    def test_logger_exists(self):
        """Test that the logger object exists and has correct name."""
        self.assertEqual(logger.name, "easyroutine")
        self.assertIsInstance(logger, logging.Logger)

    def test_warning_once_function_exists(self):
        """Test that warning_once is available as a function."""
        # Test that we can call warning_once
        with patch.object(logger, 'warning') as mock_warning:
            warning_once("Test message")
            mock_warning.assert_called_once_with("Test message")

    def test_warning_once_method_exists(self):
        """Test that warning_once is attached to logger as a method."""
        self.assertTrue(hasattr(logger, 'warning_once'))
        self.assertTrue(callable(logger.warning_once))

    def test_warning_once_logs_first_time(self):
        """Test that warning_once logs the first occurrence of a message."""
        with patch.object(logger, 'warning') as mock_warning:
            logger.warning_once("First time message")
            mock_warning.assert_called_once_with("First time message")

    def test_warning_once_ignores_duplicate(self):
        """Test that warning_once ignores subsequent identical messages."""
        with patch.object(logger, 'warning') as mock_warning:
            # Log the same message twice
            logger.warning_once("Duplicate message")
            logger.warning_once("Duplicate message")
            
            # Should only be called once
            mock_warning.assert_called_once_with("Duplicate message")

    def test_warning_once_different_messages(self):
        """Test that warning_once logs different messages separately."""
        with patch.object(logger, 'warning') as mock_warning:
            logger.warning_once("Message A")
            logger.warning_once("Message B")
            logger.warning_once("Message A")  # Should not log again
            
            # Should be called twice (once for each unique message)
            self.assertEqual(mock_warning.call_count, 2)
            mock_warning.assert_any_call("Message A")
            mock_warning.assert_any_call("Message B")

    def test_setup_default_logging(self):
        """Test that setup_default_logging configures the logger correctly."""
        # Clear existing handlers
        logger.handlers.clear()
        
        setup_default_logging()
        
        # Check that logger has handlers
        self.assertTrue(len(logger.handlers) > 0)
        self.assertEqual(logger.level, logging.INFO)
        self.assertFalse(logger.propagate)

    def test_setup_default_logging_no_duplicate_handlers(self):
        """Test that setup_default_logging doesn't add duplicate handlers."""
        # Call setup_default_logging multiple times
        setup_default_logging()
        handler_count_after_first = len(logger.handlers)
        
        setup_default_logging()
        handler_count_after_second = len(logger.handlers)
        
        # Should not add duplicate handlers
        self.assertEqual(handler_count_after_first, handler_count_after_second)

    def test_setup_logging_file_only(self):
        """Test setup_logging with file output only."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            setup_logging(level="DEBUG", file=tmp_filename, console=False)
            
            # Test logging to file
            logger.debug("Debug message")
            logger.info("Info message")
            
            # Check file contents
            with open(tmp_filename, 'r') as f:
                content = f.read()
                self.assertIn("Debug message", content)
                self.assertIn("Info message", content)
                
            # Check logger configuration
            self.assertEqual(logger.level, logging.DEBUG)
            
        finally:
            os.unlink(tmp_filename)

    def test_setup_logging_console_only(self):
        """Test setup_logging with console output only."""
        setup_logging(level="WARNING", file=None, console=True)
        
        # Should have at least one handler (console)
        self.assertTrue(len(logger.handlers) > 0)
        self.assertEqual(logger.level, logging.WARNING)

    def test_setup_logging_both_file_and_console(self):
        """Test setup_logging with both file and console output."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            setup_logging(level="INFO", file=tmp_filename, console=True)
            
            # Should have at least two handlers
            self.assertTrue(len(logger.handlers) >= 2)
            self.assertEqual(logger.level, logging.INFO)
            
        finally:
            os.unlink(tmp_filename)

    def test_enable_debug_logging(self):
        """Test enable_debug_logging function."""
        enable_debug_logging()
        
        self.assertEqual(logger.level, logging.DEBUG)
        # Check that all handlers have debug level
        for handler in logger.handlers:
            self.assertEqual(handler.level, logging.DEBUG)

    def test_enable_info_logging(self):
        """Test enable_info_logging function."""
        enable_info_logging()
        
        self.assertEqual(logger.level, logging.INFO)
        for handler in logger.handlers:
            self.assertEqual(handler.level, logging.INFO)

    def test_enable_warning_logging(self):
        """Test enable_warning_logging function."""
        enable_warning_logging()
        
        self.assertEqual(logger.level, logging.WARNING)
        for handler in logger.handlers:
            self.assertEqual(handler.level, logging.WARNING)

    def test_disable_logging(self):
        """Test disable_logging function."""
        disable_logging()
        
        # Logger level should be set higher than CRITICAL
        self.assertGreater(logger.level, logging.CRITICAL)
        for handler in logger.handlers:
            self.assertGreater(handler.level, logging.CRITICAL)

    def test_setup_logging_custom_format(self):
        """Test setup_logging with custom format."""
        custom_format = "%(levelname)s: %(message)s"
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            setup_logging(
                level="INFO", 
                file=tmp_filename, 
                console=False, 
                fmt=custom_format
            )
            
            logger.info("Test message")
            
            # Check that custom format is used
            with open(tmp_filename, 'r') as f:
                content = f.read()
                self.assertIn("INFO: Test message", content)
                
        finally:
            os.unlink(tmp_filename)

    def test_setup_logging_invalid_level(self):
        """Test setup_logging with invalid level defaults to INFO."""
        setup_logging(level="INVALID_LEVEL", console=True, file=None)
        
        # Should default to INFO level
        self.assertEqual(logger.level, logging.INFO)

    def test_logging_level_hierarchy(self):
        """Test that logging level hierarchy works correctly."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_filename = tmp_file.name
        
        try:
            # Set to WARNING level
            setup_logging(level="WARNING", file=tmp_filename, console=False)
            
            # Log messages at different levels
            logger.debug("Debug message")  # Should not appear
            logger.info("Info message")    # Should not appear
            logger.warning("Warning message")  # Should appear
            logger.error("Error message")    # Should appear
            
            # Check file contents
            with open(tmp_filename, 'r') as f:
                content = f.read()
                self.assertNotIn("Debug message", content)
                self.assertNotIn("Info message", content)
                self.assertIn("Warning message", content)
                self.assertIn("Error message", content)
                
        finally:
            os.unlink(tmp_filename)

    def test_logger_propagation(self):
        """Test that logger propagation is set correctly."""
        setup_default_logging()
        self.assertFalse(logger.propagate)

    def test_warning_once_persistence(self):
        """Test that warning_once messages persist across logger reconfigurations."""
        # Log a message with warning_once
        with patch.object(logger, 'warning') as mock_warning:
            logger.warning_once("Persistent message")
            mock_warning.assert_called_once()
            mock_warning.reset_mock()
            
            # Reconfigure logger
            setup_logging(level="INFO", console=True, file=None)
            
            # Try to log the same message again
            logger.warning_once("Persistent message")
            mock_warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()