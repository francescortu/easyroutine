import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Literal, Optional
from rich.logging import RichHandler
from pathlib import Path

import logging
import sys
from rich.logging import RichHandler

# Library-wide logger
logger = logging.getLogger("easyroutine")
_logged_once_messages = set()

def warning_once(message: str):
    """
    Logs a warning message only once per runtime session.
    
    This function prevents duplicate warning messages from cluttering the logs by
    tracking which messages have already been logged. Subsequent calls with the
    same message will be silently ignored.
    
    Args:
        message (str): The warning message to log. Only the first occurrence
            of this exact message will be logged during the program's runtime.
    
    Returns:
        None: This function has no return value.
    
    Side Effects:
        - Logs the warning message to the easyroutine logger (first occurrence only)
        - Adds the message to an internal set to track logged messages
    
    Example:
        >>> warning_once("This is a warning")
        # Logs: WARNING - This is a warning
        >>> warning_once("This is a warning")
        # No output - message already logged
        >>> warning_once("Different warning")
        # Logs: WARNING - Different warning
    
    Note:
        The tracking of logged messages persists for the entire program runtime.
        Messages are compared for exact string equality.
    """
    if message not in _logged_once_messages:
        _logged_once_messages.add(message)
        logger.warning(message)

# Attach the method to the logger for convenience
logger.warning_once = warning_once

def setup_default_logging():
    """
    Set up default logging configuration for the easyroutine package.
    
    This function initializes the easyroutine logger with sensible defaults,
    ensuring that INFO-level logs are displayed to the console using RichHandler
    for enhanced formatting. The configuration can be overridden by calling
    setup_logging() with custom parameters.
    
    The default configuration includes:
    - Log level: INFO
    - Handler: RichHandler with rich tracebacks and markup support
    - Formatter: Simple message format (RichHandler handles its own formatting)
    - Propagation: Disabled to prevent duplicate messages in root logger
    
    Returns:
        None: This function configures the logger in-place.
    
    Side Effects:
        - Configures the global easyroutine logger
        - Adds a console handler if none exists
        - Sets logger level to INFO
        - Disables log propagation to root logger
    
    Note:
        This function only adds handlers if the logger doesn't already have any,
        preventing duplicate handlers from being added if called multiple times.
        This function is called automatically when the logger module is imported.
    """
    if not logger.hasHandlers():  # Avoid adding multiple handlers
        logger.setLevel(logging.INFO)  # Default level (user can change)

        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter("%(message)s")  # RichHandler formats on its own
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        # Ensure logs don’t propagate to the root logger (prevents duplicate messages)
        logger.propagate = False


setup_default_logging()  # Apply default configuration


def setup_logging(level="INFO", file=None, console=True, fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"):
    """
    Configure comprehensive logging settings for the easyroutine package.
    
    This function provides full control over the logging configuration, allowing
    users to customize logging levels, output destinations, and message formatting.
    It replaces any existing handlers with the new configuration.
    
    Args:
        level (str, optional): The logging level to set. Must be one of:
            'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
            Defaults to 'INFO'. Case-insensitive.
        file (str, optional): Path to a log file where messages should be saved.
            If None, no file logging is performed. If provided, creates a
            FileHandler to write logs to this file. Defaults to None.
        console (bool, optional): Whether to enable console logging output.
            If True, logs are displayed in the terminal using RichHandler.
            Defaults to True.
        fmt (str, optional): Log message format string for file output.
            Uses standard Python logging format specifiers. Note that console
            output uses RichHandler's built-in formatting. Defaults to a
            standard format with timestamp, logger name, level, and message.
    
    Returns:
        None: This function configures the logger in-place.
    
    Side Effects:
        - Clears all existing handlers from the easyroutine logger
        - Sets the logger level according to the 'level' parameter
        - Adds a FileHandler if 'file' is specified
        - Adds a RichHandler for console output if 'console' is True
        - Logs a confirmation message about the new configuration
    
    Example:
        >>> # Basic setup with file logging
        >>> setup_logging(level="DEBUG", file="debug.log")
        
        >>> # Console-only logging with custom format
        >>> setup_logging(level="WARNING", console=True, file=None)
        
        >>> # Both file and console with custom format
        >>> setup_logging(
        ...     level="INFO", 
        ...     file="app.log", 
        ...     console=True,
        ...     fmt="[%(asctime)s] %(levelname)s: %(message)s"
        ... )
    
    Note:
        - Invalid level names will default to INFO level
        - RichHandler provides enhanced console output with colors and formatting
        - File handler uses the provided format, console uses RichHandler's format
        - This function completely replaces the existing logging configuration
    """

    # Clear any existing handlers (to prevent duplicates)
    logger.handlers.clear()

    # Set log level
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create formatter
    formatter = logging.Formatter(fmt)

    # Add file handler if specified
    if file:
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add console handler if specified
    if console:
        console_handler = RichHandler(rich_tracebacks=True, markup=True)
        console_handler.setLevel(logger.level)
        console_handler.setFormatter(logging.Formatter("%(message)s"))  # RichHandler does its own formatting
        logger.addHandler(console_handler)

    logger.info(f"Logging configured. Level: {level}, File: {file or 'None'}")


def enable_debug_logging():
    """
    Enable debug-level logging for the easyroutine package.
    
    This convenience function sets the logger and all its handlers to DEBUG level,
    which is the most verbose logging level. This will display all log messages
    including DEBUG, INFO, WARNING, ERROR, and CRITICAL messages.
    
    Returns:
        None: This function modifies the logger configuration in-place.
    
    Side Effects:
        - Sets the easyroutine logger level to DEBUG
        - Sets all existing handlers to DEBUG level
        - Logs a debug message confirming the change
    
    Example:
        >>> enable_debug_logging()
        # DEBUG - Debug logging enabled for easyroutine.
        
    Note:
        This function operates on the existing handlers. If no handlers are
        configured, you may need to call setup_logging() or setup_default_logging()
        first to ensure log messages are displayed.
    """
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled for easyroutine.")
    
def enable_info_logging():
    """
    Enable info-level logging for the easyroutine package.
    
    This convenience function sets the logger and all its handlers to INFO level,
    which will display INFO, WARNING, ERROR, and CRITICAL messages while
    filtering out DEBUG messages.
    
    Returns:
        None: This function modifies the logger configuration in-place.
    
    Side Effects:
        - Sets the easyroutine logger level to INFO
        - Sets all existing handlers to INFO level
        - Logs an info message confirming the change
    
    Example:
        >>> enable_info_logging()
        # INFO - Info logging enabled for easyroutine.
        
    Note:
        This is typically the default logging level for most applications.
        This function operates on existing handlers - ensure handlers are
        configured before calling this function.
    """
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)
    logger.info("Info logging enabled for easyroutine.")

def enable_warning_logging():
    """
    Enable warning-level logging for the easyroutine package.
    
    This convenience function sets the logger and all its handlers to WARNING level,
    which will display only WARNING, ERROR, and CRITICAL messages while
    filtering out DEBUG and INFO messages. This is useful for quieter operation
    where only important issues should be reported.
    
    Returns:
        None: This function modifies the logger configuration in-place.
    
    Side Effects:
        - Sets the easyroutine logger level to WARNING
        - Sets all existing handlers to WARNING level
        - Logs a warning message confirming the change
    
    Example:
        >>> enable_warning_logging()
        # WARNING - Warning logging enabled for easyroutine.
        
    Note:
        This level is useful for production environments where you want to
        reduce log verbosity but still see potential issues.
    """
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logging.WARNING)
    logger.warning("Warning logging enabled for easyroutine.")


def disable_logging():
    """
    Disable all logging output for the easyroutine package.
    
    This function effectively turns off all logging by setting the logger level
    to a value higher than CRITICAL, ensuring that no log messages will be
    displayed regardless of their severity. This is useful for silent operation
    or when logging output is not desired.
    
    Returns:
        None: This function modifies the logger configuration in-place.
    
    Side Effects:
        - Sets the easyroutine logger level to CRITICAL + 1 (effectively off)
        - Sets all existing handlers to CRITICAL + 1 level
        - Attempts to log a final info message (which may not be displayed
          due to the timing of when the level change takes effect)
    
    Example:
        >>> disable_logging()
        # All subsequent log messages will be suppressed
        
    Note:
        To re-enable logging after calling this function, use one of the
        enable_*_logging() functions or call setup_logging() with desired parameters.
        The final "logging disabled" message may or may not appear depending on
        when the level change takes effect relative to the logging call.
    """
    logger.setLevel(logging.CRITICAL + 1)  # Effectively turns off logging
    for handler in logger.handlers:
        handler.setLevel(logging.CRITICAL + 1)
    logger.info("Logging has been disabled.")

# class LambdaLogger():
#     @staticmethod
#     def log(message: str, level: str = "INFO"):
#         """
#         Log a message to AWS CloudWatch Logs.
#         args:
#         message (str): The message to log.
#         level (str): The logging level. Default is INFO.
#         """
#         log_level = getattr(logging, level.upper(), logging.INFO)
#         logging.getLogger().log(log_level, message)
        
#     def info(self, message: str):
#         return LambdaLogger.log(message, "INFO")
#     def warning(self, message: str):
#         return LambdaLogger.log(message, "WARNING")
#     def error(self, message: str):
#         return LambdaLogger.log(message, "ERROR")
    

# class Logger:
#     """
#     Logger class to log messages to a file and optionally to the console using rich for console output.
#     """
#     def __init__(
#         self,
#         logname: str,
#         level: str = "INFO",
#         disable_file: bool = False,
#         disable_stdout: bool = False,
#         log_file_path: Optional[str] = None,
#     ):
#         """
#         args:
#         logname (str): The name of the logger.
#         level (str): The logging level. Default is INFO.
#         disable_file (bool): If True, the logger will not log to a file. Default is False.
#         disable_stdout (bool): If True, the logger will not log to the console. Default is False.
#         log_file_path (str): The path to the log file. If None, the log file will be saved as {logname}.log.
#         """
#         self.logname = logname
#         self.level = getattr(logging, level.upper(), logging.INFO)
#         self.file_log = log_file_path
#         self.maxBytes = 1024 * 1024 * 10  # 10 MB
#         self.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#         if self.file_log: # if not just stdout
#             self.file_logger = self._init_file_logger()
#             self.file_logger.disabled = disable_file
#         else:
#             self.file_logger = None
#         self.std_out_logger = self._init_stdout_logger()
#         self.std_out_logger.disabled = disable_stdout

#     def __call__(
#         self,
#         msg: str,
#         level: Literal["info", "debug", "warning", "error"],
#         std_out: bool = False,
#     ):
#         self.log(msg=msg, level=level, std_out=std_out)

#     def _init_file_logger(self):
#         logger = logging.getLogger(self.logname)
#         logger.setLevel(self.level)
#         if not self.file_log:
#             self.file_log = f"{self.logname}.log"
#         if not any(
#             isinstance(handler, RotatingFileHandler) for handler in logger.handlers
#         ):
#             logging_handler = RotatingFileHandler(
#                 filename=Path(self.file_log),
#                 mode="a",
#                 maxBytes=self.maxBytes,
#                 backupCount=2,
#                 encoding=None,
#                 delay=False,
#             )
#             logging_handler.setFormatter(logging.Formatter(self.format))
#             logger.addHandler(logging_handler)

#         return logger

#     def _init_stdout_logger(self):
#         stdout_logger = logging.getLogger(f"{self.logname}_stdout")
#         stdout_logger.setLevel(self.level)

#         if not any(
#             isinstance(handler, RichHandler)
#             for handler in stdout_logger.handlers
#         ):
#             stdout_handler = RichHandler()
#             stdout_handler.setFormatter(logging.Formatter(self.format))
#             stdout_logger.addHandler(stdout_handler)

#         return stdout_logger

#     def log(
#         self,
#         msg: str,
#         level: Literal["info", "debug", "warning", "error"],
#         std_out: bool = False,
#     ):
#         log_method = {
#             "info": logging.INFO,
#             "debug": logging.DEBUG,
#             "warning": logging.WARNING,
#             "error": logging.ERROR,
#         }.get(level, logging.INFO)

#         if self.file_log and self.file_logger:
#             self.file_logger.log(log_method, msg)
#         if std_out:
#             self.std_out_logger.log(log_method, msg)

#     def info(self, msg: str, std_out: bool = False):
#         self.log(msg, "info", std_out)
    
#     def debug(self, msg: str, std_out: bool = False):
#         self.log(msg, "debug", std_out)
        
#     def warning(self, msg: str, std_out: bool = False):
#         self.log(msg, "warning", std_out)
        
#     def error(self, msg: str, std_out: bool = False):
#         self.log(msg, "error", std_out)
