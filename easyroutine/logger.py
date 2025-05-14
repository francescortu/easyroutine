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
    Subsequent calls with the same message will be ignored.
    """
    if message not in _logged_once_messages:
        _logged_once_messages.add(message)
        logger.warning(message)

# Attach the method to the logger for convenience
logger.warning_once = warning_once

def setup_default_logging():
    """
    Set up default logging for easyroutine. This ensures that INFO-level logs are printed
    to the console using RichHandler, while allowing the user to override this configuration.
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
    Configure logging for easyroutine.

    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Default: INFO.
        file (str, optional): Path to log file. If None, logs are not saved to a file.
        console (bool): Whether to log to the console. Default: True.
        fmt (str): Log message format. Default: standard logging format.
    
    Example Usage:
        setup_logging(level="DEBUG", file="easyroutine.log", console=True)
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
    Enable debug logging for easyroutine. Prints all DEBUG-level logs.
    """
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled for easyroutine.")
    
def enable_info_logging():
    """
    Enable info logging for easyroutine. Prints all INFO-level logs.
    """
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)
    logger.info("Info logging enabled for easyroutine.")

def enable_warning_logging():
    """
    Enable warning logging for easyroutine. Prints all WARNING-level logs.
    """
    logger.setLevel(logging.WARNING)
    for handler in logger.handlers:
        handler.setLevel(logging.WARNING)
    logger.warning("Warning logging enabled for easyroutine.")


def disable_logging():
    """
    Disable all logging for easyroutine.
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
