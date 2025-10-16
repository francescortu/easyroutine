"""
Utility functions for dashboard functionality.

Includes output capture, persistence helpers, etc.
"""

import sys
import io
from typing import Optional, Callable


class OutputCapture:
    """
    Context manager to capture stdout/stderr output.
    """

    def __init__(self, callback: Optional[Callable[[str], None]] = None):
        """
        Initialize output capture.

        Args:
            callback: Optional function to call with each captured line
        """
        self.callback = callback
        self._original_stdout = None
        self._original_stderr = None
        self._capture_stdout = None
        self._capture_stderr = None

    def __enter__(self):
        """Start capturing output."""
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

        # Create string buffers
        self._capture_stdout = io.StringIO()
        self._capture_stderr = io.StringIO()

        # Redirect stdout and stderr
        sys.stdout = TeeOutput(
            self._original_stdout, self._capture_stdout, self.callback
        )
        sys.stderr = TeeOutput(
            self._original_stderr, self._capture_stderr, self.callback
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop capturing output."""
        # Restore original stdout/stderr
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

        return False

    def get_output(self) -> str:
        """Get captured output as string."""
        stdout = self._capture_stdout.getvalue() if self._capture_stdout else ""
        stderr = self._capture_stderr.getvalue() if self._capture_stderr else ""
        return stdout + stderr


class TeeOutput:
    """
    Output stream that writes to multiple destinations.
    Allows seeing output in terminal while also capturing it.
    """

    def __init__(self, original, capture, callback=None):
        self.original = original
        self.capture = capture
        self.callback = callback
        self._line_buffer = ""

    def write(self, text):
        """Write to both original and capture streams."""
        self.original.write(text)
        self.capture.write(text)

        # Call callback for each complete line
        if self.callback:
            self._line_buffer += text
            while "\n" in self._line_buffer:
                line, self._line_buffer = self._line_buffer.split("\n", 1)
                self.callback(line)

    def flush(self):
        """Flush both streams."""
        self.original.flush()
        self.capture.flush()

    def isatty(self):
        """Check if original stream is a TTY."""
        return self.original.isatty()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Examples:
        1.5 -> "1.5s"
        65 -> "1m 5s"
        3661 -> "1h 1m"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    if minutes < 60:
        if remaining_seconds > 0:
            return f"{minutes}m {remaining_seconds:.0f}s"
        return f"{minutes}m"

    hours = int(minutes // 60)
    remaining_minutes = minutes % 60

    if remaining_minutes > 0:
        return f"{hours}h {remaining_minutes}m"
    return f"{hours}h"


def format_value(value) -> str:
    """
    Format a value for display.
    """
    if isinstance(value, float):
        # Use scientific notation for very small/large numbers
        if abs(value) < 0.001 or abs(value) > 10000:
            return f"{value:.2e}"
        return f"{value:.4f}"
    return str(value)
