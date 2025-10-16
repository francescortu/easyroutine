"""
CLI backend for dashboard using Rich library.

Displays dashboard information in the terminal with nice formatting.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from typing import Optional
import time
import threading
import sys
import io

from .state import DashboardState, StepState
from .utils import format_duration, format_value


class CaptureStream:
    """Stream wrapper that captures output to dashboard while allowing normal output."""

    def __init__(self, original_stream, state: DashboardState, is_stderr: bool = False):
        self.original_stream = original_stream
        self.state = state
        self.is_stderr = is_stderr
        self._line_buffer = ""

    def write(self, text):
        """Write to original stream and capture to dashboard."""
        # Write to original stream
        self.original_stream.write(text)

        # Capture complete lines to dashboard
        if text:
            self._line_buffer += text
            while "\n" in self._line_buffer:
                line, self._line_buffer = self._line_buffer.split("\n", 1)
                if line.strip():  # Only capture non-empty lines
                    self.state.add_output_line(line)

    def flush(self):
        """Flush the original stream."""
        self.original_stream.flush()

    def isatty(self):
        """Check if original stream is a TTY."""
        return hasattr(self.original_stream, "isatty") and self.original_stream.isatty()

    def __getattr__(self, name):
        """Delegate other attributes to original stream."""
        return getattr(self.original_stream, name)


class DashboardConsole(Console):
    """Custom Console that also logs to dashboard state."""

    def __init__(self, state: DashboardState, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = state
        self._original_file = None

    def print(self, *args, **kwargs):
        """Override print to capture output."""
        # Capture the output
        if args:
            line = " ".join(str(arg) for arg in args)
            self.state.add_output_line(line)
        # Still print normally
        super().print(*args, **kwargs)


class CLIBackend:
    """
    Terminal-based dashboard display using Rich.
    """

    def __init__(
        self,
        state: DashboardState,
        auto_refresh: bool = True,
        refresh_rate: float = 0.5,
    ):
        """
        Initialize CLI backend.

        Args:
            state: Dashboard state to display
            auto_refresh: Whether to auto-refresh display
            refresh_rate: Refresh interval in seconds
        """
        self.state = state
        self.console = Console()
        self.auto_refresh = auto_refresh
        self.refresh_rate = refresh_rate

        self._live: Optional[Live] = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        """Start the live display."""
        if self.auto_refresh:
            self._running = True
            # Don't redirect stdout - it conflicts with Live display
            # Users should use dash.print() for output capture
            self._live = Live(
                self._build_layout(),
                console=self.console,
                refresh_per_second=1 / self.refresh_rate,
                screen=False,
            )
            self._live.start()

            # Register callback for state changes
            self.state.add_callback(self._on_state_change)

    def stop(self):
        """Stop the live display."""
        self._running = False
        if self._live:
            self._live.stop()
            self._live = None

    def _on_state_change(self):
        """Called when state changes - update display."""
        if self._live and self._running:
            try:
                self._live.update(self._build_layout())
            except Exception:
                pass  # Ignore errors during refresh

    def _build_layout(self):
        """Build the complete dashboard layout."""
        layout = Layout()

        # Calculate header size based on metadata
        header_size = 3
        if self.state.metadata or self.state.arguments:
            header_size = 5  # More room for metadata

        # Split into sections
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="body"),
            Layout(name="terminal", size=12),
            Layout(name="footer", size=3),
        )

        # Header
        layout["header"].update(self._build_header())

        # Body - split into left (steps) and right (variables)
        layout["body"].split_row(
            Layout(name="steps", ratio=2), Layout(name="variables", ratio=1)
        )

        layout["body"]["steps"].update(self._build_steps_panel())
        layout["body"]["variables"].update(self._build_variables_panel())

        # Terminal output (full width)
        layout["terminal"].update(self._build_terminal_panel())

        # Footer
        layout["footer"].update(self._build_footer())

        return layout

    def _build_header(self) -> Panel:
        """Build header panel with title and metadata."""
        title_text = Text(self.state.title, style="bold cyan", justify="center")

        # Add arguments if available
        if self.state.arguments:
            args_parts = [
                f"{k}={format_value(v)}"
                for k, v in list(self.state.arguments.items())[:6]
            ]
            args_text = " | ".join(args_parts)
            if len(self.state.arguments) > 6:
                args_text += f" (+{len(self.state.arguments) - 6} more)"
            title_text.append(f"\nArgs: {args_text}", style="yellow dim")

        # Add metadata if available
        if self.state.metadata:
            meta_parts = [
                f"{k}={format_value(v)}"
                for k, v in list(self.state.metadata.items())[:6]
            ]
            meta_text = " | ".join(meta_parts)
            if len(self.state.metadata) > 6:
                meta_text += f" (+{len(self.state.metadata) - 6} more)"
            title_text.append(f"\nConfig: {meta_text}", style="green dim")

        return Panel(title_text, box=box.ROUNDED, style="cyan")

    def _build_steps_panel(self) -> Panel:
        """Build panel showing all steps/stages."""
        table = Table(
            show_header=True, header_style="bold magenta", box=box.SIMPLE, expand=True
        )

        table.add_column("Step", style="cyan", no_wrap=True)
        table.add_column("Progress", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Time", justify="right")

        steps = self.state.get_step_hierarchy()

        if not steps:
            table.add_row("No steps yet", "-", "-", "-")
        else:
            for step in steps:
                # Add indentation for nested steps
                indent = "  " * self._get_nesting_level(step)
                description = f"{indent}{step.description}"

                # Progress
                if step.total:
                    progress_pct = f"{step.progress:.0f}%"
                    progress_bar = self._make_progress_bar(step.progress or 0, width=10)
                    progress = (
                        f"{progress_bar} {progress_pct} ({step.completed}/{step.total})"
                    )
                else:
                    progress = f"{step.completed} items"

                # Status
                if step.status == "completed":
                    status = "✓"
                    status_style = "green"
                elif step.status == "failed":
                    status = "✗"
                    status_style = "red"
                else:
                    status = "●"
                    status_style = "yellow"

                # Time
                time_str = format_duration(step.elapsed_time)

                table.add_row(
                    description, progress, Text(status, style=status_style), time_str
                )

        return Panel(table, title="Steps", box=box.ROUNDED, style="magenta")

    def _build_variables_panel(self) -> Panel:
        """Build panel showing tracked variables."""
        if not self.state.variables:
            return Panel("No variables tracked yet", title="Variables", box=box.ROUNDED)

        table = Table(
            show_header=True, header_style="bold green", box=box.SIMPLE, expand=True
        )

        table.add_column("Variable", style="cyan")
        table.add_column("Latest", justify="right")
        table.add_column("Count", justify="right")

        for name, history in sorted(self.state.variables.items()):
            latest_value = history.values[-1] if history.values else "N/A"
            count = len(history.values)

            table.add_row(name, format_value(latest_value), str(count))

        return Panel(table, title="Variables", box=box.ROUNDED, style="green")

    def _build_terminal_panel(self) -> Panel:
        """Build panel showing terminal output."""
        output = self.state.get_output(last_n=10)

        if not output:
            content = Text("No output yet", style="dim")
        else:
            content = Text("\n".join(output[-10:]))  # Last 10 lines

        return Panel(content, title="Terminal Output", box=box.ROUNDED, style="blue")

    def _build_footer(self) -> Panel:
        """Build footer with summary stats."""
        active_steps = self.state.get_active_steps()
        completed_steps = [
            s for s in self.state.steps.values() if s.status == "completed"
        ]

        stats = (
            f"Active: {len(active_steps)} | "
            f"Completed: {len(completed_steps)} | "
            f"Total Steps: {len(self.state.steps)} | "
            f"Variables: {len(self.state.variables)}"
        )

        return Panel(Text(stats, justify="center"), box=box.ROUNDED, style="dim")

    def _get_nesting_level(self, step: StepState) -> int:
        """Calculate nesting level of a step."""
        level = 0
        current = step
        while current.parent_id:
            level += 1
            current = self.state.get_step(current.parent_id)
            if not current:
                break
        return level

    def _make_progress_bar(self, percentage: float, width: int = 10) -> str:
        """Create a simple ASCII progress bar."""
        filled = int((percentage / 100) * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar

    def print_summary(self):
        """Print a final summary (non-live)."""
        self.console.print("\n")
        self.console.print(
            Panel(
                f"[bold cyan]{self.state.title}[/bold cyan]\n"
                f"Completed in {format_duration((time.time() - self.state.start_time.timestamp()))}",
                title="Experiment Complete",
                box=box.DOUBLE,
            )
        )

        # Print final step status
        if self.state.steps:
            self.console.print(self._build_steps_panel())

        # Print variable summary
        if self.state.variables:
            self.console.print(self._build_variables_panel())
