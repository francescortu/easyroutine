"""
Research Dashboard - Main Interface

Simple, intuitive dashboard for tracking experiments in real-time.
Supports both CLI and web interfaces.

Usage:
    from easyroutine.console import Dashboard

    dash = Dashboard(title="My Experiment")

    # Log configuration
    dash.log_arguments(args)
    dash.log_metadata({"model": "gpt2"})

    # Track progress (automatic)
    for item in dash.progress(data, description="Processing"):
        process(item)

    # Track variables
    dash.track_variable("loss", loss_value, step=epoch)

    # Manual step control (if needed)
    step_id = dash.start_step("Training", total=100)
    for i in range(100):
        train()
        dash.update_step(step_id)
    dash.stop_step(step_id)

    # Serve web UI
    dash.serve(port=8080)
"""

from typing import Optional, Any, Dict, Iterable, TypeVar
from pathlib import Path
import atexit
import sys

from .state import DashboardState
from .cli_backend import CLIBackend
from .none_backend import NoneBackend
from .utils import OutputCapture

T = TypeVar("T")


class StepContext:
    """Context manager for manual step control."""

    def __init__(self, dashboard: "Dashboard", step_id: str):
        self.dashboard = dashboard
        self.step_id = step_id

    def update(self, advance: int = 1):
        """Update step progress."""
        self.dashboard.update_step(self.step_id, advance=advance)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            # Step failed
            self.dashboard.fail_step(self.step_id, str(exc_val))
        else:
            # Step completed
            self.dashboard.stop_step(self.step_id)
        return False


class Dashboard:
    """
    Main dashboard interface for tracking experiments.

    Features:
    - Automatic progress tracking with nested support
    - Variable tracking over time
    - Output capture
    - Multiple backends (CLI, Streamlit)
    - Auto-save to JSON
    """

    def __init__(
        self,
        title: str = "Experiment",
        description: str = "",
        mode: str = "auto",
        save_dir: Optional[str] = None,
        auto_save: bool = False,
        auto_refresh: bool = True,
        refresh_rate: float = 0.5,
        capture_output: bool = True,
        streamlit_port: int = 8501,
        streamlit_open_browser: bool = True,
    ):
        """
        Initialize dashboard.

        Args:
            title: Experiment title
            description: Experiment description
            mode: Display mode - "auto", "cli", "streamlit", "none"
                - "cli": Rich-based terminal dashboard with Live display
                - "streamlit": Web-based dashboard (enables stdout capture)
                - "none": No dashboard, fallback to standard output
                - "auto": Automatically detect best mode
            save_dir: Directory to save dashboard data (None = no save)
            auto_save: Whether to auto-save state periodically
            auto_refresh: Whether to auto-refresh display
            refresh_rate: Display refresh rate in seconds
            capture_output: Whether to automatically capture stdout/stderr
                           (disabled for CLI mode with auto_refresh due to conflicts)
            streamlit_port: Port for Streamlit server (mode="streamlit" only)
            streamlit_open_browser: Whether to open browser automatically
        """
        self.state = DashboardState(title=title, description=description)
        self.save_dir = Path(save_dir) if save_dir else None
        self.auto_save = auto_save
        self._capture_output_enabled = capture_output
        self._streamlit_port = streamlit_port
        self._streamlit_open_browser = streamlit_open_browser

        # Mode selection
        self.mode = self._detect_mode(mode)
        self.backend = None

        # Output capture context
        self._output_capture_context = None

        # Initialize backend based on mode
        if self.mode == "cli":
            self.backend = CLIBackend(
                self.state, auto_refresh=auto_refresh, refresh_rate=refresh_rate
            )
            # Disable auto-capture for CLI with live display due to conflicts
            # Users should use dash.print() instead
            if self._capture_output_enabled and auto_refresh:
                self._capture_output_enabled = False
            self.backend.start()
            
        elif self.mode == "streamlit":
            # Import here to avoid dependency issues
            from .streamlit_backend import StreamlitBackend
            
            self.backend = StreamlitBackend(
                self.state,
                auto_refresh=auto_refresh,
                refresh_rate=refresh_rate,
                port=streamlit_port,
                open_browser=streamlit_open_browser,
            )
            # Enable output capture for Streamlit (no conflicts with Rich Live!)
            # stdout/stderr will be captured and displayed in web UI
            if not self._capture_output_enabled:
                self._capture_output_enabled = True
            self.backend.start()
            
        elif self.mode == "none":
            self.backend = NoneBackend(
                self.state, auto_refresh=False, refresh_rate=refresh_rate
            )
            self.backend.start()
            # No live display, so we can capture if needed
            # But for "none" mode, we typically don't capture

        # Start output capture if enabled (only for non-live backends)
        if self._capture_output_enabled:
            self._start_output_capture()

        # Auto-save on exit
        if self.auto_save and self.save_dir:
            atexit.register(self._save_on_exit)

    def _detect_mode(self, mode: str) -> str:
        """Detect appropriate display mode."""
        if mode != "auto":
            return mode

        # Check if in Jupyter
        try:
            get_ipython = sys.modules.get("IPython")
            if get_ipython:
                return "cli"  # For now, use CLI even in notebooks
        except Exception:
            pass

        # Check if stdout is a terminal
        if sys.stdout.isatty():
            return "cli"

        # Default to CLI
        return "cli"

    def _save_on_exit(self):
        """Save state on program exit."""
        if self.save_dir:
            self.save()

    def _start_output_capture(self):
        """Start capturing stdout/stderr."""

        def callback(line: str):
            self.state.add_output_line(line)

        self._output_capture_context = OutputCapture(callback=callback)
        self._output_capture_context.__enter__()

    def _stop_output_capture(self):
        """Stop capturing output."""
        if self._output_capture_context:
            self._output_capture_context.__exit__(None, None, None)
            self._output_capture_context = None

    # --- Configuration ---

    def log_arguments(self, args: Any):
        """
        Log experiment arguments/configuration.

        Args:
            args: Argparse namespace, dict, or any object with __dict__
        """
        self.state.log_arguments(args)

    def log_metadata(self, metadata: Dict[str, Any]):
        """
        Log or update experiment metadata.

        Args:
            metadata: Dictionary of metadata to log
        """
        self.state.log_metadata(metadata)

    # --- Progress Tracking ---

    def progress(
        self,
        iterable: Iterable[T],
        description: str = "",
        total: Optional[int] = None,
        **kwargs,
    ) -> Iterable[T]:
        """
        Track progress through an iterable (like tqdm).
        Automatically creates and completes a step.

        Args:
            iterable: The iterable to track
            description: Description of this step
            total: Total items (auto-detected if possible)
            **kwargs: Additional arguments passed to progress bar

        Yields:
            Items from iterable

        Example:
            for item in dash.progress(data, description="Processing"):
                process(item)
        """
        # Infer total if not provided
        if total is None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                pass

        # Create step in dashboard state
        step_id = self.state.create_step(description, total=total)

        try:
            # Don't show progress bar visually - it's already in the dashboard Steps panel
            # Just iterate and update the dashboard
            count = 0
            for item in iterable:
                yield item
                count += 1
                # Update dashboard step
                self.state.update_step(step_id, advance=1)

            # Mark complete
            self.state.complete_step(step_id)

        except Exception as e:
            # Mark failed
            self.state.fail_step(step_id, str(e))
            raise

    # --- Manual Step Control ---

    def start_step(self, description: str, total: Optional[int] = None) -> str:
        """
        Manually start a step.

        Args:
            description: Step description
            total: Total number of items/iterations (optional)

        Returns:
            step_id: ID to use for updating/completing the step

        Example:
            step_id = dash.start_step("Loading data", total=1000)
            for i in range(1000):
                load_item(i)
                dash.update_step(step_id)
            dash.stop_step(step_id)
        """
        return self.state.create_step(description, total=total)

    def update_step(self, step_id: str, advance: int = 1):
        """
        Update step progress.

        Args:
            step_id: Step ID returned from start_step()
            advance: Amount to advance (default: 1)
        """
        self.state.update_step(step_id, advance=advance)

    def stop_step(self, step_id: str):
        """
        Mark a step as completed.

        Args:
            step_id: Step ID to complete
        """
        self.state.complete_step(step_id)

    def fail_step(self, step_id: str, error: Optional[str] = None):
        """
        Mark a step as failed.

        Args:
            step_id: Step ID to mark as failed
            error: Optional error message
        """
        self.state.fail_step(step_id, error)

    def step(self, description: str, total: Optional[int] = None) -> StepContext:
        """
        Context manager for step control.

        Args:
            description: Step description
            total: Total iterations (optional)

        Returns:
            Context manager that auto-completes on exit

        Example:
            with dash.step("Training", total=1000) as step:
                for i in range(1000):
                    train()
                    step.update()
        """
        step_id = self.start_step(description, total)
        return StepContext(self, step_id)

    # --- Variable Tracking ---

    def track_variable(self, name: str, value: Any, step: Optional[int] = None):
        """
        Track a variable value over time.

        Args:
            name: Variable name
            value: Variable value
            step: Optional step/epoch number

        Example:
            for epoch in range(10):
                loss = train()
                dash.track_variable("loss", loss, step=epoch)
        """
        self.state.track_variable(name, value, step)

    def get_variable_history(self, name: str):
        """
        Get the history of a tracked variable.

        Args:
            name: Variable name

        Returns:
            VariableHistory object or None
        """
        return self.state.get_variable_history(name)

    def plot_variable(self, name: str, **kwargs):
        """
        Plot a tracked variable (requires matplotlib).

        Args:
            name: Variable name to plot
            **kwargs: Additional arguments for matplotlib
        """
        history = self.get_variable_history(name)
        if not history:
            print(f"Variable '{name}' not found")
            return

        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.figure(figsize=(10, 6))
            plt.plot(history.steps, history.values, **kwargs)
            plt.xlabel("Step")
            plt.ylabel(name)
            plt.title(f"{name} over time")
            plt.grid(True, alpha=0.3)
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Install with: pip install matplotlib")

    # --- Output Capture ---

    def print(self, *args, **kwargs):
        """
        Print to both terminal and dashboard.
        Use this instead of regular print() to capture output in dashboard.

        Example:
            dash.print("Training started")
            dash.print(f"Epoch {epoch}: loss = {loss:.4f}")
        """
        import builtins

        # Print normally
        builtins.print(*args, **kwargs)
        # Capture to dashboard
        line = " ".join(str(arg) for arg in args)
        self.state.add_output_line(line)

    def capture_output(self) -> OutputCapture:
        """
        Context manager to capture stdout/stderr.

        Example:
            with dash.capture_output():
                print("This will be captured")
                model.summary()
        """

        def callback(line: str):
            self.state.add_output_line(line)

        return OutputCapture(callback=callback)

    # --- Persistence ---

    def save(self, filepath: Optional[str] = None):
        """
        Save dashboard state to JSON file.

        Args:
            filepath: Path to save to (uses save_dir if not specified)
        """
        if filepath is None:
            if self.save_dir is None:
                raise ValueError("No save_dir specified")
            self.save_dir.mkdir(parents=True, exist_ok=True)
            filepath = str(self.save_dir / "dashboard_state.json")

        self.state.to_json(filepath)

    @classmethod
    def load(cls, filepath: str) -> "Dashboard":
        """
        Load dashboard state from JSON file.

        Args:
            filepath: Path to load from

        Returns:
            Dashboard instance with loaded state
        """
        state = DashboardState.from_json(filepath)
        dash = cls(title=state.title, description=state.description, mode="none")
        dash.state = state
        return dash

    # --- Web Interface ---

    def serve(self, port: int = 8080, backend: str = "streamlit"):
        """
        Launch web interface (non-blocking).

        Args:
            port: Port to serve on
            backend: "streamlit" or "gradio"
        """
        if backend == "streamlit":
            try:
                self._serve_streamlit(port)
            except ImportError:
                print("Streamlit not installed. Install with: pip install streamlit")
        else:
            print(f"Backend '{backend}' not yet implemented")

    def _serve_streamlit(self, port: int):
        """Launch Streamlit interface."""
        # Will implement in streamlit_backend.py
        print("Streamlit interface not yet implemented")
        print(f"Would serve on port {port}")

    # --- Context Manager ---

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Stop output capture first
        if self._capture_output_enabled:
            self._stop_output_capture()

        if self.backend:
            self.backend.stop()

        if self.auto_save:
            self.save()

        return False

    # --- Summary ---

    def print_summary(self):
        """Print final summary."""
        if self.backend and hasattr(self.backend, "print_summary"):
            self.backend.print_summary()
        else:
            print(f"\n{'=' * 60}")
            print(f"Experiment: {self.state.title}")
            print(
                f"Steps completed: {len([s for s in self.state.steps.values() if s.status == 'completed'])}/{len(self.state.steps)}"
            )
            print(f"Variables tracked: {len(self.state.variables)}")
            print(f"{'=' * 60}\n")
