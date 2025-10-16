"""
None Backend - Passthrough to standard output

This backend provides no dashboard UI, just standard output.
All Dashboard methods work but use standard rich functions instead of Live display.
"""

from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich import print as rich_print

from .state import DashboardState


class NoneBackend:
    """
    Backend that provides no dashboard - just standard output.
    
    All Dashboard methods work, but they use standard rich functions:
    - dash.print() -> rich.print()
    - dash.progress() -> rich.progress.track()
    - etc.
    
    This is useful for:
    - Quick scripts where you don't need the dashboard
    - When you want clean stdout output
    - Testing without UI overhead
    """

    def __init__(
        self,
        state: DashboardState,
        auto_refresh: bool = True,
        refresh_rate: float = 0.1,
    ):
        """
        Initialize None backend.

        Args:
            state: Dashboard state (still tracked but not displayed)
            auto_refresh: Ignored for this backend
            refresh_rate: Ignored for this backend
        """
        self.state = state
        self.console = Console()
        self._active_progress: Optional[Progress] = None

    def start(self):
        """Start backend (no-op for none backend)."""
        pass

    def stop(self):
        """Stop backend (no-op for none backend)."""
        pass

    def create_progress(self, description: str, total: Optional[int] = None):
        """
        Create a progress bar using standard rich progress.
        
        Returns a rich Progress context manager.
        """
        if self._active_progress is None:
            self._active_progress = Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=self.console,
            )
            
        return self._active_progress

    def print(self, *args, **kwargs):
        """Print using rich.print()."""
        rich_print(*args, **kwargs)

    def print_summary(self):
        """Print final summary using rich."""
        from rich.panel import Panel
        from rich.table import Table
        
        # Count completed steps
        completed = len([s for s in self.state.steps.values() if s.status == "completed"])
        total_steps = len(self.state.steps)
        
        # Create summary table
        table = Table(show_header=False, box=None)
        table.add_row("Experiment", self.state.title)
        table.add_row("Steps completed", f"{completed}/{total_steps}")
        table.add_row("Variables tracked", str(len(self.state.variables)))
        
        panel = Panel(table, title="Summary", border_style="green")
        self.console.print(panel)

    def __del__(self):
        """Cleanup."""
        if self._active_progress:
            try:
                self._active_progress.stop()
            except Exception:
                pass
