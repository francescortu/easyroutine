from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from typing import Optional, TypeVar, Iterable
import sys
import time
import os
import logging

T = TypeVar("T")

# Configure logging for batch mode
# Use a simpler format for cleaner output in job logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("progress")


class LoggingProgress:
    """
    A progress tracker designed for batch environments (sbatch, etc.)
    that outputs clean, consistent progress updates to stdout/stderr.
    """

    def __init__(self, log_interval: int = 5, update_frequency: int = 0):
        """
        Initialize the logging progress tracker.

        Args:
            log_interval: How often to log progress updates (in seconds)
            update_frequency: Alternative to log_interval - update every N items (0 = use log_interval only)
        """
        self.tasks = {}
        self.log_interval = log_interval
        self.update_frequency = update_frequency

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def add_task(self, description: str, total: int = None, **kwargs):
        """Add a task to track."""
        task_id = len(self.tasks)
        self.tasks[task_id] = {
            "description": description,
            "total": total,
            "completed": 0,
            "start_time": time.time(),
            "last_log_time": 0,  # 0 ensures first update is always logged
            "last_item_logged": 0,
        }
        if description:
            print(f"\n[PROGRESS] Starting: {description} (Total: {total or 'unknown'})")
        return task_id

    def update(self, task_id, advance=1, **kwargs):
        """Update task progress."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task["completed"] += advance
        current_time = time.time()

        # Determine if we should log based on either time interval or item count
        should_log = False

        # Check time interval
        if current_time - task["last_log_time"] >= self.log_interval:
            should_log = True

        # Check item count interval (if specified)
        if self.update_frequency > 0:
            items_since_log = task["completed"] - task["last_item_logged"]
            if items_since_log >= self.update_frequency:
                should_log = True

        if should_log:
            elapsed = current_time - task["start_time"]
            if task["total"]:
                percentage = (task["completed"] / task["total"]) * 100
                remaining = (
                    (elapsed / task["completed"]) * (task["total"] - task["completed"])
                    if task["completed"] > 0
                    else 0
                )
                print(
                    f"[PROGRESS] {task['description']}: {task['completed']}/{task['total']} "
                    f"({percentage:.1f}%) - Elapsed: {format_time(elapsed)}, "
                    f"Remaining: {format_time(remaining)}"
                )
            else:
                print(
                    f"[PROGRESS] {task['description']}: {task['completed']} items - "
                    f"Elapsed: {format_time(elapsed)}"
                )
            task["last_log_time"] = current_time
            task["last_item_logged"] = task["completed"]

    def track(
        self, iterable: Iterable[T], total: Optional[int] = None, description: str = ""
    ) -> Iterable[T]:
        """Track progress through an iterable."""
        if total is None:
            try:
                total = len(iterable)
            except (TypeError, AttributeError):
                pass

        task_id = self.add_task(description, total=total)
        count = 0

        for item in iterable:
            count += 1
            yield item
            self.update(task_id)

        # Final update to show 100% completion
        if total:
            elapsed = time.time() - self.tasks[task_id]["start_time"]
            print(
                f"[PROGRESS] Complete: {description} - {count}/{total} items in {format_time(elapsed)}"
            )


def format_time(seconds: float) -> str:
    """Format seconds into a readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class _NoOpProgress:
    """A progress bar that does nothing, for use when progress is disabled."""

    def track(self, iterable, *args, **kwargs):
        # Just yield from the iterable without displaying a progress bar.
        yield from iterable

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def add_task(self, *args, **kwargs):
        """Returns a dummy task ID."""
        return 0

    def update(self, *args, **kwargs):
        """A no-op update."""
        pass


def is_non_interactive_batch() -> bool:
    """
    Detect if running in a non-interactive batch job (like sbatch) where
    fancy progress bars won't display properly.

    Returns:
        bool: True if in a non-interactive batch job, False otherwise
    """
    # Definite indicators of batch execution
    batch_env_vars = ["SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID"]
    for var in batch_env_vars:
        if var in os.environ:
            # Running in a batch system, now check if it's non-interactive
            if not sys.stdout.isatty():
                return True

    # If specific batch execution indicators weren't found,
    # use more general checks for non-interactive environment

    # Check if TERM is set to "dumb" (common in batch environments)
    if os.environ.get("TERM", "") == "dumb":
        return True

    # Check for output redirection
    if not sys.stdout.isatty():
        # Special case: when using "srun --pty" on Slurm, we might be in
        # a pseudo-terminal that can handle rich output
        if "SLURM_PTY_PORT" in os.environ:
            return False
        return True

    return False


def get_progress_bar(
    disable: bool = False,
    force_batch_mode: bool = False,
    log_interval: int = 1,
    update_frequency: int = 0,
):
    """
    Returns a progress tracker appropriate for the current environment.

    In interactive environments (including interactive Slurm sessions),
    this will use a rich progress bar. In non-interactive batch jobs
    (like sbatch), it will use simpler text-based output.

    Args:
        disable: If True, returns a No-Op progress object that does nothing.
        force_batch_mode: If True, use the text-based progress tracker
                         even in interactive environments.
        log_interval: How often (in seconds) to log progress in batch mode.
        update_frequency: In batch mode, update progress after this many items
                        (0 means use only time-based updates)

    Returns:
        A progress tracker compatible with the current environment.
    """
    if disable:
        return _NoOpProgress()

    # Check if we're in a non-interactive batch environment (e.g., sbatch)
    is_batch = force_batch_mode or is_non_interactive_batch()

    # For batch jobs, use the simplified logging-based progress
    if is_batch:
        # In sbatch, use more frequent updates by default, including per-item updates
        if "SLURM_JOB_ID" in os.environ and update_frequency == 0:
            # By default in sbatch jobs, use both time-based and every 5 items
            update_frequency = 1

        return LoggingProgress(
            log_interval=log_interval, update_frequency=update_frequency
        )

    # Use rich progress for interactive environments
    return Progress(
        TextColumn("[progress.description]{task.description}:"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


def progress(
    iterable,
    description: str = "",
    desc: Optional[str] = None,
    total: Optional[int] = None,
    disable: bool = False,
    force_batch_mode: bool = False,
    log_interval: int = 1,
    update_frequency: int = 0,
):
    """
    A tqdm-style progress bar that can be wrapped around an iterable.

    This function automatically adapts to the environment:
    - In interactive sessions (including interactive Slurm jobs), it shows a rich progress bar
    - In non-interactive batch jobs (like sbatch), it uses simple text-based progress tracking

    e.g. `for i in progress(range(10)):`

    Args:
        iterable: The iterable to wrap with a progress bar.
        description (str): Description to display.
        total (int, optional): The total number of items. If None, it's inferred from len(iterable).
        disable (bool): If True, the progress bar is disabled completely.
        force_batch_mode (bool): If True, use text-based progress tracking even in interactive environments.
        log_interval (int): In batch mode, how often (in seconds) to log progress updates.
        update_frequency (int): In batch mode, update progress after processing this many items.
                               Set to 0 to use only time-based updates.
    """
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            pass
    if desc is not None:
        description = desc

    with get_progress_bar(
        disable=disable,
        force_batch_mode=force_batch_mode,
        log_interval=log_interval,
        update_frequency=update_frequency,
    ) as progress:
        yield from progress.track(iterable, total=total, description=description)
