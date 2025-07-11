"""
Progress tracking utilities for interactive and batch environments.

This module provides adaptive progress tracking that automatically adjusts its
behavior based on the execution environment. It offers rich, interactive progress
bars for terminal sessions and clean, log-friendly progress updates for batch
jobs and non-interactive environments.

Key Features:
    - Automatic environment detection (interactive vs batch)
    - Rich progress bars for interactive terminals
    - Clean text-based logging for batch jobs
    - Support for both time-based and item-count-based update intervals
    - Compatible with common batch systems (SLURM, PBS, SGE, etc.)
    - tqdm-style interface for easy integration

Main Components:
    - LoggingProgress: Text-based progress tracker for batch environments
    - _NoOpProgress: Null progress tracker for disabled progress
    - progress(): Main function providing tqdm-style progress tracking
    - get_progress_bar(): Factory function for creating appropriate progress trackers

The module automatically detects execution context and chooses the most appropriate
progress tracking method:
    - Interactive terminals: Rich progress bars with visual elements
    - Batch jobs (sbatch, etc.): Text-based logging with timestamps
    - Disabled mode: No-op tracker that doesn't display anything

Environment Detection:
    The module detects batch environments by checking for:
    - Batch system environment variables (SLURM_JOB_ID, PBS_JOBID, etc.)
    - Non-interactive terminals (!sys.stdout.isatty())
    - "dumb" terminal types
    - Output redirection

Example Usage:
    >>> # Simple iteration with progress
    >>> from easyroutine.console import progress
    >>> for item in progress(my_list, description="Processing"):
    ...     process(item)
    
    >>> # Manual progress tracking
    >>> with get_progress_bar() as pbar:
    ...     task = pbar.add_task("Training", total=epochs)
    ...     for epoch in range(epochs):
    ...         train_epoch()
    ...         pbar.update(task)
"""

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
    A progress tracker designed for non-interactive batch environments.
    
    This class provides progress tracking functionality specifically optimized for
    batch job environments (like sbatch, PBS, SGE) where fancy progress bars won't
    display properly. Instead of using visual progress bars, it outputs clean,
    timestamped progress updates to stdout/stderr that are suitable for job logs.
    
    The progress tracker supports both time-based and item-count-based update
    intervals, allowing flexible control over update frequency to balance
    informativeness with log verbosity.
    
    Attributes:
        tasks (dict): Internal storage for tracking multiple tasks and their progress.
        log_interval (int): Minimum time interval between progress updates in seconds.
        update_frequency (int): Item count interval for progress updates (0 = disabled).
    
    Example:
        >>> with LoggingProgress(log_interval=10, update_frequency=100) as progress:
        ...     task_id = progress.add_task("Processing files", total=1000)
        ...     for i in range(1000):
        ...         # do work
        ...         progress.update(task_id)
        
        >>> # Or use the track method for iterables
        >>> with LoggingProgress() as progress:
        ...     for item in progress.track(data_list, description="Processing"):
        ...         # process item
    """

    def __init__(self, log_interval: int = 5, update_frequency: int = 0):
        """
        Initialize the logging progress tracker with customizable update intervals.

        Args:
            log_interval (int, optional): Minimum time interval between progress 
                updates in seconds. Progress will be logged at most once per this
                interval, regardless of how frequently update() is called.
                Defaults to 5 seconds.
            update_frequency (int, optional): Alternative update trigger based on
                item count. If set to a positive value, progress will be logged
                every N items processed, in addition to time-based updates.
                Set to 0 to disable item-count-based updates. Defaults to 0.
                
        Note:
            Both update triggers work together - progress is logged when either
            the time interval has elapsed OR the item count threshold is reached,
            whichever comes first.
        """
        self.tasks = {}
        self.log_interval = log_interval
        self.update_frequency = update_frequency

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def add_task(self, description: str, total: int = None, **kwargs):
        """
        Add a new task to track with the progress tracker.
        
        Args:
            description (str): Human-readable description of the task to be
                displayed in progress updates. Should be concise but descriptive.
            total (int, optional): Total number of items/steps expected for this
                task. If provided, enables percentage calculation and ETA estimates.
                If None, only item counts and elapsed time will be displayed.
                Defaults to None.
            **kwargs: Additional keyword arguments (currently unused but accepted
                for compatibility with other progress bar interfaces).
        
        Returns:
            int: Unique task identifier that should be used with update() calls
                to track progress for this specific task.
        
        Side Effects:
            - Creates a new task entry in the internal tasks dictionary
            - Prints an initial status message indicating the task has started
            - Records the task start time for elapsed time calculations
        
        Example:
            >>> progress = LoggingProgress()
            >>> task_id = progress.add_task("Training model", total=100)
            # Output: [PROGRESS] Starting: Training model (Total: 100)
        """
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
        """
        Update the progress of a tracked task.
        
        This method increments the completion counter for a task and conditionally
        logs progress updates based on the configured time and item count intervals.
        Progress is logged when either sufficient time has elapsed since the last
        update OR when enough items have been processed since the last log.
        
        Args:
            task_id (int): The task identifier returned by add_task().
            advance (int, optional): Number of items/steps to increment the
                progress counter by. Defaults to 1.
            **kwargs: Additional keyword arguments (currently unused but accepted
                for compatibility with other progress bar interfaces).
        
        Returns:
            None: This function has no return value.
        
        Side Effects:
            - Increments the task's completion counter
            - May print a progress update message if update conditions are met
            - Updates internal timing information for the task
        
        Example:
            >>> task_id = progress.add_task("Processing", total=100)
            >>> for i in range(100):
            ...     # do work
            ...     progress.update(task_id)  # Increment by 1
            >>> # Or increment by multiple items at once
            >>> progress.update(task_id, advance=5)  # Increment by 5
        
        Note:
            If the specified task_id doesn't exist, the call is silently ignored.
        """
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
        """
        Track progress through an iterable with automatic progress updates.
        
        This method wraps an iterable and automatically updates progress as items
        are yielded. It's a convenience method that combines add_task() and update()
        calls for simple iteration tracking.
        
        Args:
            iterable (Iterable[T]): The iterable to track progress through.
                Can be any iterable object (list, generator, etc.).
            total (int, optional): Total number of items in the iterable.
                If None, attempts to determine the length using len().
                If the length cannot be determined, only item counts and
                elapsed time will be displayed. Defaults to None.
            description (str, optional): Description for the task to be displayed
                in progress updates. Defaults to empty string.
        
        Yields:
            T: Items from the original iterable, yielded one at a time.
        
        Returns:
            Iterable[T]: A generator that yields items from the original iterable
                while tracking progress.
        
        Side Effects:
            - Creates a new task for tracking this iterable
            - Prints progress updates as items are processed
            - Prints a completion message when the iterable is exhausted
        
        Example:
            >>> data = list(range(1000))
            >>> with LoggingProgress() as progress:
            ...     for item in progress.track(data, description="Processing data"):
            ...         # process item
            ...         result = expensive_operation(item)
        
        Note:
            This method creates a new task internally and manages all progress
            updates automatically. No manual update() calls are needed.
        """
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
    """
    Format a time duration in seconds into a human-readable string.
    
    This utility function converts a floating-point duration in seconds
    into a more readable format using appropriate time units (seconds,
    minutes, or hours) based on the magnitude of the duration.
    
    Args:
        seconds (float): Duration in seconds to format. Can be fractional.
    
    Returns:
        str: Formatted time string with one decimal place and appropriate unit:
            - Durations < 60 seconds: "X.Xs" (e.g., "45.2s")
            - Durations < 3600 seconds: "X.Xm" (e.g., "12.5m") 
            - Durations >= 3600 seconds: "X.Xh" (e.g., "2.3h")
    
    Example:
        >>> format_time(45.7)
        '45.7s'
        >>> format_time(125.3)
        '2.1m'
        >>> format_time(7890.5)
        '2.2h'
        >>> format_time(0.123)
        '0.1s'
    
    Note:
        The function rounds to one decimal place for readability. For very
        large durations, the hours format provides a compact representation.
    """
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
    A tqdm-style progress tracker that automatically adapts to the environment.

    This function provides a drop-in replacement for tqdm that automatically
    detects the execution environment and displays appropriate progress feedback:
    - Rich progress bars in interactive terminals
    - Clean text-based logging in batch jobs
    - No output when disabled

    The function wraps an iterable and yields items while tracking progress,
    making it easy to add progress tracking to existing loops with minimal
    code changes.

    Args:
        iterable: The iterable to wrap with progress tracking. Can be any
            iterable object (list, tuple, generator, etc.).
        description (str, optional): Description to display with the progress.
            Shows what operation is being performed. Defaults to empty string.
        desc (str, optional): Alternative parameter name for description,
            provided for tqdm compatibility. If provided, takes precedence
            over description parameter. Defaults to None.
        total (int, optional): Expected total number of items in the iterable.
            If None, attempts to determine automatically using len(iterable).
            Required for accurate percentage and ETA calculations.
            Defaults to None.
        disable (bool, optional): If True, completely disables progress tracking
            and returns the original iterable unchanged. Useful for conditional
            progress display. Defaults to False.
        force_batch_mode (bool, optional): If True, forces text-based progress
            tracking even in interactive environments. Useful for testing or
            when rich output is not desired. Defaults to False.
        log_interval (int, optional): In batch mode, minimum seconds between
            progress updates. Prevents excessive logging while ensuring regular
            updates. Defaults to 1 second.
        update_frequency (int, optional): In batch mode, number of items to
            process between progress updates. Works in addition to time-based
            updates. Set to 0 to disable item-count-based updates.
            Defaults to 0.

    Yields:
        Items from the original iterable, one at a time, with progress tracking.

    Returns:
        Generator: A generator that yields items from the iterable while
            tracking and displaying progress.

    Examples:
        >>> # Basic usage - simple iteration with progress
        >>> for item in progress(range(1000), description="Processing"):
        ...     process_item(item)

        >>> # With custom total for generators
        >>> data_gen = generate_data()
        >>> for item in progress(data_gen, total=expected_count, desc="Loading"):
        ...     handle_item(item)

        >>> # Conditional progress (disable in quiet mode)
        >>> for item in progress(data, disable=quiet_mode):
        ...     process(item)

        >>> # Force batch mode for consistent output
        >>> for item in progress(items, force_batch_mode=True, log_interval=5):
        ...     slow_operation(item)

    Environment Behavior:
        - Interactive terminals: Displays rich progress bar with percentage,
          visual bar, item counts, and time estimates
        - Batch jobs: Outputs timestamped text updates at specified intervals
        - Non-interactive: Falls back to text-based logging
        - Disabled: Returns original iterable with no progress tracking

    Note:
        The function automatically handles cases where len(iterable) is not
        available (e.g., generators) by falling back to count-only progress
        without percentage or ETA calculations.
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
