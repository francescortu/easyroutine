"""
Data models for dashboard state management.

This module contains simple dataclasses to track experiment state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import threading
import json


@dataclass
class StepState:
    """Represents a single step/stage in the experiment."""

    id: str
    description: str
    total: Optional[int] = None
    completed: int = 0
    status: str = "running"  # "running", "completed", "failed"
    parent_id: Optional[str] = None  # For nested steps
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> Optional[float]:
        """Progress as a percentage (0-100) or None if total is unknown."""
        if self.total and self.total > 0:
            return (self.completed / self.total) * 100
        return None

    @property
    def elapsed_time(self) -> float:
        """Elapsed time in seconds."""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "description": self.description,
            "total": self.total,
            "completed": self.completed,
            "status": self.status,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metadata": self.metadata,
            "progress": self.progress,
            "elapsed_time": self.elapsed_time,
        }


@dataclass
class VariableHistory:
    """Tracks a variable over time."""

    name: str
    values: List[Any] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)

    def add(self, value: Any, step: Optional[int] = None):
        """Add a value to the history."""
        self.values.append(value)
        self.steps.append(step if step is not None else len(self.values) - 1)
        self.timestamps.append(datetime.now())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "values": self.values,
            "steps": self.steps,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
        }


class DashboardState:
    """
    Central state management for the dashboard.
    Thread-safe for concurrent access.
    """

    def __init__(self, title: str = "Experiment", description: str = ""):
        self.title = title
        self.description = description
        self.start_time = datetime.now()

        # Experiment metadata
        self.arguments: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

        # Step tracking
        self.steps: Dict[str, StepState] = {}
        self.step_order: List[str] = []  # Maintain insertion order
        self._step_counter = 0
        self._current_step_stack: List[str] = []  # Track nesting

        # Variable tracking
        self.variables: Dict[str, VariableHistory] = {}

        # Output capture
        self.output_lines: List[str] = []
        self.max_output_lines: int = 1000  # Limit to prevent memory issues

        # Thread safety
        self._lock = threading.RLock()

        # State change callbacks for real-time updates
        self._callbacks: List[Callable] = []

    def add_callback(self, callback: Callable):
        """Add a callback function that gets called on state changes."""
        with self._lock:
            self._callbacks.append(callback)

    def _notify_callbacks(self):
        """Notify all registered callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Callback error: {e}")

    # --- Arguments & Metadata ---

    def log_arguments(self, args: Any):
        """Log experiment arguments."""
        with self._lock:
            if hasattr(args, "__dict__"):
                self.arguments = vars(args)
            elif isinstance(args, dict):
                self.arguments = args
            else:
                self.arguments = {"args": str(args)}
            self._notify_callbacks()

    def log_metadata(self, metadata: Dict[str, Any]):
        """Log or update experiment metadata."""
        with self._lock:
            self.metadata.update(metadata)
            self._notify_callbacks()

    # --- Step Management ---

    def create_step(
        self,
        description: str,
        total: Optional[int] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Create a new step and return its ID."""
        with self._lock:
            self._step_counter += 1
            step_id = f"step_{self._step_counter}"

            # Auto-detect parent from stack if not specified
            if parent_id is None and self._current_step_stack:
                parent_id = self._current_step_stack[-1]

            step = StepState(
                id=step_id, description=description, total=total, parent_id=parent_id
            )

            self.steps[step_id] = step
            self.step_order.append(step_id)
            self._current_step_stack.append(step_id)

            self._notify_callbacks()
            return step_id

    def update_step(self, step_id: str, advance: int = 1, **kwargs):
        """Update step progress."""
        with self._lock:
            if step_id in self.steps:
                step = self.steps[step_id]
                step.completed += advance

                # Update any additional fields
                for key, value in kwargs.items():
                    if hasattr(step, key):
                        setattr(step, key, value)

                # Auto-complete if total is reached
                if step.total and step.completed >= step.total:
                    self.complete_step(step_id)

                self._notify_callbacks()

    def complete_step(self, step_id: str):
        """Mark a step as completed."""
        with self._lock:
            if step_id in self.steps:
                step = self.steps[step_id]
                step.status = "completed"
                step.end_time = datetime.now()

                # Remove from stack if present
                if step_id in self._current_step_stack:
                    self._current_step_stack.remove(step_id)

                self._notify_callbacks()

    def fail_step(self, step_id: str, error: Optional[str] = None):
        """Mark a step as failed."""
        with self._lock:
            if step_id in self.steps:
                step = self.steps[step_id]
                step.status = "failed"
                step.end_time = datetime.now()
                if error:
                    step.metadata["error"] = error

                # Remove from stack if present
                if step_id in self._current_step_stack:
                    self._current_step_stack.remove(step_id)

                self._notify_callbacks()

    def get_step(self, step_id: str) -> Optional[StepState]:
        """Get a step by ID."""
        with self._lock:
            return self.steps.get(step_id)

    def get_active_steps(self) -> List[StepState]:
        """Get all currently running steps."""
        with self._lock:
            return [step for step in self.steps.values() if step.status == "running"]

    def get_step_hierarchy(self) -> List[StepState]:
        """Get steps in hierarchical order (respecting parent-child relationships)."""
        with self._lock:
            # Return in insertion order, UI can handle nesting based on parent_id
            return [
                self.steps[step_id]
                for step_id in self.step_order
                if step_id in self.steps
            ]

    # --- Variable Tracking ---

    def track_variable(self, name: str, value: Any, step: Optional[int] = None):
        """Track a variable value over time."""
        with self._lock:
            if name not in self.variables:
                self.variables[name] = VariableHistory(name=name)

            self.variables[name].add(value, step)
            self._notify_callbacks()

    def get_variable_history(self, name: str) -> Optional[VariableHistory]:
        """Get the history of a tracked variable."""
        with self._lock:
            return self.variables.get(name)

    # --- Output Capture ---

    def add_output_line(self, line: str):
        """Add a line to the output buffer."""
        with self._lock:
            self.output_lines.append(line)

            # Trim if exceeds max
            if len(self.output_lines) > self.max_output_lines:
                self.output_lines = self.output_lines[-self.max_output_lines :]

            self._notify_callbacks()

    def get_output(self, last_n: Optional[int] = None) -> List[str]:
        """Get output lines."""
        with self._lock:
            if last_n:
                return self.output_lines[-last_n:]
            return self.output_lines.copy()

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire state to dictionary for JSON serialization."""
        with self._lock:
            return {
                "title": self.title,
                "description": self.description,
                "start_time": self.start_time.isoformat(),
                "arguments": self.arguments,
                "metadata": self.metadata,
                "steps": {
                    step_id: step.to_dict() for step_id, step in self.steps.items()
                },
                "step_order": self.step_order,
                "variables": {
                    name: var.to_dict() for name, var in self.variables.items()
                },
                "output": self.output_lines,
            }

    def to_json(self, filepath: str):
        """Save state to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DashboardState":
        """Reconstruct state from dictionary."""
        state = cls(title=data.get("title", "Experiment"))
        state.description = data.get("description", "")
        state.arguments = data.get("arguments", {})
        state.metadata = data.get("metadata", {})
        state.output_lines = data.get("output", [])

        # Reconstruct steps
        for step_id, step_data in data.get("steps", {}).items():
            step = StepState(
                id=step_data["id"],
                description=step_data["description"],
                total=step_data.get("total"),
                completed=step_data.get("completed", 0),
                status=step_data.get("status", "running"),
                parent_id=step_data.get("parent_id"),
                start_time=datetime.fromisoformat(step_data["start_time"]),
                end_time=datetime.fromisoformat(step_data["end_time"])
                if step_data.get("end_time")
                else None,
                metadata=step_data.get("metadata", {}),
            )
            state.steps[step_id] = step

        state.step_order = data.get("step_order", [])

        # Reconstruct variables
        for var_name, var_data in data.get("variables", {}).items():
            var = VariableHistory(name=var_name)
            var.values = var_data["values"]
            var.steps = var_data["steps"]
            var.timestamps = [
                datetime.fromisoformat(ts) for ts in var_data["timestamps"]
            ]
            state.variables[var_name] = var

        return state

    @classmethod
    def from_json(cls, filepath: str) -> "DashboardState":
        """Load state from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
