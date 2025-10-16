"""
Streamlit Backend - Web-based dashboard

This backend provides a web UI using Streamlit.
Since we're not using Rich Live display, we can capture stdout/stderr properly!
"""

import threading
import time
import sys
import json
import atexit
from typing import Optional
from pathlib import Path

from .state import DashboardState


class StreamlitBackend:
    """
    Web-based dashboard backend using Streamlit.
    
    Features:
    - Real-time updates in browser
    - Clean web UI with metrics and charts
    - Proper stdout/stderr capture (no conflicts!)
    - Shareable URL for remote viewing
    
    Implementation:
    - Uses file-based state sharing between main process and Streamlit
    - Main process periodically saves state to temp JSON file
    - Streamlit app reads JSON file and displays it
    - Auto-refresh in Streamlit for real-time updates
    """

    def __init__(
        self,
        state: DashboardState,
        auto_refresh: bool = True,
        refresh_rate: float = 1.0,
        port: int = 8501,
        open_browser: bool = True,
    ):
        """
        Initialize Streamlit backend.

        Args:
            state: Dashboard state to display
            auto_refresh: Whether to auto-refresh display
            refresh_rate: Refresh rate in seconds
            port: Port to run Streamlit on
            open_browser: Whether to open browser automatically
        """
        self.state = state
        self.auto_refresh = auto_refresh
        self.refresh_rate = refresh_rate
        self.port = port
        self.open_browser = open_browser
        
        self._streamlit_thread: Optional[threading.Thread] = None
        self._app_file: Optional[Path] = None
        self._state_file: Optional[Path] = None
        self._running = False
        self._update_thread: Optional[threading.Thread] = None
        
        # Register cleanup
        atexit.register(self.stop)

    def start(self):
        """Start Streamlit server in background thread."""
        try:
            import streamlit  # noqa
        except ImportError:
            raise ImportError(
                "Streamlit is required for web dashboard mode. "
                "Install it with: poetry add streamlit"
            )
        
        # Create state file for sharing
        self._create_state_file()
        
        # Create temporary Streamlit app file
        self._create_app_file()
        
        # Start state update thread
        self._running = True
        self._update_thread = threading.Thread(
            target=self._update_state_loop,
            daemon=True,
        )
        self._update_thread.start()
        
        # Start Streamlit in background thread
        self._streamlit_thread = threading.Thread(
            target=self._run_streamlit,
            daemon=True,
        )
        self._streamlit_thread.start()
        
        # Wait a bit for server to start
        time.sleep(3)
        
        print(f"\n🌐 Dashboard running at http://localhost:{self.port}")
        print("   Press Ctrl+C to stop\n")

    def stop(self):
        """Stop Streamlit server and cleanup."""
        self._running = False
        
        # Cleanup files
        if self._app_file and self._app_file.exists():
            try:
                self._app_file.unlink()
            except Exception:
                pass
                
        if self._state_file and self._state_file.exists():
            try:
                self._state_file.unlink()
            except Exception:
                pass

    def _create_state_file(self):
        """Create temporary state file for sharing with Streamlit."""
        import tempfile
        
        temp_dir = Path(tempfile.gettempdir())
        self._state_file = temp_dir / f"easyroutine_state_{id(self)}.json"
        
        # Initial save
        self._save_state()

    def _save_state(self):
        """Save current state to JSON file."""
        if self._state_file:
            try:
                state_dict = self.state.to_dict()
                self._state_file.write_text(json.dumps(state_dict, indent=2))
            except Exception as e:
                # Don't crash on save errors
                pass

    def _update_state_loop(self):
        """Continuously update state file for Streamlit."""
        while self._running:
            self._save_state()
            time.sleep(self.refresh_rate)

    def _create_app_file(self):
        """Create temporary Streamlit app file."""
        import tempfile
        
        # Create app in temp directory
        temp_dir = Path(tempfile.gettempdir())
        self._app_file = temp_dir / f"easyroutine_dashboard_{id(self)}.py"
        
        # Generate Streamlit app code
        app_code = self._generate_app_code()
        self._app_file.write_text(app_code)

    def _generate_app_code(self) -> str:
        """Generate Streamlit app code that reads from state file."""
        state_file = str(self._state_file)
        
        return f'''
import streamlit as st
import time
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Read state from JSON file
state_file = Path("{state_file}")

try:
    if state_file.exists():
        state = json.loads(state_file.read_text())
    else:
        state = {{"title": "Dashboard", "steps": {{}}, "variables": {{}}, "output_lines": []}}
except Exception as e:
    state = {{"title": "Dashboard (error loading)", "steps": {{}}, "variables": {{}}, "output_lines": []}}
    st.error(f"Error loading state: {{e}}")

# Title
st.title(state.get("title", "Experiment Dashboard"))

# Header with arguments and metadata
args = state.get("arguments", {{}})
meta = state.get("metadata", {{}})

if args or meta:
    cols = st.columns(2)
    
    with cols[0]:
        if args:
            st.subheader("⚙️ Arguments")
            for key, value in args.items():
                st.text(f"{{key}}: {{value}}")
    
    with cols[1]:
        if meta:
            st.subheader("📋 Configuration")
            for key, value in meta.items():
                st.text(f"{{key}}: {{value}}")
    
    st.divider()

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("� Progress")
    
    steps = state.get("steps", {{}})
    if steps:
        # Show active and recent steps
        for step_id, step in steps.items():
            # Only show top-level steps (no parent)
            if step.get("parent_id") is None:
                status = step.get("status", "active")
                current = step.get("current", 0)
                total = step.get("total", 0)
                description = step.get("description", "Step")
                
                # Calculate progress
                if total > 0:
                    progress = current / total
                else:
                    progress = 0.0
                
                # Status emoji
                status_emoji = "🔄" if status == "active" else ("✅" if status == "completed" else "❌")
                
                # Show progress bar
                st.progress(progress, text=f"{{status_emoji}} {{description}}: {{current}}/{{total}}")
    else:
        st.info("No steps yet")

with col2:
    st.subheader("📊 Variables")
    
    variables = state.get("variables", {{}})
    if variables:
        for var_name, history in variables.items():
            if history and len(history) > 0:
                # Get latest value
                latest = history[-1]
                value = latest.get("value", 0)
                
                # Format value
                if isinstance(value, float):
                    display_value = f"{{value:.4f}}"
                else:
                    display_value = str(value)
                
                # Calculate delta if we have history
                delta = None
                if len(history) > 1:
                    prev_value = history[-2].get("value", value)
                    if isinstance(value, (int, float)) and isinstance(prev_value, (int, float)):
                        delta = value - prev_value
                
                st.metric(
                    label=var_name,
                    value=display_value,
                    delta=f"{{delta:.4f}}" if delta is not None and isinstance(delta, float) else delta
                )
    else:
        st.info("No variables tracked")

# Terminal output
st.divider()
st.subheader("💻 Terminal Output")

output_lines = state.get("output_lines", [])
if output_lines:
    # Show last 100 lines
    output_text = "\\n".join(output_lines[-100:])
    st.code(output_text, language="text", line_numbers=False)
else:
    st.info("No output yet")

# Footer with stats
st.divider()
completed_steps = len([s for s in steps.values() if s.get("status") == "completed"])
total_steps = len(steps)
st.caption(f"Steps: {{completed_steps}}/{{total_steps}} completed | Variables: {{len(variables)}} tracked | Output lines: {{len(output_lines)}}")

# Auto-refresh
if {str(self.auto_refresh).lower()}:
    time.sleep({self.refresh_rate})
    st.rerun()
'''

    def _run_streamlit(self):
        """Run Streamlit server."""
        import subprocess
        
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(self._app_file),
            "--server.port",
            str(self.port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        
        # Add browser opening flag
        if not self.open_browser:
            cmd.extend(["--server.runOnSave", "false"])
        
        try:
            # Redirect streamlit output to suppress noise
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL if not self.open_browser else None,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception as e:
            print(f"Error running Streamlit: {e}")

    def print(self, *args, **kwargs):
        """Print to stdout (will be captured and added to state)."""
        # Print normally - will be captured by OutputCapture if enabled
        print(*args, **kwargs)

    def print_summary(self):
        """Print summary (just use regular print for now)."""
        completed = len([s for s in self.state.steps.values() if s.status == "completed"])
        total_steps = len(self.state.steps)
        
        print("\n" + "=" * 60)
        print(f"Experiment: {self.state.title}")
        print(f"Steps completed: {completed}/{total_steps}")
        print(f"Variables tracked: {len(self.state.variables)}")
        print("=" * 60 + "\n")
