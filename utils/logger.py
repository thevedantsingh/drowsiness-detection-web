"""
Session Logger

Records all drowsiness events, alerts, and session metadata to a JSON file.
This data can be visualised in the analytics dashboard.
"""

import json
import os
import datetime
import time


class SessionLogger:
    def __init__(self, filepath: str = "session_log.json"):
        self.filepath = filepath
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.events = []
        self.start_time = time.time()

    def log_event(self, event_type: str, data: dict):
        """
        Log a single event.

        Args:
            event_type : "yawn" | "alert" | "blink_burst" | "head_droop" | "custom"
            data       : dict of relevant metric values
        """
        entry = {
            "type": event_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            **data
        }
        self.events.append(entry)

    def save_summary(self, summary: dict):
        """Save full session log with summary to JSON file."""
        output = {
            "session_id": self.session_id,
            "start_time": datetime.datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.datetime.now().isoformat(),
            "summary": summary,
            "events": self.events,
        }

        # Load existing log file or create new
        all_sessions = []
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r") as f:
                    all_sessions = json.load(f)
            except (json.JSONDecodeError, IOError):
                all_sessions = []

        all_sessions.append(output)

        with open(self.filepath, "w") as f:
            json.dump(all_sessions, f, indent=2)

        print(f"[Logger] Session saved: {self.session_id} | "
              f"{len(self.events)} events | {self.filepath}")
        return output

    def get_recent_events(self, event_type: str = None, last_n: int = 10):
        """Retrieve recent events, optionally filtered by type."""
        events = self.events
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-last_n:]
