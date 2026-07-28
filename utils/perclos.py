"""
PERCLOS (PERcentage of eyelid CLOSure over the pupil over time)

The gold-standard drowsiness metric used in research and commercial systems.
PERCLOS = proportion of time in a rolling window that eyes are at least 80% closed.

Reference: Wierwille et al. (1994), NHTSA research.

A PERCLOS value > 0.35 (35%) indicates significant drowsiness.
"""

import time
from collections import deque


class PERCLOSTracker:
    """
    Sliding window PERCLOS calculator.

    Each frame, call update(eye_closed: bool).
    Call get_perclos() to get the current PERCLOS value [0.0 – 1.0].
    """

    def __init__(self, window_seconds: float = 60.0, fps_estimate: int = 30):
        """
        Args:
            window_seconds : Rolling window length in seconds (default 60s).
            fps_estimate   : Estimated frame rate for maxlen calculation.
        """
        self.window_seconds = window_seconds
        maxlen = int(window_seconds * fps_estimate * 1.5)  # buffer with margin
        self._timestamps   = deque()   # timestamps of all frames
        self._closed_flags = deque()   # True/False per frame (eye closed?)

    def update(self, eye_closed: bool):
        """Record current frame's eye state."""
        now = time.time()
        self._timestamps.append(now)
        self._closed_flags.append(eye_closed)

        # Evict frames older than the window
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._closed_flags.popleft()

    def get_perclos(self) -> float:
        """
        Returns PERCLOS value:
            0.0 = eyes open the entire window
            1.0 = eyes closed the entire window
            >0.35 = drowsy (research threshold)
        """
        total = len(self._closed_flags)
        if total == 0:
            return 0.0
        closed_count = sum(self._closed_flags)
        return closed_count / total

    def get_window_stats(self):
        """Return debug stats about the current window."""
        total = len(self._closed_flags)
        closed = sum(self._closed_flags)
        return {
            "window_frames": total,
            "closed_frames": closed,
            "open_frames": total - closed,
            "perclos": self.get_perclos(),
        }

    def reset(self):
        """Clear all history."""
        self._timestamps.clear()
        self._closed_flags.clear()
