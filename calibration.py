"""
calibration.py — Baseline Calibration System
=============================================
Why this is novel:
  Every person has different natural EAR/MAR values.
  A person with naturally small eyes will always have low EAR —
  using a fixed threshold (0.22) will trigger false alerts for them.

  This module measures the driver's PERSONAL baseline during a
  15-second calibration phase, then sets thresholds dynamically.

  This is exactly what commercial systems (Bosch, Seeing Machines) do.

Usage:
  from calibration import Calibrator
  cal = Calibrator()
  cal.start()              # begins collecting samples
  # ... run for 15 seconds while driver looks normally at camera ...
  thresholds = cal.finish() # returns personal EAR/MAR thresholds
"""

import numpy as np
import time
import json
import os

CALIBRATION_SECONDS  = 15     # How long to collect samples
CALIBRATION_FILE     = "calibration_data.json"
EAR_MARGIN           = 0.75   # threshold = baseline * this (e.g. 75% of normal)
MAR_MARGIN           = 1.40   # threshold = baseline * this (e.g. 140% of normal)


class Calibrator:
    def __init__(self):
        self.ear_samples   = []
        self.mar_samples   = []
        self.start_time    = None
        self.is_running    = False
        self.is_done       = False
        self.progress      = 0.0       # 0.0 to 1.0
        self.status        = "idle"    # idle | running | done | failed
        self.ear_threshold = 0.22      # fallback default
        self.mar_threshold = 0.60      # fallback default

    def start(self):
        """Begin calibration — call this when user clicks Calibrate."""
        self.ear_samples  = []
        self.mar_samples  = []
        self.start_time   = time.time()
        self.is_running   = True
        self.is_done      = False
        self.progress     = 0.0
        self.status       = "running"
        print("[Calibration] Started — keep eyes open and look at camera normally")

    def add_sample(self, ear: float, mar: float):
        """
        Feed each frame's EAR and MAR during calibration.
        Call this every frame while is_running is True.
        """
        if not self.is_running:
            return

        now     = time.time()
        elapsed = now - self.start_time
        self.progress = min(elapsed / CALIBRATION_SECONDS, 1.0)

        # Only accept samples where eyes are clearly open (filter blinks)
        if ear > 0.15:
            self.ear_samples.append(ear)
        if mar > 0.05:
            self.mar_samples.append(mar)

        if elapsed >= CALIBRATION_SECONDS:
            self._finish()

    def _finish(self):
        """Compute thresholds from collected samples."""
        self.is_running = False

        if len(self.ear_samples) < 30:
            self.status = "failed"
            print("[Calibration] Failed — not enough samples. Using defaults.")
            return

        # Use the 20th percentile of EAR (accounts for natural blinks during calibration)
        ear_baseline = float(np.percentile(self.ear_samples, 20))
        mar_baseline = float(np.percentile(self.mar_samples, 80))

        # Personal thresholds
        self.ear_threshold = round(ear_baseline * EAR_MARGIN, 4)
        self.mar_threshold = round(mar_baseline * MAR_MARGIN, 4)

        # Sanity clamp — don't go crazy
        self.ear_threshold = max(0.10, min(self.ear_threshold, 0.28))
        self.mar_threshold = max(0.40, min(self.mar_threshold, 0.85))

        self.is_done  = True
        self.progress = 1.0
        self.status   = "done"

        self._save()
        print(f"[Calibration] Done! EAR threshold: {self.ear_threshold:.4f} | MAR threshold: {self.mar_threshold:.4f}")

    def _save(self):
        """Save calibration to disk so it persists across sessions."""
        data = {
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "ear_baseline":  float(np.mean(self.ear_samples)),
            "mar_baseline":  float(np.mean(self.mar_samples)),
            "sample_count":  len(self.ear_samples),
            "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(CALIBRATION_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load_saved(self) -> bool:
        """
        Load a previously saved calibration.
        Returns True if found, False if not.
        """
        if not os.path.exists(CALIBRATION_FILE):
            return False
        try:
            with open(CALIBRATION_FILE) as f:
                data = json.load(f)
            self.ear_threshold = data["ear_threshold"]
            self.mar_threshold = data["mar_threshold"]
            self.is_done       = True
            self.status        = "done"
            print(f"[Calibration] Loaded saved calibration: EAR={self.ear_threshold}, MAR={self.mar_threshold}")
            return True
        except Exception as e:
            print(f"[Calibration] Failed to load: {e}")
            return False

    def get_thresholds(self):
        """Return current thresholds (personal if calibrated, default otherwise)."""
        return self.ear_threshold, self.mar_threshold

    def get_status_dict(self):
        """Return status dict for the frontend."""
        return {
            "status":        self.status,
            "progress":      round(self.progress * 100),
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "is_done":       self.is_done,
        }
