"""
detector.py — Main Detection Loop
==================================
Updated with:
  - Baseline calibration (personal EAR/MAR thresholds)
  - Text-to-speech alerts
  - Real-time graph data buffer (for dashboard charts)
  - Yawn TTS announcement
"""

import cv2, time, numpy as np, threading, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from collections import deque
import urllib.request, os
import config
from notifier    import send_notification
from calibration import Calibrator
from tts_alert   import speak_alert, speak_yawn, speak_calibration_done

# ── Model download ──────────────────────────────────────────────────────────
MODEL_PATH = "face_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face landmarker model (~30MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model downloaded!")

# ── Landmark indices ────────────────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH     = [61,  39,  0,  269, 291, 405, 17, 181]

# ── Graph data buffer — stores last 60 seconds of readings ─────────────────
GRAPH_MAXLEN = 300   # 300 points @ ~5 per second = 60 seconds of history


def euclidean(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def lm_pt(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))

def compute_EAR(lms, indices, w, h):
    pts = [lm_pt(lms[i], w, h) for i in indices]
    hor = euclidean(pts[0], pts[3])
    if hor < 1e-6: return 0.0
    return (euclidean(pts[1], pts[5]) + euclidean(pts[2], pts[4])) / (2.0 * hor)

def compute_MAR(lms, indices, w, h):
    pts = [lm_pt(lms[i], w, h) for i in indices]
    hor = euclidean(pts[0], pts[4])
    if hor < 1e-6: return 0.0
    return (euclidean(pts[1],pts[7]) + euclidean(pts[2],pts[6]) + euclidean(pts[3],pts[5])) / (3.0 * hor)


class DrowsinessDetector:
    def __init__(self):
        self.cap              = None
        self.running          = False
        self.frame_bytes      = None
        self.lock             = threading.Lock()
        self._latest_result   = None
        self._result_lock     = threading.Lock()

        # ── Calibrator (personal thresholds) ──
        self.calibrator       = Calibrator()
        cal_loaded            = self.calibrator.load_saved()
        if not cal_loaded:
            print("[Detector] No calibration found — using defaults. Click 'Calibrate' in UI.")

        # ── State ──
        self.ear              = 0.0
        self.mar              = 0.0
        self.perclos          = 0.0
        self.blink_rate       = 0.0
        self.alert_level      = 0
        self.total_blinks     = 0
        self.total_yawns      = 0
        self.eye_closed_ctr   = 0
        self.yawn_ctr         = 0
        self.perclos_buf      = deque()
        self.blink_times      = deque()
        self.session_start    = time.time()
        self.face_detected    = False
        self.last_alert_time  = 0

        # ── Graph data buffers (thread-safe) ──
        self.graph_lock       = threading.Lock()
        self.graph_times      = deque(maxlen=GRAPH_MAXLEN)   # elapsed seconds
        self.graph_ear        = deque(maxlen=GRAPH_MAXLEN)
        self.graph_mar        = deque(maxlen=GRAPH_MAXLEN)
        self.graph_perclos    = deque(maxlen=GRAPH_MAXLEN)
        self.graph_alert      = deque(maxlen=GRAPH_MAXLEN)
        self._graph_frame_ctr = 0   # only log every N frames

    # ── Graph data ──────────────────────────────────────────────────────────
    def _log_graph_point(self):
        """Log a data point every 6 frames (~5 points/sec at 30fps)."""
        self._graph_frame_ctr += 1
        if self._graph_frame_ctr % 6 != 0:
            return
        elapsed = round(time.time() - self.session_start, 1)
        with self.graph_lock:
            self.graph_times.append(elapsed)
            self.graph_ear.append(round(self.ear, 3))
            self.graph_mar.append(round(self.mar, 3))
            self.graph_perclos.append(round(self.perclos, 3))
            self.graph_alert.append(self.alert_level)

    def get_graph_data(self):
        """Return graph data for the dashboard charts API."""
        with self.graph_lock:
            return {
                "times":   list(self.graph_times),
                "ear":     list(self.graph_ear),
                "mar":     list(self.graph_mar),
                "perclos": list(self.graph_perclos),
                "alert":   list(self.graph_alert),
                "ear_threshold":  round(self.calibrator.ear_threshold, 3),
                "mar_threshold":  round(self.calibrator.mar_threshold, 3),
            }

    # ── Calibration controls ────────────────────────────────────────────────
    def start_calibration(self):
        """Called when user clicks 'Calibrate' button."""
        self.calibrator.start()

    def get_calibration_status(self):
        return self.calibrator.get_status_dict()

    # ── Internal helpers ────────────────────────────────────────────────────
    def _result_callback(self, result, output_image, timestamp_ms):
        with self._result_lock:
            self._latest_result = result

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def _update_perclos(self, closed):
        now = time.time()
        self.perclos_buf.append((now, closed))
        cutoff = now - config.PERCLOS_WINDOW
        while self.perclos_buf and self.perclos_buf[0][0] < cutoff:
            self.perclos_buf.popleft()
        if self.perclos_buf:
            self.perclos = sum(1 for _, c in self.perclos_buf if c) / len(self.perclos_buf)

    def _get_blink_rate(self):
        now = time.time()
        while self.blink_times and self.blink_times[0] < now - 60:
            self.blink_times.popleft()
        return len(self.blink_times)

    def _trigger_alert(self, level, msg):
        now      = time.time()
        cooldown = {1: 10, 2: 6, 3: 3}.get(level, 5)
        if (now - self.last_alert_time) < cooldown:
            return
        self.last_alert_time = now
        titles = {1: "⚠️ Mild Drowsiness", 2: "⚠️ Moderate Drowsiness", 3: "🚨 SEVERE — Wake Up!"}
        # TTS alert (local speaker)
        speak_alert(level)
        # Telegram notification (phone) — only level 2+
        if level >= 2:
            send_notification(titles[level], msg, level)

    def _draw_hud(self, frame, w, h, ear_thresh, mar_thresh):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (230, h), (10, 12, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        def bar(label, val, maxv, y, col):
            cv2.putText(frame, label, (8, y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170,170,170), 1)
            cv2.rectangle(frame, (8, y), (215, y+10), (40,40,40), -1)
            fill = int(min(val/maxv, 1) * 207)
            cv2.rectangle(frame, (8, y), (8+fill, y+10), col, -1)
            cv2.putText(frame, f"{val:.3f}", (218, y+9), cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1)

        ec = (0,220,80) if self.ear > ear_thresh else (0,60,230)
        mc = (0,220,80) if self.mar < mar_thresh  else (0,155,255)
        pc = (0,220,80) if self.perclos < config.PERCLOS_THRESHOLD else (0,60,230)

        bar("EAR  (eye open)",    self.ear,     0.40, 42,  ec)
        bar("MAR  (mouth open)",  self.mar,     1.00, 70,  mc)
        bar("PERCLOS (% closed)", self.perclos, 1.00, 98,  pc)

        # Calibration indicator
        cal_status = self.calibrator.status
        cal_col    = (0,200,100) if cal_status == "done" else (0,140,255) if cal_status == "running" else (100,100,100)
        cal_label  = f"CAL: {'✓ Personal' if cal_status=='done' else 'Running...' if cal_status=='running' else 'Default'}"
        cv2.putText(frame, cal_label, (8, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.37, cal_col, 1)

        # Calibration progress bar
        if cal_status == "running":
            prog = int(self.calibrator.progress * 207)
            cv2.rectangle(frame, (8, 130), (215, 137), (30,30,30), -1)
            cv2.rectangle(frame, (8, 130), (8+prog, 137), (0,180,120), -1)

        info = [("Blinks/min", f"{self.blink_rate:.0f}"),
                ("Yawns",      str(self.total_yawns)),
                ("Blinks",     str(self.total_blinks))]
        for i, (l, v) in enumerate(info):
            y = 152 + i * 20
            cv2.putText(frame, l+":", (8, y),  cv2.FONT_HERSHEY_SIMPLEX, 0.37, (140,140,140), 1)
            cv2.putText(frame, v,     (130, y), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (255,255,255), 1)

        # Alert banner
        colors = [(0,0,0),(0,190,255),(0,130,255),(0,50,200)]
        labels = ["", "MILD — Eyes closing", "MODERATE — Drowsy!", "SEVERE — WAKE UP!"]
        if self.alert_level > 0:
            cv2.rectangle(frame, (230, 0), (w, 42), colors[self.alert_level], -1)
            cv2.putText(frame, labels[self.alert_level], (238, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        if not self.face_detected:
            cv2.putText(frame, "NO FACE DETECTED", (w//2-110, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,220), 2)

    def _loop(self):
        base_opts  = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts       = mp_vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_faces=3,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback
        )
        landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
        frame_ts   = 0

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.03)
                continue

            h, w      = frame.shape[:2]
            frame_ts += 1

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_img, frame_ts)

            with self._result_lock:
                result = self._latest_result

            self.face_detected = bool(result and result.face_landmarks)

            # Get current personal thresholds
            ear_thresh, mar_thresh = self.calibrator.get_thresholds()

            if self.face_detected:
                all_faces = result.face_landmarks
                lms = max(all_faces,
                    key=lambda f: (max(l.x for l in f)-min(l.x for l in f)) *
                                  (max(l.y for l in f)-min(l.y for l in f)))

                self.ear = (compute_EAR(lms, LEFT_EYE, w, h) + compute_EAR(lms, RIGHT_EYE, w, h)) / 2
                self.mar = compute_MAR(lms, MOUTH, w, h)

                # Feed calibrator if active
                if self.calibrator.is_running:
                    was_done = self.calibrator.is_done
                    self.calibrator.add_sample(self.ear, self.mar)
                    if not was_done and self.calibrator.is_done:
                        speak_calibration_done()
                    ear_thresh, mar_thresh = self.calibrator.get_thresholds()

                eye_closed = self.ear < ear_thresh
                self._update_perclos(eye_closed)

                if eye_closed:
                    self.eye_closed_ctr += 1
                else:
                    if 2 <= self.eye_closed_ctr <= 12:
                        self.total_blinks += 1
                        self.blink_times.append(time.time())
                    self.eye_closed_ctr = 0

                if self.mar > mar_thresh:
                    self.yawn_ctr += 1
                else:
                    if self.yawn_ctr >= config.MAR_CONSEC_FRAMES:
                        self.total_yawns += 1
                        speak_yawn()
                    self.yawn_ctr = 0

                self.blink_rate = self._get_blink_rate()

                score = 0; msgs = []
                if self.eye_closed_ctr > config.EAR_CONSEC_FRAMES:
                    score += 2; msgs.append("eyes closed")
                if self.perclos > config.PERCLOS_THRESHOLD:
                    score += 2; msgs.append(f"PERCLOS {self.perclos:.0%}")
                if self.yawn_ctr > config.MAR_CONSEC_FRAMES:
                    score += 1; msgs.append("yawning")
                if self.blink_rate > 25:
                    score += 1; msgs.append("rapid blinking")

                self.alert_level = 3 if score >= 4 else 2 if score >= 2 else 1 if score >= 1 else 0

                if self.alert_level >= 1:
                    self._trigger_alert(self.alert_level, "Detected: " + ", ".join(msgs))

                # Draw contours
                eye_closed_now = self.ear < ear_thresh
                for indices, col in [(LEFT_EYE,(0,220,80)), (RIGHT_EYE,(0,220,80)), (MOUTH,(0,200,255))]:
                    pts = np.array([(int(lms[i].x*w), int(lms[i].y*h)) for i in indices], np.int32)
                    cv2.polylines(frame, [pts], True,
                                  (0,60,230) if (indices in [LEFT_EYE,RIGHT_EYE] and eye_closed_now) else col, 1)

            self._draw_hud(frame, w, h, ear_thresh, mar_thresh)
            self._log_graph_point()

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.frame_bytes = buf.tobytes()

        landmarker.close()

    def get_frame(self):
        with self.lock:
            return self.frame_bytes

    def get_stats(self):
        ear_thresh, mar_thresh = self.calibrator.get_thresholds()
        return {
            "ear":           round(self.ear, 3),
            "mar":           round(self.mar, 3),
            "perclos":       round(self.perclos, 3),
            "blink_rate":    round(self.blink_rate, 1),
            "alert_level":   self.alert_level,
            "total_blinks":  self.total_blinks,
            "total_yawns":   self.total_yawns,
            "face":          self.face_detected,
            "elapsed":       int(time.time() - self.session_start),
            "ear_threshold": round(ear_thresh, 3),
            "mar_threshold": round(mar_thresh, 3),
            "calibrated":    self.calibrator.is_done,
            "cal_status":    self.calibrator.status,
            "cal_progress":  round(self.calibrator.progress * 100),
        }


detector = DrowsinessDetector()
