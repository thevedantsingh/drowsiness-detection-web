# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system built with **MediaPipe Face Landmarker**, **OpenCV**, and **Flask**. It tracks eye closure, yawning, and head nodding through facial geometry, then serves a live analytics dashboard in the browser with instant Telegram/Pushover/Pushbullet alerts.

---

## 🌟 Features

- **Eye closure detection** — EAR (Eye Aspect Ratio) from eye landmark geometry
- **Yawn detection** — MAR (Mouth Aspect Ratio) from mouth landmark geometry
- **PERCLOS tracking** — rolling-window percentage of eye closure (the NHTSA-standard drowsiness metric)
- **Personal calibration** — a 15-second calibration step tunes EAR/MAR thresholds to your own face
- **Night mode** — CLAHE-based low-light image enhancement
- **Voice alerts** — text-to-speech warnings for drowsiness and yawning
- **Remote notifications** — Telegram, Pushover, or Pushbullet push alerts
- **Live web dashboard** — MJPEG video stream + real-time EAR/MAR/PERCLOS charts, built with Flask + Chart.js

---

## 📁 Project Structure

> Note: `face_landmarker.task` (the MediaPipe model file) is auto-downloaded by `detector.py` on first run if it isn't already present.

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/thevedantsingh/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure alerts
Copy the example config and fill in your own credentials:
```bash
cp config.example.py config.py
```
Then edit `config.py` and add your Telegram bot token / chat ID (or Pushover / Pushbullet keys). **`config.py` is gitignored — never commit your real tokens.**

### 4. Run the app
```bash
python app.py
```
Then open **http://localhost:5000** in your browser. On your phone (same WiFi), use `http://<your-laptop-ip>:5000`.

---

## 🔬 How Detection Works

**EAR (Eye Aspect Ratio)**
**MAR (Mouth Aspect Ratio)**
**PERCLOS**
PERCLOS is the metric used in NHTSA (National Highway Traffic Safety Administration) drowsiness research.

Because detection is based on facial landmark geometry rather than raw pixel color, it stays robust with glasses, in low light (with night mode), and under partial occlusion.

---

## ⚙️ Configuration

Detection thresholds live in `config.py`:

```python
EAR_THRESHOLD          = 0.22   # Lower = more tolerant
MAR_THRESHOLD           = 0.60   # Higher = less sensitive to yawns
EAR_CONSEC_FRAMES        = 20     # Frames of closed eyes before alert
MAR_CONSEC_FRAMES         = 15
PERCLOS_THRESHOLD          = 0.35   # 35% closure = drowsy
HEAD_PITCH_THRESH           = 15     # Degrees of head nod
PERCLOS_WINDOW                = 60     # Rolling window, seconds
NOTIFICATION_COOLDOWN          = 30     # Seconds between repeat alerts
```

---

## 📱 Notifications Setup

Pick one method in `config.py` via `NOTIFY_METHOD`:

- **Telegram** — create a bot with [@BotFather](https://t.me/BotFather), set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`
- **Pushover** — set `PUSHOVER_TOKEN` and `PUSHOVER_USER`
- **Pushbullet** — set `PUSHBULLET_TOKEN`

---

## 🛠️ Tech Stack

- **MediaPipe Face Landmarker** — facial landmark detection
- **OpenCV** — camera capture & image processing
- **Flask** — web server & REST endpoints
- **Chart.js** — live dashboard charts
- **pygame** — alert sound playback
