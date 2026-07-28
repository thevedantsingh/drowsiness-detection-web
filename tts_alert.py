"""
tts_alert.py — Text-to-Speech Alert System
==========================================
Uses Windows built-in TTS (pyttsx3) — NO internet needed, NO API key.
Falls back gracefully if pyttsx3 is unavailable.

Install: pip install pyttsx3

Alert messages are:
  Level 1: "Warning. Your eyes are closing. Please stay alert."
  Level 2: "Caution! Drowsiness detected. Take a break soon."
  Level 3: "DANGER! You are falling asleep! Pull over immediately!"

Messages are spoken in a background thread so they never block the
video stream.
"""

import threading
import time

try:
    import pyttsx3
    _engine = pyttsx3.init()
    _engine.setProperty("rate",   150)    # words per minute (default ~200)
    _engine.setProperty("volume", 1.0)    # 0.0 to 1.0
    # Try to use a female voice if available (sounds clearer for alerts)
    voices = _engine.getProperty("voices")
    for v in voices:
        if "female" in v.name.lower() or "zira" in v.name.lower():
            _engine.setProperty("voice", v.id)
            break
    TTS_AVAILABLE = True
    print("[TTS] pyttsx3 initialised successfully")
except Exception as e:
    TTS_AVAILABLE = False
    print(f"[TTS] pyttsx3 not available: {e}. Run: pip install pyttsx3")

# Alert messages — designed to escalate in urgency
ALERT_MESSAGES = {
    1: "Warning. Your eyes are closing. Please stay alert.",
    2: "Caution! Drowsiness detected. Please take a break soon.",
    3: "Danger! You are falling asleep! Pull over immediately!",
}

# Custom yawn message
YAWN_MESSAGE = "You just yawned. Consider taking a short break."

_tts_lock      = threading.Lock()
_last_spoken   = {}      # level → last time spoken
TTS_COOLDOWNS  = {1: 12, 2: 8, 3: 4}    # seconds between same-level speech


def speak(text: str, block: bool = False):
    """
    Speak text in a background thread.
    If block=True, waits for speech to finish (use only in non-time-critical code).
    """
    if not TTS_AVAILABLE:
        print(f"[TTS fallback] {text}")
        return

    def _speak():
        with _tts_lock:
            try:
                _engine.say(text)
                _engine.runAndWait()
            except Exception as e:
                print(f"[TTS error] {e}")

    if block:
        _speak()
    else:
        threading.Thread(target=_speak, daemon=True).start()


def speak_alert(level: int):
    """
    Speak the alert message for the given level.
    Respects per-level cooldowns to avoid repetition.
    """
    now      = time.time()
    cooldown = TTS_COOLDOWNS.get(level, 8)
    last     = _last_spoken.get(level, 0)

    if (now - last) < cooldown:
        return   # Still in cooldown

    _last_spoken[level] = now
    message = ALERT_MESSAGES.get(level, "Please stay alert.")
    speak(message)


def speak_calibration_prompt():
    """Announce the start of calibration to the user."""
    speak("Calibration starting. Please look at the camera normally for 15 seconds.", block=False)


def speak_calibration_done():
    """Announce calibration completion."""
    speak("Calibration complete. Your personal drowsiness thresholds have been set.", block=False)


def speak_yawn():
    """Announce a detected yawn."""
    now  = time.time()
    last = _last_spoken.get("yawn", 0)
    if (now - last) < 20:   # Don't announce yawn more than once per 20 seconds
        return
    _last_spoken["yawn"] = now
    speak(YAWN_MESSAGE)


def set_voice_rate(rate: int = 150):
    """Adjust speech speed. 100=slow, 150=normal, 200=fast."""
    if TTS_AVAILABLE:
        _engine.setProperty("rate", rate)


def set_volume(volume: float = 1.0):
    """Adjust volume 0.0 to 1.0."""
    if TTS_AVAILABLE:
        _engine.setProperty("volume", max(0.0, min(1.0, volume)))
