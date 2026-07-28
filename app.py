"""
app.py — Flask Web Server
==========================
Routes:
  GET  /              → Main dashboard (video + charts + controls)
  GET  /video_feed    → MJPEG camera stream
  GET  /stats         → JSON stats (polled every 800ms by frontend)
  GET  /graph_data    → JSON graph history (polled every 2s by charts)
  POST /calibrate     → Start calibration
  GET  /cal_status    → Calibration status + progress
  GET  /stop          → Stop detector
"""

from flask import Flask, render_template, Response, jsonify
import detector as det

app = Flask(__name__)

@app.before_request
def _start():
    if not det.detector.running:
        det.detector.start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def gen():
        import time
        while True:
            frame = det.detector.get_frame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.03)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stats")
def stats():
    return jsonify(det.detector.get_stats())

@app.route("/graph_data")
def graph_data():
    """Returns last 60s of EAR/MAR/PERCLOS/Alert data for charts."""
    return jsonify(det.detector.get_graph_data())

@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Start the 15-second personal calibration."""
    det.detector.start_calibration()
    return jsonify({"status": "started"})

@app.route("/cal_status")
def cal_status():
    """Poll calibration progress."""
    return jsonify(det.detector.get_calibration_status())

@app.route("/stop")
def stop():
    det.detector.stop()
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    print("[INFO] Drowsiness Monitor → http://localhost:5000")
    print("[INFO] On your phone (same WiFi) → http://<your-laptop-ip>:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
