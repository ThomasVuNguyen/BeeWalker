"""Flask web server for training visualization."""

import os
import sys
import time
import cv2
from flask import Flask, render_template, Response, jsonify

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import shared_state

app = Flask(__name__)

# Suppress Flask logging
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)


def generate_frames():
    """Generator for MJPEG video stream."""
    while True:
        frame = shared_state.get_frame()
        
        if frame is None:
            # Return a placeholder frame
            time.sleep(0.1)
            continue
        
        # Convert RGB (from MuJoCo) to BGR (for OpenCV encoding)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Serve the main dashboard page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Serve the MJPEG video stream."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def stats():
    """Return current training statistics as JSON."""
    current_stats, history = shared_state.get_stats()
    return jsonify(stats=current_stats, history=history)


def run_server(host='0.0.0.0', port=1306):
    """Start the Flask server."""
    print(f"Starting web server on http://{host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)


if __name__ == '__main__':
    run_server()
