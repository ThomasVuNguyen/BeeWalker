"""
BeeWalker Web Dashboard - Real-time MJPEG Streaming
Uses Flask with multipart streaming for live training visualization.
"""

from flask import Flask, render_template_string, Response, jsonify
import cv2
import time
import numpy as np
from pathlib import Path
from src.shared_state import shared_state

app = Flask(__name__)


def generate_frames():
    """Generator for MJPEG video stream."""
    while True:
        frame = shared_state.get_frame()
        
        if frame is None:
            # Return a blank frame while waiting
            time.sleep(0.05)
            continue
        
        # Convert RGB to BGR for OpenCV encoding
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS cap


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/stats')
def stats():
    """JSON endpoint for training statistics."""
    s, history = shared_state.get_stats()
    return jsonify(stats=s, history=history)


def run_server():
    """Start the Flask server."""
    app.run(host='0.0.0.0', port=1306, debug=False, use_reloader=False, threaded=True)


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>BeeWalker Training</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', sans-serif; 
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
            color: white; 
            min-height: 100vh; 
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { font-size: 28px; margin-bottom: 20px; }
        .stats { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-box { 
            background: rgba(255,255,255,0.1); 
            padding: 15px 25px; 
            border-radius: 12px;
            min-width: 150px;
        }
        .stat-label { font-size: 12px; color: #888; text-transform: uppercase; }
        .stat-value { font-size: 24px; font-weight: bold; color: #4ade80; margin-top: 5px; }
        .video-container { 
            background: #000; 
            border-radius: 12px; 
            overflow: hidden;
            display: inline-block;
        }
        #video-feed { 
            max-width: 100%; 
            height: auto;
            display: block;
        }
        .chart-container { 
            margin-top: 20px; 
            height: 200px; 
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 15px;
        }
        canvas { width: 100%; height: 100%; }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>🐝 BeeWalker Training</h1>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-label">Episode</div>
                <div class="stat-value" id="episode">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Step</div>
                <div class="stat-value" id="step">0</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Reward</div>
                <div class="stat-value" id="reward">0.00</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Best Reward</div>
                <div class="stat-value" id="best">0.00</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Status</div>
                <div class="stat-value" id="status" style="font-size: 14px; color: #888;">Starting...</div>
            </div>
        </div>
        
        <div class="video-container">
            <img id="video-feed" src="/video_feed" alt="Training visualization">
        </div>
        
        <div class="chart-container">
            <canvas id="rewardChart"></canvas>
        </div>
    </div>
    
    <script>
        // Initialize chart
        const ctx = document.getElementById('rewardChart').getContext('2d');
        const chart = new Chart(ctx, {
            type: 'line',
            data: { labels: [], datasets: [{ 
                label: 'Episode Reward', 
                data: [], 
                borderColor: '#4ade80',
                backgroundColor: 'rgba(74, 222, 128, 0.1)',
                tension: 0.3,
                fill: true
            }]},
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: { grid: { color: 'rgba(255,255,255,0.1)' }, ticks: { color: '#888' } }
                },
                plugins: { legend: { display: false } }
            }
        });
        
        // Poll stats every second
        setInterval(async () => {
            try {
                const res = await fetch('/stats');
                const data = await res.json();
                
                document.getElementById('episode').textContent = data.stats.episode;
                document.getElementById('step').textContent = data.stats.step.toLocaleString();
                document.getElementById('reward').textContent = data.stats.reward.toFixed(2);
                document.getElementById('best').textContent = data.stats.best_reward.toFixed(2);
                document.getElementById('status').textContent = data.stats.status;
                
                // Update chart
                chart.data.labels = data.history.map((_, i) => i);
                chart.data.datasets[0].data = data.history;
                chart.update('none');
            } catch(e) {}
        }, 1000);
    </script>
</body>
</html>
'''


if __name__ == '__main__':
    run_server()
