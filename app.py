import cv2
import numpy as np
import mediapipe as mp
import time
import json
from flask import Flask, Response, request, render_template_string, jsonify

app = Flask(__name__)

# MediaPipe Face Mesh Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Keypoints ของตาซ้ายและขวา
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Global System State
presets = {
    1: {"name": "User 1", "ear_open": 0.30, "ear_closed": 0.10},
    2: {"name": "User 2", "ear_open": 0.30, "ear_closed": 0.10},
    3: {"name": "User 3", "ear_open": 0.30, "ear_closed": 0.10}
}
active_preset_id = 1

latest_frame = None
current_closure_pct = 0.0
closed_duration_sec = 0.0
closed_start_time = None
alarm_state = False

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = np.array([(landmarks[idx].x * img_w, landmarks[idx].y * img_h) for idx in eye_indices])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    h = np.linalg.norm(pts[0] - pts[3])
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
    return ear

def process_frame(image_bytes):
    global latest_frame, current_closure_pct, closed_duration_sec, closed_start_time, alarm_state
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        # ดึงค่า calibration จาก Preset ปัจจุบัน
        p = presets[active_preset_id]
        ear_open = p["ear_open"]
        ear_closed = p["ear_closed"]

        # คำนวณ % การหลับตา
        if ear_open > ear_closed:
            raw_pct = (1.0 - (avg_ear - ear_closed) / (ear_open - ear_closed)) * 100.0
            current_closure_pct = max(0.0, min(100.0, raw_pct))
        else:
            current_closure_pct = 0.0

        # ตรวจจับเวลาถ้าหลับตาเกิน 70%
        if current_closure_pct >= 70.0:
            if closed_start_time is None:
                closed_start_time = time.time()
            closed_duration_sec = time.time() - closed_start_time
            if closed_duration_sec >= 10.0:
                alarm_state = True
        else:
            closed_start_time = None
            closed_duration_sec = 0.0
            alarm_state = False

        # วาดข้อความลงบนภาพ Preview
        color = (0, 0, 255) if alarm_state else (0, 255, 0)
        cv2.putText(frame, f"Closure: {current_closure_pct:.1f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Closed Time: {closed_duration_sec:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if alarm_state:
            cv2.putText(frame, "DROWSINESS ALARM!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    _, jpeg = cv2.imencode('.jpg', frame)
    latest_frame = jpeg.tobytes()

@app.route('/upload', methods=['POST'])
def upload():
    image_bytes = request.data
    process_frame(image_bytes)
    return jsonify({"status": "ok", "alarm": alarm_state})

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "closure_pct": round(current_closure_pct, 1),
        "closed_sec": round(closed_duration_sec, 1),
        "alarm": alarm_state,
        "active_preset": active_preset_id,
        "presets": presets
    })

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            if latest_frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + latest_frame + b'\r\n')
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/select_preset', methods=['POST'])
def select_preset():
    global active_preset_id
    data = request.json
    preset_id = int(data.get('preset_id', 1))
    if preset_id in presets:
        active_preset_id = preset_id
    return jsonify({"status": "success", "active_preset": active_preset_id})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    global presets
    data = request.json
    preset_id = int(data.get('preset_id'))
    step = data.get('step')  # 'open' or 'closed'
    
    if latest_frame and preset_id in presets:
        nparr = np.frombuffer(latest_frame, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w, _ = frame.shape
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            l_ear = calculate_ear(landmarks, LEFT_EYE, w, h)
            r_ear = calculate_ear(landmarks, RIGHT_EYE, w, h)
            avg_ear = (l_ear + r_ear) / 2.0
            
            if step == 'open':
                presets[preset_id]['ear_open'] = round(avg_ear, 4)
            elif step == 'closed':
                presets[preset_id]['ear_closed'] = round(avg_ear, 4)
            return jsonify({"status": "success", "preset": presets[preset_id]})
            
    return jsonify({"status": "failed", "reason": "No face detected"}), 400

# HTML UI Template
HTML_UI = """
<!DOCTYPE html>
<html>
<head>
    <title>Driver Drowsiness Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; background: #121212; color: #fff; margin: 0; padding: 20px; text-align: center; }
        .container { max-width: 700px; margin: auto; background: #1e1e1e; padding: 20px; border-radius: 12px; }
        img { width: 100%; max-width: 500px; border-radius: 8px; border: 2px solid #333; }
        .stats-box { display: flex; justify-content: space-around; margin: 20px 0; }
        .card { background: #2a2a2a; padding: 15px; border-radius: 8px; width: 40%; }
        .alarm-active { background: #d32f2f !important; animation: blink 1s infinite; }
        @keyframes blink { 50% { opacity: 0.5; } }
        button, select { padding: 10px 15px; margin: 5px; border: none; border-radius: 5px; font-weight: bold; cursor: pointer; }
        .btn-primary { background: #007bff; color: white; }
        .btn-warn { background: #ff9800; color: white; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Driver Drowsiness Detection System</h2>
        <img src="/video_feed" alt="Camera Feed">
        
        <div class="stats-box">
            <div class="card" id="card-closure">
                <h3>Eye Closure</h3>
                <h1 id="closure-val">0 %</h1>
            </div>
            <div class="card" id="card-time">
                <h3>Closed Duration</h3>
                <h1 id="time-val">0.0 s</h1>
            </div>
        </div>

        <div style="margin-bottom: 20px;">
            <label>Active Profile: </label>
            <select id="preset-select" onchange="switchPreset()">
                <option value="1">Preset 1 (User 1)</option>
                <option value="2">Preset 2 (User 2)</option>
                <option value="3">Preset 3 (User 3)</option>
            </select>
        </div>

        <div style="border-top: 1px solid #444; padding-top: 15px;">
            <h3>Train / Calibrate Current Preset</h3>
            <p>นำใบหน้าเข้าใกล้กล้อง แล้วกดบันทึกตามขั้นตอน</p>
            <button class="btn-primary" onclick="calibrate('open')">1. บันทึกตอน "เปิดตา"</button>
            <button class="btn-warn" onclick="calibrate('closed')">2. บันทึกตอน "หลับตา"</button>
            <p id="calib-status" style="color: #4caf50;"></p>
        </div>
    </div>

    <script>
        function updateStats() {
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('closure-val').innerText = data.closure_pct + ' %';
                    document.getElementById('time-val').innerText = data.closed_sec + ' s';
                    document.getElementById('preset-select').value = data.active_preset;

                    const cardTime = document.getElementById('card-time');
                    if(data.alarm) {
                        cardTime.classList.add('alarm-active');
                    } else {
                        cardTime.classList.remove('alarm-active');
                    }
                });
        }
        setInterval(updateStats, 500);

        function switchPreset() {
            const id = document.getElementById('preset-select').value;
            fetch('/select_preset', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({preset_id: id})
            });
        }

        function calibrate(step) {
            const id = document.getElementById('preset-select').value;
            fetch('/calibrate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({preset_id: id, step: step})
            })
            .then(r => r.json())
            .then(res => {
                if(res.status === 'success') {
                    document.getElementById('calib-status').innerText = `บันทึก ${step} เรียบร้อย! (EAR: ${res.preset['ear_' + step]})`;
                } else {
                    alert('สแกนไม่สำเร็จ! กรุณาหันหน้าเข้ากล้องให้ชัดเจน');
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_UI)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
