import os, json, time
from flask import Flask, request, jsonify, Response, send_from_directory
import numpy as np, cv2
from eye_utils import get_ear_from_image

app = Flask(__name__, static_folder="static", template_folder="templates")

DATA_DIR = os.environ.get("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)
PRESET_FILE = os.path.join(DATA_DIR, "presets.json")

def load_presets():
    if not os.path.exists(PRESET_FILE):
        return {"presets": [], "active_id": None}
    with open(PRESET_FILE) as f:
        return json.load(f)

def save_presets(data):
    with open(PRESET_FILE, "w") as f:
        json.dump(data, f)

state = {"eye_state": "unknown", "closed_percent": 0, "closed_duration": 0,
         "alarm": False, "updated_at": 0}
last_frame_bytes = {"jpg": None}
train_session = {"active": False, "phase": "open", "samples_open": [], "samples_closed": []}

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")

# ---------- PRESET APIs ----------
@app.route("/api/presets")
def get_presets():
    return jsonify(load_presets())

@app.route("/api/preset/select", methods=["POST"])
def select_preset():
    body = request.json
    data = load_presets()
    data["active_id"] = body.get("id")
    save_presets(data)
    return jsonify({"ok": True})

@app.route("/api/preset/active")
def active_preset():
    data = load_presets()
    p = next((x for x in data["presets"] if x["id"] == data.get("active_id")), None)
    return jsonify({"active": bool(p), "preset": p})

@app.route("/api/preset/delete", methods=["POST"])
def delete_preset():
    body = request.json
    data = load_presets()
    data["presets"] = [p for p in data["presets"] if p["id"] != body.get("id")]
    if data.get("active_id") == body.get("id"):
        data["active_id"] = None
    save_presets(data)
    return jsonify({"ok": True})

# ---------- TRAINING / CALIBRATION ----------
@app.route("/api/train/start", methods=["POST"])
def train_start():
    global train_session
    train_session = {"active": True, "phase": "open", "samples_open": [], "samples_closed": []}
    return jsonify({"ok": True})

@app.route("/api/train/phase", methods=["POST"])
def train_phase():
    train_session["phase"] = request.json.get("phase")  # "open" หรือ "closed"
    return jsonify({"ok": True})

@app.route("/api/train/frame", methods=["POST"])
def train_frame():
    if not train_session["active"]:
        return jsonify({"error": "no session"}), 400
    file = request.files.get("image")
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    ear = get_ear_from_image(img)
    if ear is None:
        return jsonify({"ok": False, "msg": "ไม่เจอใบหน้า ขยับเข้าใกล้กล้อง"})
    key = "samples_open" if train_session["phase"] == "open" else "samples_closed"
    train_session[key].append(ear)
    return jsonify({"ok": True, "ear": ear,
                     "count_open": len(train_session["samples_open"]),
                     "count_closed": len(train_session["samples_closed"])})

@app.route("/api/train/save", methods=["POST"])
def train_save():
    body = request.json
    name = body.get("name", "Preset")
    slot_id = body.get("id")
    opens, closes = train_session["samples_open"], train_session["samples_closed"]
    if len(opens) < 5 or len(closes) < 5:
        return jsonify({"error": "เก็บตัวอย่างไม่พอ ต้องอย่างน้อย 5 เฟรมต่อช่วง"}), 400

    ear_open_avg = float(np.mean(opens))
    ear_closed_avg = float(np.mean(closes))
    ear_threshold = ear_closed_avg + (ear_open_avg - ear_closed_avg) * 0.4

    preset = {
        "id": slot_id, "name": name,
        "ear_open_avg": ear_open_avg, "ear_closed_avg": ear_closed_avg,
        "ear_threshold": ear_threshold,
        "percent_threshold": body.get("percent_threshold", 70),
        "duration_threshold_sec": body.get("duration_threshold_sec", 10),
    }
    data = load_presets()
    data["presets"] = [p for p in data["presets"] if p["id"] != slot_id]
    if len(data["presets"]) >= 3:
        return jsonify({"error": "เก็บได้สูงสุด 3 preset กรุณาลบอันเก่าก่อน"}), 400
    data["presets"].append(preset)
    save_presets(data)
    return jsonify({"ok": True, "preset": preset})

# ---------- ESP32 -> PREDICT ----------
@app.route("/api/predict", methods=["POST"])
def predict():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "no image"}), 400
    img_bytes = file.read()
    last_frame_bytes["jpg"] = img_bytes

    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    ear = get_ear_from_image(img)

    data = load_presets()
    active = next((p for p in data["presets"] if p["id"] == data.get("active_id")), None)

    if ear is None or active is None:
        return jsonify({"eye_state": "unknown", "ear": ear,
                         "threshold": active["ear_threshold"] if active else None})

    eye_state = "closed" if ear < active["ear_threshold"] else "open"
    return jsonify({"eye_state": eye_state, "ear": ear, "threshold": active["ear_threshold"]})

# ---------- ESP32 -> STATS ----------
@app.route("/api/stats", methods=["POST"])
def post_stats():
    body = request.json
    state.update({**body, "updated_at": time.time()})
    return jsonify({"ok": True})

@app.route("/api/stats")
def get_stats():
    return jsonify(state)

@app.route("/api/frame")
def get_frame():
    if not last_frame_bytes["jpg"]:
        return "", 204
    return Response(last_frame_bytes["jpg"], mimetype="image/jpeg")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))