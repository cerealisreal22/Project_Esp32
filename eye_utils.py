import threading
import numpy as np
import mediapipe as mp
import cv2

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

_lock = threading.Lock()   # เพิ่มบรรทัดนี้ - กันการเรียก face_mesh ซ้อนกันจากหลาย thread

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def _dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def _ear(landmarks, idx, w, h):
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in idx]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return None
    return vertical / (2.0 * horizontal)

def get_ear_from_image(img_bgr):
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with _lock:   # เพิ่มบรรทัดนี้ - บังคับให้ประมวลผลทีละภาพเท่านั้น
        result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None
    lm = result.multi_face_landmarks[0].landmark
    left = _ear(lm, LEFT_EYE, w, h)
    right = _ear(lm, RIGHT_EYE, w, h)
    if left is None or right is None:
        return None
    return (left + right) / 2.0
