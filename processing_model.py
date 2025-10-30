import json
import numpy as np
from input_capture import capture_landmarks

# Eye landmark indices from MediaPipe
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Thresholds for attention detection
EAR_THRESHOLD = 0.25       # Eyes closed threshold
HEAD_DEV_THRESHOLD = 0.05  # Head rotation threshold


def eye_aspect_ratio(eye_points, h, w):
    """Calculate Eye Aspect Ratio (EAR) for blink/closure detection."""
    coords = [(int(p['x'] * w), int(p['y'] * h)) for p in eye_points]
    A = np.linalg.norm(np.array(coords[1]) - np.array(coords[5]))
    B = np.linalg.norm(np.array(coords[2]) - np.array(coords[4]))
    C = np.linalg.norm(np.array(coords[0]) - np.array(coords[3]))
    if C == 0:
        return 0
    return (A + B) / (2.0 * C)


def process_attention(json_input):
    """Evaluate attention state using EAR and head deviation."""
    data = json.loads(json_input)
    h, w = data['height'], data['width']
    landmarks = data['landmarks']

    if len(landmarks) < 400:  # no valid face
        return json.dumps({"attentive": False, "status": "No Face"})

    # Select eye and nose landmarks
    left_eye = [landmarks[i] for i in LEFT_EYE_IDX]
    right_eye = [landmarks[i] for i in RIGHT_EYE_IDX]
    nose_tip = landmarks[1]

    left_ear = eye_aspect_ratio(left_eye, h, w)
    right_ear = eye_aspect_ratio(right_eye, h, w)

    attentive = True

    # Condition 1: Eyes closed
    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
        attentive = False

    # Condition 2: Head deviation from center
    nose_x = nose_tip['x'] * w
    left_x = left_eye[0]['x'] * w
    right_x = right_eye[3]['x'] * w
    deviation = abs(nose_x - (left_x + right_x) / 2)
    if deviation > w * HEAD_DEV_THRESHOLD:
        attentive = False

    return json.dumps({
        "attentive": attentive,
        "leftEAR": round(left_ear, 3),
        "rightEAR": round(right_ear, 3),
        "status": "ATTENTIVE" if attentive else "NOT ATTENTIVE"
    })


if __name__ == "__main__":
    print("Starting attention processing... Press 'q' in camera window to stop.")
    for frame_data in capture_landmarks():
        result_json = process_attention(frame_data)
        result = json.loads(result_json)
        print(f"Status: {result['status']} | Left EAR: {result['leftEAR']} | Right EAR: {result['rightEAR']}")
