import cv2
import mediapipe as mp
import json

mp_face_mesh = mp.solutions.face_mesh

def capture_landmarks():
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera not available.")
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        landmarks = []
        if results.multi_face_landmarks:
            for lm in results.multi_face_landmarks[0].landmark:
                landmarks.append({"x": lm.x, "y": lm.y})

        data = {"height": h, "width": w, "landmarks": landmarks}
        yield json.dumps(data)   # use generator instead of return

        # Exit manually
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    face_mesh.close()

if __name__ == "__main__":
    for frame_data in capture_landmarks():
        print(frame_data)
