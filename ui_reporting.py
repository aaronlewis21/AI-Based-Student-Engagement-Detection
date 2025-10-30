import cv2
import mediapipe as mp
from processing_model import process_attention
from input_capture import capture_landmarks
import json

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def display_attention():
    cap = cv2.VideoCapture(0)
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            attentive = False
            if results.multi_face_landmarks:
                data = capture_landmarks()
                result = json.loads(process_attention(data))
                attentive = result.get("attentive", False)
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, face_landmarks,
                                              mp_face_mesh.FACEMESH_CONTOURS)
            text = "ATTENTIVE" if attentive else "NOT ATTENTIVE"
            color = (0, 255, 0) if attentive else (0, 0, 255)
            cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.imshow("Student Monitoring", frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    display_attention()
