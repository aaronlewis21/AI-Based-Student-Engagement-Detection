# AI-Based-Student-Engagement-Detection
This project detects student attentiveness in real time using a webcam feed.   It uses facial landmark tracking (MediaPipe) and Eye Aspect Ratio (EAR) calculations to determine if a student is attentive or not.
---

## 📂 Project Structure

| File | Description |
|------|--------------|
| `input_capture.py` | Captures live webcam frames and extracts facial landmarks |
| `processing_model.py` | Calculates EAR and head deviation to detect attentiveness |
| `post_processing.py` | Logs attention data to a CSV file |
| `ui_reporting.py` | Displays a real-time annotated feed with attention status |
| `requirements.txt` | Dependencies needed to run the project |

---

## 🧠 Working

1. `input_capture.py` uses **MediaPipe FaceMesh** to capture 468 facial landmarks.
2. `processing_model.py` computes the **Eye Aspect Ratio (EAR)** and **head deviation**.
3. If both eyes are closed or the head is turned, the student is marked as **NOT ATTENTIVE**.
4. Results are stored by `post_processing.py` and displayed through `ui_reporting.py`.

---

## 🚀 How to Run

```bash
git clone https://github.com/<your-username>/Student-Attention-Monitoring.git
cd Student-Attention-Monitoring
pip install -r requirements.txt
python ui_reporting.py
