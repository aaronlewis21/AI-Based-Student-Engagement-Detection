import json
import pandas as pd
from datetime import datetime

def summarize_attention(json_input, log_file="attention_log.csv"):
    data = json.loads(json_input)
    status = "ATTENTIVE" if data.get("attentive") else "NOT ATTENTIVE"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([[timestamp, status, data.get("leftEAR"), data.get("rightEAR")]],
                      columns=["Timestamp", "Status", "LeftEAR", "RightEAR"])
    df.to_csv(log_file, mode='a', header=False, index=False)

    return json.dumps({"timestamp": timestamp, "status": status})

if __name__ == "__main__":
    from processing_model import process_attention
    from input_capture import capture_landmarks
    print(summarize_attention(process_attention(capture_landmarks())))
