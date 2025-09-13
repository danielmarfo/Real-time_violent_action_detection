import cv2
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify
from tensorflow.keras.models import load_model
import os
import sys
import webbrowser

# Configuration should be similar to that of the trained model
MODEL_PATH = "MobBiLSTM_model_saved101 (3).keras" # path to trained model
CAMERA_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CLIP_LEN = 16
TARGET_SIZE = (64, 64)
PRED_THRESH = 0.5
ALARM_FILE = "mixkit-facility-alarm-sound-999.mp3" 

# Check model
if not os.path.exists(MODEL_PATH):
    print("ERROR: model file not found at:", MODEL_PATH, file=sys.stderr)
    sys.exit(1)

# Load model
print("Loading model from", MODEL_PATH)
model = load_model(MODEL_PATH)
print("Model loaded.")

app = Flask(__name__, static_folder='static', template_folder='templates')

# Shared frame and lock
output_frame = None
frame_lock = threading.Lock()

# For clip buffering
clip_buffer = deque(maxlen=CLIP_LEN)

# Alarm control variables
violence_detected = False
alarm_thread = None
pred_prob = None

alarm_lock = threading.Lock()

# Video capture thread
class VideoCaptureThread(threading.Thread):
    def __init__(self, src=0):
        super().__init__()
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.running = True

    def run(self):
        global output_frame, clip_buffer, violence_detected
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            clip_buffer.append(rgb.copy())

            pred_label = None
            if len(clip_buffer) == CLIP_LEN:
                clip_arr = np.zeros((1, CLIP_LEN, TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.float32)
                for i, f in enumerate(clip_buffer):
                    resized = cv2.resize(f, TARGET_SIZE)
                    clip_arr[0, i] = resized / 255.0

                try:
                    preds = model.predict(clip_arr, verbose=0)
                except Exception as e:
                    print("Prediction error:", e)
                    preds = np.array([[0.0]])

                if preds.ndim == 2 and preds.shape[1] == 1:
                    prob = float(preds[0,0])
                elif preds.ndim == 1:
                    prob = float(preds[0])
                else:
                    prob = float(preds.flatten()[0])

                pred_prob = prob
                pred_label = "Non-Violent" if prob >= PRED_THRESH else "Violent"

                # Alarm trigger logic
                if pred_label == "Violent":
                    violence_detected = True
                    # start_alarm()
                else:
                    violence_detected = False
                    # stop_alarm()
                    pass

            draw_frame = frame.copy()
            if pred_label is not None:
                color = (0,0,255) if pred_label == "Violent" else (0,255,0)
                cv2.rectangle(draw_frame, (5,5), (FRAME_WIDTH-5, FRAME_HEIGHT-5), color, 4)
                text = f"{pred_label} ({pred_prob:.2f})"
                cv2.putText(draw_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            else:
                cv2.putText(draw_frame, "Warming up... collecting frames", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            with frame_lock:
                output_frame = draw_frame

            time.sleep(0.01)

        self.cap.release()

    def stop(self):
        self.running = False

vc_thread = VideoCaptureThread(src=CAMERA_ID)
vc_thread.daemon = True
vc_thread.start()

@app.route("/")
def index():
    return render_template("index.html")

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                _, jpeg = cv2.imencode('.jpg', blank)
                frame_bytes = jpeg.tobytes()
            else:
                _, jpeg = cv2.imencode('.jpg', output_frame)
                frame_bytes = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/predict_from_file", methods=["POST"])
def predict_from_file():
    return jsonify({"status":"not_implemented","msg":"Use the desktop server to stream from webcam."})

@app.route("/status")
def status():
    global violence_detected, pred_prob
    return jsonify({"violence": violence_detected,"threat": pred_prob})

def shutdown():
    try:
        vc_thread.stop()
    except Exception:
        pass

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}/"
    print("Starting Flask app - opening web UI at", url)
    threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        app.run(host="127.0.0.1", port=port, threaded=True)
    finally:
        shutdown()


##different code for another model but just made changes to the configuration part.

# import cv2
# import time
# import threading
# import numpy as np
# from collections import deque
# from flask import Flask, render_template, Response, jsonify
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input
# from tensorflow.keras.models import Model
# import os
# import sys
# import webbrowser

# # path to trained model (expects 15x2048)
# MODEL_PATH = "mymodel101.keras"  
# CAMERA_ID = 0
# FRAME_WIDTH = 640
# FRAME_HEIGHT = 480
# SEQ_LENGTH = 15
# IMG_SIZE = 128
# PRED_THRESH = 0.5

# # Check model
# if not os.path.exists(MODEL_PATH):
#     print("ERROR: model file not found at:", MODEL_PATH, file=sys.stderr)
#     sys.exit(1)

# # Load LSTM violence model
# print("Loading model from", MODEL_PATH)
# model = load_model(MODEL_PATH)
# print("Model loaded.")

# # Load feature extractor (InceptionV3 -> 2048 features)
# base_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
# feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

# app = Flask(__name__, static_folder='static', template_folder='templates')

# # Shared frame and lock
# output_frame = None
# frame_lock = threading.Lock()

# # Buffers
# feature_buffer = deque(maxlen=SEQ_LENGTH)

# # Alarm control variables
# violence_detected = False
# pred_prob = None


# def extract_features(frame):
#     """Preprocess frame and extract CNN features (2048-dim)."""
#     frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = np.expand_dims(frame, axis=0)
#     frame = preprocess_input(frame)
#     features = feature_extractor.predict(frame, verbose=0)
#     return features.squeeze()  # (2048,)


# # Video capture thread
# class VideoCaptureThread(threading.Thread):
#     def __init__(self, src=0):
#         super().__init__()
#         self.cap = cv2.VideoCapture(src)
#         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
#         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
#         self.running = True

#     def run(self):
#         global output_frame, feature_buffer, violence_detected, pred_prob
#         while self.running:
#             ret, frame = self.cap.read()
#             if not ret:
#                 time.sleep(0.01)
#                 continue

#             # Step 1: extract CNN features
#             features = extract_features(frame)
#             feature_buffer.append(features)

#             pred_label = None
#             if len(feature_buffer) == SEQ_LENGTH:
#                 # Step 2: prepare sequence (1, 15, 2048)
#                 input_seq = np.expand_dims(np.array(feature_buffer), axis=0)

#                 try:
#                     preds = model.predict(input_seq, verbose=0)
#                 except Exception as e:
#                     print("Prediction error:", e)
#                     preds = np.array([[0.0]])

#                 prob = float(preds[0][0])
#                 pred_prob = prob
#                 pred_label = "Non-Violent" if prob >= PRED_THRESH else "Violent"

#                 # Set flag
#                 violence_detected = pred_label == "Violent"

#             # Step 3: Draw overlay
#             draw_frame = frame.copy()
#             if pred_label is not None:
#                 color = (0, 0, 255) if pred_label == "Violent" else (0, 255, 0)
#                 cv2.rectangle(draw_frame, (5, 5), (FRAME_WIDTH - 5, FRAME_HEIGHT - 5), color, 4)
#                 text = f"{pred_label} ({pred_prob:.2f})"
#                 cv2.putText(draw_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
#             else:
#                 cv2.putText(draw_frame, "Collecting frames...", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

#             with frame_lock:
#                 output_frame = draw_frame

#             time.sleep(0.01)

#         self.cap.release()

#     def stop(self):
#         self.running = False


# vc_thread = VideoCaptureThread(src=CAMERA_ID)
# vc_thread.daemon = True
# vc_thread.start()


# @app.route("/")
# def index():
#     return render_template("index.html")


# def generate_frames():
#     global output_frame
#     while True:
#         with frame_lock:
#             if output_frame is None:
#                 blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
#                 _, jpeg = cv2.imencode('.jpg', blank)
#                 frame_bytes = jpeg.tobytes()
#             else:
#                 _, jpeg = cv2.imencode('.jpg', output_frame)
#                 frame_bytes = jpeg.tobytes()

#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
#         time.sleep(0.03)


# @app.route("/video_feed")
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype="multipart/x-mixed-replace; boundary=frame")


# @app.route("/status")
# def status():
#     global violence_detected, pred_prob
#     return jsonify({"violence": violence_detected, "threat": pred_prob})


# def shutdown():
#     try:
#         vc_thread.stop()
#     except Exception:
#         pass


# if __name__ == "__main__":
#     port = 5000
#     url = f"http://127.0.0.1:{port}/"
#     print("Starting Flask app - opening web UI at", url)
#     threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
#     try:
#         app.run(host="127.0.0.1", port=port, threaded=True)
#     finally:
#         shutdown()
