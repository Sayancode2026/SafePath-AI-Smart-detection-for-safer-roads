import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from ultralytics import YOLO

# ─────────────────────────────────────────────
# Streamlit Page Configuration
st.set_page_config(
    page_title="Realtime Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Load YOLO model from local path
MODEL_LOCAL_PATH = Path(r"C:\Users\SAYAN\RoadDamageDetection\models\best.pt")

cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

# ─────────────────────────────────────────────
# Class names for detection
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

# Detection NamedTuple for predictions
class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# ─────────────────────────────────────────────
# Realtime Detection UI
st.title("Road Damage Detection - Realtime")
st.write("Detect road damage in real time using your webcam. Useful for on-site live monitoring.")

# Confidence slider (move this above the callback)
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Adjust the threshold to fine-tune detection sensitivity.")

# Queue to hold results from thread
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# Frame callback function for Streamlit WebRTC
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori, w_ori = image.shape[:2]

    image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)

    # Collect detections
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        result_queue.put(detections)

    # Draw results and resize back
    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    return av.VideoFrame.from_ndarray(_image, format="bgr24")

# ─────────────────────────────────────────────
# WebRTC Streamer Setup
webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280, "min": 800},
        },
        "audio": False
    },
    async_processing=True,
)

# ─────────────────────────────────────────────
# Optional Prediction Table Viewer
st.divider()
if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
