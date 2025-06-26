import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

from ultralytics import YOLO
from PIL import Image
from io import BytesIO

# ────────────────────────────────
st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────
logger = logging.getLogger(__name__)
HERE = Path(__file__).parent
ROOT = HERE.parent

# Load model from local path only
MODEL_LOCAL_PATH = Path(r"C:\Users\SAYAN\RoadDamageDetection\models\best.pt")  # 👈 Replace with your path


# Load the model into session cache
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

# ────────────────────────────────
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# ────────────────────────────────
st.title("Road Damage Detection - Image")
st.write("Upload an image to detect road damage types.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower threshold to detect more damage, or raise it to reduce false positives.")

if image_file is not None:
    image = Image.open(image_file)
    col1, col2 = st.columns(2)

    _image = np.array(image)
    h_ori, w_ori = _image.shape[:2]
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)

    results = net.predict(image_resized, conf=score_threshold)

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

    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    with col1:
        st.write("#### Original Image")
        st.image(_image)

    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
