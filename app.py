# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os
import requests
import io
import time
import base64
from datetime import datetime
from pymongo import MongoClient

# ----------------------------
# MONGO CONNECTION (Step 3)
# ----------------------------
mongo_uri = st.secrets["mongo"]["uri"]
client = MongoClient(mongo_uri)
db = client["microscopy_db"]
collection = db["detections"]

# ----------------------------
# STREAMLIT CONFIG
# ----------------------------
st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")
st.title("Microscopy Detector (ONNX via Ultralytics)")

# ----------------------------
# MODEL SETTINGS
# ----------------------------
MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""   # keep empty, you uploaded best.onnx to GitHub
MODEL_IMG_SIZE = 1024
CONF_THRESH = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25)

# ----------------------------
# MODEL LOAD
# ----------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

with st.spinner("Loading model..."):
    model = load_model(MODEL_LOCAL_PATH)
st.success("Model loaded successfully!")

# ----------------------------
# DRAW PREDICTIONS
# ----------------------------
def draw_predictions(pil_img, results, conf_thresh=0.25):
    draw = ImageDraw.Draw(pil_img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()

    counts = {}

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            score = float(box.conf[0])
            cls = int(box.cls[0])

            if score < conf_thresh:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[cls]

            counts[label] = counts.get(label, 0) + 1

            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            text = f"{label} {score:.2f}"

            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

            draw.rectangle([x1, y1-th, x1+tw, y1], fill="red")
            draw.text((x1, y1-th), text, fill="white", font=font)

    return pil_img, counts

# ----------------------------
# IMAGE INPUT
# ----------------------------
uploaded = st.file_uploader("Upload microscope image", type=["png", "jpg", "jpeg"])
camera = st.camera_input("Or take a photo (browser support only)")

if not uploaded and not camera:
    st.info("Please upload an image or capture using camera.")
    st.stop()

img_bytes = uploaded.read() if uploaded else camera.read()
pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

st.image(pil_img, caption="Input Image", width=350)

# ----------------------------
# RUN INFERENCE
# ----------------------------
if st.button("Run Inference"):
    start = time.time()

    results = model.predict(
        source=np.array(pil_img),
        imgsz=MODEL_IMG_SIZE,
        conf=CONF_THRESH,
        verbose=False
    )

    pil_out, counts = draw_predictions(pil_img.copy(), results, CONF_THRESH)

    st.image(pil_out, caption="Detections", use_column_width=True)
    st.write("Counts:", counts)
    st.success(f"Inference completed in {time.time() - start:.2f}s")

    # -----------------------------------------
    # STEP 4 — Convert DETECTED image to Base64
    # -----------------------------------------
    buffer_out = io.BytesIO()
    pil_out.save(buffer_out, format="PNG")
    detected_base64 = base64.b64encode(buffer_out.getvalue()).decode("utf-8")

    # Also save ORIGINAL INPUT image
    buffer_in = io.BytesIO()
    pil_img.save(buffer_in, format="PNG")
    input_base64 = base64.b64encode(buffer_in.getvalue()).decode("utf-8")

    # -----------------------------------------
    # STEP 5 — Prepare MongoDB Document
    # -----------------------------------------
    document = {
        "timestamp": datetime.utcnow(),
        "counts": counts,
        "input_image": input_base64,
        "detected_image": detected_base64
    }

    # -----------------------------------------
    # STEP 6 — Insert into MongoDB
    # -----------------------------------------
    insertion_result = collection.insert_one(document)

    st.success(f"Saved to MongoDB! Document ID: {insertion_result.inserted_id}")
