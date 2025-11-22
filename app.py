import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64
import requests
from pymongo import MongoClient
import gridfs
from datetime import datetime
import hashlib

st.set_page_config(layout="wide", page_title="Microscopy Detector")

# ------------------ MONGO CONNECTION ------------------
mongo_uri = st.secrets["mongo"]["uri"] if "mongo" in st.secrets else os.getenv("MONGO_URI")
client = MongoClient(mongo_uri)
db = client["microscopy_db"]
users = db["users"]
fs = gridfs.GridFS(db)
detections = db["detections"]

# ------------------ PASSWORD HASHING ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ AUTH FUNCTIONS ------------------
def signup(username, password):
    if users.find_one({"username": username}):
        return False, "Username already exists."
    users.insert_one({"username": username, "password": hash_password(password)})
    return True, "Signup successful!"

def login(username, password):
    user = users.find_one({"username": username})
    if not user:
        return False, "User not found."
    if user["password"] != hash_password(password):
        return False, "Incorrect password."
    return True, "Login successful!"

# ------------------ MODEL LOADING ------------------
MODEL_LOCAL_PATH = "best.onnx"
MODEL_IMG_SIZE = 1024

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_LOCAL_PATH)
model_names = model.names

# ------------------ DRAW PREDICTIONS ------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except:
        return (len(text)*6, 12)

def draw_predictions(pil_img, results, conf_thresh=0.25):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()
    counts = {}

    for r in results:
        boxes = getattr(r, "boxes", None)
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
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            draw.rectangle([x1, y1 - th, x1 + tw, y1], fill="red")
            draw.text((x1, y1 - th), text, fill="white", font=font)
    return pil_img, counts

# ------------------ AUTH UI ------------------
if "user" not in st.session_state:
    st.session_state.user = None

st.title("Microscopy Detector (ONNX + MongoDB + Auth)")

# Show auth forms when NOT logged in
if st.session_state.user is None:

    tab1, tab2 = st.tabs(["🔑 Sign In", "🆕 Sign Up"])

    # ------------------ SIGN IN ------------------
    with tab1:
        st.subheader("Login to Continue")
        login_user = st.text_input("Username")
        login_pass = st.text_input("Password", type="password")
        if st.button("Sign In"):
            ok, msg = login(login_user, login_pass)
            if ok:
                st.session_state.user = login_user
                st.success(msg)
                st.experimental_rerun()
            else:
                st.error(msg)

    # ------------------ SIGN UP ------------------
    with tab2:
        st.subheader("Create New Account")
        signup_user = st.text_input("New Username")
        signup_pass = st.text_input("New Password", type="password")
        if st.button("Sign Up"):
            ok, msg = signup(signup_user, signup_pass)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.stop()

# ------------------ LOGGED IN UI ------------------
st.success(f"Logged in as **{st.session_state.user}**")

if st.button("Logout"):
    st.session_state.user = None
    st.experimental_rerun()

st.header("Run Detection")
conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25)

uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Input image", width=400)

    if st.button("Run Inference"):
        results = model.predict(np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf)

        out_img, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf)
        st.image(out_img, caption="Detections", use_column_width=True)
        st.write("Counts:", counts)

        buf = io.BytesIO()
        out_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        file_id = fs.put(img_bytes, filename=f"detect_{int(time.time())}.png")

        detections.insert_one({
            "username": st.session_state.user,
            "timestamp": datetime.utcnow(),
            "counts": counts,
            "img_gridfs_id": file_id
        })

        st.success("Saved to MongoDB successfully.")
