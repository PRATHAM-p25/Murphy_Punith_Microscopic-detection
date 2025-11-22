import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time
import requests
from pymongo import MongoClient
import gridfs
from datetime import datetime
import bcrypt

st.set_page_config(layout="wide", page_title="Microscopy Detection App")

# -------------------- MongoDB Connection ----------------------
def get_mongo_uri():
    if "mongo" in st.secrets and "uri" in st.secrets["mongo"]:
        return st.secrets["mongo"]["uri"]
    return None

MONGO_URI = get_mongo_uri()
client = MongoClient(MONGO_URI)
db = client["microscopy_db"]
users_col = db["users"]
detections_col = db["detections"]
fs = gridfs.GridFS(db)

# -------------------- Authentication Helpers ------------------
def signup_user(username, password):
    existing = users_col.find_one({"username": username})
    if existing:
        return False, "Username already exists."

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    users_col.insert_one({"username": username, "password": hashed})
    return True, "Account created successfully!"

def login_user(username, password):
    user = users_col.find_one({"username": username})
    if not user:
        return False, "Invalid username."

    if bcrypt.checkpw(password.encode(), user["password"]):
        return True, "Login successful!"
    return False, "Incorrect password."

# -------------------- UI Header -------------------------
st.title("🔬 Microscopy Detection App with Authentication")

# -------------------- Session State ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# -------------------- Login / Signup UI -------------------
if not st.session_state.logged_in:

    tab1, tab2 = st.tabs(["🔐 Login", "🆕 Signup"])

    with tab1:
        st.subheader("Login")

        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            success, msg = login_user(login_username, login_password)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = login_username
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab2:
        st.subheader("Create Account")

        signup_username = st.text_input("New Username")
        signup_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            success, msg = signup_user(signup_username, signup_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)

    st.stop()

# -------------------- Logout Button -----------------------
st.sidebar.success(f"Logged in as: {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# -------------------- Model Load --------------------------
MODEL_LOCAL_PATH = "best.onnx"
MODEL_IMG_SIZE = 1024

@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_LOCAL_PATH)

# -------------------- Detection UI -------------------------
st.header("🔍 Run Microscopy Detection")

conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.25)
uploaded = st.file_uploader("Upload Image", ["jpg", "png", "jpeg", "tif", "tiff"])

if uploaded:
    pil_img = Image.open(uploaded).convert("RGB")
    st.image(pil_img, caption="Input Image")

    if st.button("Run Detection"):
        results = model.predict(np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf)

        # Draw results
        draw = ImageDraw.Draw(pil_img)
        counts = {}
        for r in results:
            for box in r.boxes:
                x1,y1,x2,y2 = box.xyxy[0].tolist()
                cls_name = model.names[int(box.cls)]
                score = float(box.conf)

                counts[cls_name] = counts.get(cls_name, 0) + 1

                draw.rectangle([x1,y1,x2,y2], outline="red", width=2)
                draw.text((x1,y1), f"{cls_name} {score:.2f}", fill="yellow")

        st.image(pil_img, caption="Detections")

        # Store in DB --------------------
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        file_id = fs.put(img_bytes, filename=f"{st.session_state.username}_{int(time.time())}.png")

        detections_col.insert_one({
            "user": st.session_state.username,
            "timestamp": datetime.utcnow(),
            "counts": counts,
            "file_id": file_id
        })

        st.success("Saved to MongoDB!")

