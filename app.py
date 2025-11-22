# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, os, time, base64
import requests
from pymongo import MongoClient, errors
import gridfs
from datetime import datetime
import hashlib, binascii, secrets

st.set_page_config(layout="wide", page_title="Microscopy ONNX Demo")

# ---------------------------
# Configuration
# ---------------------------
MODEL_LOCAL_PATH = "best.onnx"
GDRIVE_FILE_ID = ""       # optional
MODEL_IMG_SIZE = 1024
DEFAULT_CONF = 0.25

# ---------------------------
# Utilities: Mongo URI
# ---------------------------
def get_mongo_uri():
    # prefer Streamlit secrets
    try:
        mongo_conf = st.secrets.get("mongo") if st.secrets else None
        if mongo_conf and "uri" in mongo_conf:
            return mongo_conf["uri"]
    except Exception:
        pass
    # fallback to environment variable
    return os.environ.get("MONGO_URI")

MONGO_URI = get_mongo_uri()
USE_DB = bool(MONGO_URI)

# ---------------------------
# Download model helper
# ---------------------------
def download_from_gdrive(file_id, dest):
    if os.path.exists(dest):
        return dest
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return dest

# ---------------------------
# Model loading (cached)
# ---------------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# ---------------------------
# Text size helper (robust)
# ---------------------------
def get_text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return w, h
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            try:
                return font.getsize(text)
            except Exception:
                return (len(text)*6, 11)

# ---------------------------
# Draw preds
# ---------------------------
def draw_predictions(pil_img, results, conf_thresh=0.25, model_names=None):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    counts = {}
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            try:
                score = float(box.conf[0]) if hasattr(box, "conf") else float(box.confidence)
            except Exception:
                score = float(getattr(box, "confidence", 0.0))
            try:
                cls = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            except Exception:
                cls = int(getattr(box, "class_id", 0))
            if score < conf_thresh:
                continue
            try:
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
            except Exception:
                coords = getattr(box, "xyxy", None)
                if coords is not None:
                    x1, y1, x2, y2 = coords[0].tolist()
                else:
                    continue
            label = (model_names[cls] if model_names and cls < len(model_names) else str(cls))
            counts[label] = counts.get(label, 0) + 1
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            text = f"{label} {score:.2f}"
            tw, th = get_text_size(draw, text, font)
            ty1 = max(0, y1 - th)
            draw.rectangle([x1, ty1, x1 + tw, y1], fill=(255,0,0))
            draw.text((x1, ty1), text, fill=(255,255,255), font=font)
    return pil_img, counts

# ---------------------------
# Password hashing helpers (PBKDF2-HMAC-SHA256)
# ---------------------------
def hash_password(password: str, salt: bytes = None):
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 150000)
    return binascii.hexlify(dk).decode("ascii"), binascii.hexlify(salt).decode("ascii")

def verify_password(stored_hash_hex: str, stored_salt_hex: str, provided_password: str) -> bool:
    salt = binascii.unhexlify(stored_salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", provided_password.encode("utf-8"), salt, 150000)
    return binascii.hexlify(dk).decode("ascii") == stored_hash_hex

# ---------------------------
# MongoDB connection
# ---------------------------
client = None
db = None
fs = None
users_col = None
collection = None
db_error_msg = None
if USE_DB:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # trigger connection/auth check
        db = client["microscopy_db"]
        fs = gridfs.GridFS(db)
        collection = db["detections"]
        users_col = db["users"]
        # ensure index on username
        users_col.create_index("username", unique=True)
    except errors.OperationFailure as e:
        db_error_msg = ("MongoDB auth failure. Check username/password and user privileges.")
    except errors.ServerSelectionTimeoutError as e:
        db_error_msg = ("Could not connect to MongoDB Atlas. Possibly IP not whitelisted. "
                        "Add 0.0.0.0/0 temporarily or add Streamlit Cloud IPs.")
    except Exception as e:
        db_error_msg = f"MongoDB connection error: {e}"

# ---------------------------
# Download model (optional)
# ---------------------------
if GDRIVE_FILE_ID:
    try:
        download_from_gdrive(GDRIVE_FILE_ID, MODEL_LOCAL_PATH)
    except Exception as e:
        st.error(f"Downloading model failed: {e}")

# ---------------------------
# Load model
# ---------------------------
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_LOCAL_PATH)
        model_names = getattr(model, "names", None)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# ---------------------------
# Authentication UI state
# ---------------------------
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = None  # "signup" or "signin" or None

if "user" not in st.session_state:
    st.session_state.user = None  # username if logged in

# Top row: big Sign Up / Sign In buttons
c1, c2, c3 = st.columns([1,2,1])
with c1:
    if st.button("Sign up"):
        st.session_state.auth_mode = "signup"
with c2:
    st.markdown("<h1 style='text-align:center;margin:0;padding:0;'>Microscopy Detector</h1>", unsafe_allow_html=True)
with c3:
    if st.button("Sign in"):
        st.session_state.auth_mode = "signin"

# If logged in, show username + logout
if st.session_state.user:
    st.sidebar.success(f"Signed in as: {st.session_state.user}")
    if st.sidebar.button("Log out"):
        st.session_state.user = None
        st.session_state.auth_mode = None
        st.experimental_rerun()

# ---------------------------
# Authentication forms
# ---------------------------
def show_signup_form():
    st.subheader("Create an account")
    with st.form("signup_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        password2 = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account")
        if submitted:
            if not username or not password:
                st.error("Provide username and password.")
                return
            if password != password2:
                st.error("Passwords do not match.")
                return
            if not USE_DB:
                st.error("DB not configured. Add Mongo URI to secrets or environment.")
                return
            if db_error_msg:
                st.error(db_error_msg)
                return
            # check existing
            if users_col.find_one({"username": username}):
                st.error("Username already exists. Choose another.")
                return
            # hash and store
            pw_hash, salt_hex = hash_password(password)
            user_doc = {
                "username": username,
                "pw_hash": pw_hash,
                "salt": salt_hex,
                "created_at": datetime.utcnow()
            }
            try:
                users_col.insert_one(user_doc)
                st.success("Account created — you are now signed in.")
                st.session_state.user = username
                st.session_state.auth_mode = None
            except Exception as e:
                st.error(f"Failed to create account: {e}")

def show_signin_form():
    st.subheader("Sign in")
    with st.form("signin_form"):
        username = st.text_input("Username", key="signin_user")
        password = st.text_input("Password", type="password", key="signin_pw")
        submitted = st.form_submit_button("Sign in")
        if submitted:
            if not username or not password:
                st.error("Provide username and password.")
                return
            if not USE_DB:
                st.error("DB not configured. Add Mongo URI to secrets or environment.")
                return
            if db_error_msg:
                st.error(db_error_msg)
                return
            # find user
            user = users_col.find_one({"username": username})
            if not user:
                st.error("Invalid username or password.")
                return
            stored_hash = user.get("pw_hash")
            stored_salt = user.get("salt")
            if not stored_hash or not stored_salt:
                st.error("User record invalid. Contact admin.")
                return
            ok = verify_password(stored_hash, stored_salt, password)
            if not ok:
                st.error("Invalid username or password.")
                return
            st.success("Signed in.")
            st.session_state.user = username
            st.session_state.auth_mode = None

# Show auth form based on mode (but only if not already logged in)
if not st.session_state.user:
    if st.session_state.auth_mode == "signup":
        show_signup_form()
    elif st.session_state.auth_mode == "signin":
        show_signin_form()
    else:
        st.write("")  # nothing

# ---------------------------
# Detection panel - visible only when signed in
# ---------------------------
if not st.session_state.user:
    st.info("Please Sign up or Sign in to use the detector and store results.")
    st.stop()

# Main detection UI (signed-in user)
col1, col2 = st.columns([1, 1.2])
with col1:
    st.header("Run Detection")
    conf = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONF)
    uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
    camera = st.camera_input("Or take a picture (Chromium browsers)")

    if uploaded is None and camera is None:
        st.info("Upload an image or use the camera.")
    else:
        img_bytes = uploaded.read() if uploaded else camera.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        st.image(pil_img, caption="Input image", width=400)

        if st.button("Run inference"):
            start = time.time()
            try:
                results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf, verbose=False)
            except Exception as e:
                st.error(f"Model inference failed: {e}")
                st.stop()

            pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf, model_names=model_names)
            st.image(pil_out, caption="Detections", use_column_width=True)
            st.write("Counts:", counts)
            st.success(f"Inference done in {time.time()-start:.2f}s")

            # Save to DB (GridFS + document) if DB configured
            if not USE_DB:
                st.info("Mongo URI not provided. Skipping DB save.")
            else:
                if db_error_msg:
                    st.error(db_error_msg)
                else:
                    try:
                        buf = io.BytesIO()
                        pil_out.save(buf, format="PNG")
                        img_bytes_out = buf.getvalue()
                        file_id = fs.put(img_bytes_out, filename=f"det_{int(time.time())}.png", contentType="image/png")
                        document = {
                            "timestamp": datetime.utcnow(),
                            "counts": counts,
                            "model": MODEL_LOCAL_PATH,
                            "img_gridfs_id": file_id,
                            "user": st.session_state.user,
                        }
                        insertion_result = collection.insert_one(document)
                        st.success(f"Saved detection to DB. doc_id: {insertion_result.inserted_id}")
                    except Exception as e:
                        st.error(f"Failed to save to DB: {e}")

with col2:
    st.header("History (your last 5 detections)")
    if not USE_DB:
        st.info("DB not configured. Set your Mongo URI to enable saved history.")
    elif db_error_msg:
        st.error(db_error_msg)
    else:
        try:
            docs = list(collection.find({"user": st.session_state.user}).sort("timestamp", -1).limit(5))
            if not docs:
                st.info("No saved detections yet.")
            else:
                for doc in docs:
                    st.write(f"Time: {doc.get('timestamp')}, Counts: {doc.get('counts')}")
                    gfid = doc.get("img_gridfs_id")
                    if gfid:
                        try:
                            grid_out = fs.get(gfid)
                            data = grid_out.read()
                            img = Image.open(io.BytesIO(data))
                            st.image(img, width=300)
                        except Exception as e:
                            st.text(f"Could not read GridFS file {gfid}: {e}")
        except Exception as e:
            st.error(f"Failed to load history: {e}")
