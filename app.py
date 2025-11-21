# app.py
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import os
import time
from datetime import datetime

# MongoDB
from pymongo import MongoClient, errors
import gridfs

# ----------------- Config -----------------
# If you put best.onnx inside repo, keep this as "best.onnx"
MODEL_LOCAL_PATH = "best.onnx"

# If you want to have a sample image pre-bundled in your environment,
# here's the path from your session that was provided earlier:
SAMPLE_IMAGE_PATH = "/mnt/data/fac1e28d-0cba-4b9a-9e2e-6da55e713604.png"

# Model image size (use same as exported)
MODEL_IMG_SIZE = 1024

# ------------------------------------------

st.set_page_config(layout="wide", page_title="Microscopy Detector + MongoDB (GridFS)")
st.title("Microscopy Detector — save detections to MongoDB (GridFS)")

# ----------------- MongoDB Connection Helper -----------------
@st.cache_resource
def get_mongo_client(uri):
    # return MongoClient or raise
    return MongoClient(uri, serverSelectionTimeoutMS=5000)

def get_db_and_fs(client, db_name="microscopy_db"):
    db = client[db_name]
    fs = gridfs.GridFS(db)
    coll = db["detections"]
    return db, fs, coll

# ----------------- Load model -----------------
@st.cache_resource
def load_model(path):
    return YOLO(path)  # this will accept .onnx exported by Ultralytics

# ----------------- Draw & postprocess helper -----------------
def draw_predictions(pil_img, results, conf_thresh=0.25):
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    counts = {}
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            # ultralytics Boxes object metadata access
            # support both .conf/.cls arrays and older attributes
            try:
                score = float(box.conf[0])
            except Exception:
                score = float(getattr(box, "confidence", 0.0))
            try:
                cls = int(box.cls[0])
            except Exception:
                cls = int(getattr(box, "cls", 0))
            if score < conf_thresh:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[cls] if cls < len(model.names) else str(cls)
            counts[label] = counts.get(label, 0) + 1

            # draw box + label
            bbox = draw.textbbox((0,0), f"{label} {score:.2f}", font=font)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
            draw.rectangle([x1, y1-th, x1+tw, y1], fill=(255,0,0))
            draw.text((x1, y1-th), f"{label} {score:.2f}", fill=(255,255,255), font=font)

    return pil_img, counts

# ----------------- UI: Mongo URI (supports st.secrets or manual entry) -----------------
st.sidebar.header("Database (MongoDB Atlas)")

mongo_uri = None
secret_present = False
try:
    # prefer secrets if available
    mongo_uri = st.secrets["mongo"]["uri"]
    secret_present = True
except Exception:
    secret_present = False

if secret_present:
    st.sidebar.success("Mongo URI loaded from Streamlit secrets.")
    # show a masked display
    st.sidebar.text("Using Mongo URI from secrets.")
else:
    st.sidebar.info("Enter your MongoDB Atlas connection string (mongodb+srv://...)")
    mongo_uri = st.sidebar.text_input("MongoDB URI", value="", placeholder="mongodb+srv://<user>:<pass>@cluster0....")
    if not mongo_uri:
        st.sidebar.warning("No Mongo URI provided. DB features disabled until you enter URI.")

use_db = bool(mongo_uri and mongo_uri.strip() != "")

# ----------------- Load model -----------------
try:
    with st.spinner("Loading model..."):
        model = load_model(MODEL_LOCAL_PATH)
    st.sidebar.success("Model loaded.")
except Exception as e:
    st.sidebar.error(f"Failed loading model: {e}")
    st.stop()

# ----------------- Main UI -----------------
st.write("Upload an image or use the sample image to run detection and store results in MongoDB (GridFS).")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload microscope image", type=["png","jpg","jpeg","tif","tiff"])
    use_sample = st.checkbox("Use sample image", value= False)
    if use_sample:
        if os.path.exists(SAMPLE_IMAGE_PATH):
            uploaded = open(SAMPLE_IMAGE_PATH, "rb")
        else:
            st.warning("Sample image not found at SAMPLE_IMAGE_PATH.")
    conf_thresh = st.slider("Confidence threshold", 0.0, 1.0, 0.25)
    run_btn = st.button("Run inference & save to DB" if use_db else "Run inference (DB disabled)")

with col2:
    st.header("Recent DB entries")
    show_last_n = st.number_input("Show last N entries", min_value=1, max_value=20, value=5)
    refresh_db = st.button("Refresh list")

# ----------------- Connect to DB if requested -----------------
client = None; db = None; fs = None; collection = None
if use_db:
    try:
        client = get_mongo_client(mongo_uri)
        # test server selection
        client.admin.command('ping')
        db, fs, collection = get_db_and_fs(client)
        st.sidebar.success("Connected to MongoDB Atlas.")
    except errors.ServerSelectionTimeoutError as e:
        st.sidebar.error("Could not connect to MongoDB Atlas. Check URI/network. " + str(e))
        use_db = False
    except Exception as e:
        st.sidebar.error("Mongo connection error: " + str(e))
        use_db = False

# ----------------- Run inference flow -----------------
if run_btn:
    if uploaded is None:
        st.warning("Please upload an image or choose the sample image.")
    else:
        # read image bytes
        if isinstance(uploaded, io.IOBase):
            # file object (sample)
            img_bytes = uploaded.read()
        elif hasattr(uploaded, "read"):
            img_bytes = uploaded.read()
        else:
            # in case open() gave a path
            try:
                img_bytes = open(uploaded, "rb").read()
            except Exception:
                st.error("Failed to read uploaded/sample image.")
                img_bytes = None

        if img_bytes:
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            st.image(pil_img, caption="Input image", width=400)

            # run inference
            t0 = time.time()
            try:
                results = model.predict(source=np.array(pil_img), imgsz=MODEL_IMG_SIZE, conf=conf_thresh, verbose=False)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                results = []
            t1 = time.time()

            pil_out, counts = draw_predictions(pil_img.copy(), results, conf_thresh=conf_thresh)
            st.image(pil_out, caption="Detections", width=400)
            st.write("Counts:", counts)
            st.write(f"Inference time: {t1-t0:.2f}s")

            # convert output image to bytes (PNG)
            buf = io.BytesIO()
            pil_out.save(buf, format="PNG")
            buf.seek(0)
            image_bytes = buf.getvalue()

            # ----------------- Save to MongoDB (GridFS + metadata) -----------------
            if use_db and collection is not None and fs is not None:
                try:
                    ts = datetime.utcnow()
                    # store binary image in GridFS
                    file_id = fs.put(image_bytes,
                                     filename=f"detection_{ts.isoformat()}.png",
                                     contentType="image/png",
                                     metadata={"timestamp": ts, "model": os.path.basename(MODEL_LOCAL_PATH), "img_size": MODEL_IMG_SIZE})

                    # metadata document
                    doc = {
                        "timestamp": ts,
                        "gridfs_id": file_id,
                        "counts": counts,
                        "model": os.path.basename(MODEL_LOCAL_PATH),
                        "img_size": MODEL_IMG_SIZE
                    }
                    insertion_result = collection.insert_one(doc)
                    st.success(f"Saved detection to DB. doc_id={insertion_result.inserted_id}, gridfs_id={file_id}")
                except Exception as e:
                    st.error(f"Failed to save to MongoDB: {e}")
            else:
                if not use_db:
                    st.info("DB not configured — image not saved. To enable, provide a valid MongoDB URI in the sidebar.")

# ----------------- Show last N entries (GridFS retrieval) -----------------
def show_last_entries(n=5):
    if not use_db or collection is None or fs is None:
        st.info("DB not configured or connection failed.")
        return

    try:
        docs = list(collection.find().sort("timestamp", -1).limit(n))
        if not docs:
            st.info("No entries found in DB.")
            return
        for d in docs:
            st.markdown("---")
            st.write(f"Document ID: {d.get('_id')}")
            st.write(f"Timestamp (UTC): {d.get('timestamp')}")
            st.write("Counts:", d.get("counts", {}))
            gridfs_id = d.get("gridfs_id")
            try:
                file_bytes = fs.get(gridfs_id).read()
                st.image(Image.open(io.BytesIO(file_bytes)), width=400)
            except Exception as e:
                st.error(f"Could not read GridFS file {gridfs_id}: {e}")
    except Exception as e:
        st.error("Error reading DB entries: " + str(e))

if refresh_db:
    show_last_entries(show_last_n)

# show by default at page load (when DB configured)
if use_db and not refresh_db:
    show_last_entries(show_last_n)
