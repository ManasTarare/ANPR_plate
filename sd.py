import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import Counter
from PIL import Image
import torch
import os

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="ANPR Image System", layout="wide")
st.title("📸 Automatic Number Plate Recognition (Indian Plates)")

# ===============================
# MODEL PATH
# ===============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    try:
        use_gpu = torch.cuda.is_available()
        model = YOLO(MODEL_PATH)
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        device_name = "GPU" if use_gpu else "CPU"
        return model, reader, device_name
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.stop()

model, reader, device = load_models()
st.write(f"Running on: {device}")

# ===============================
# OCR CONFUSION FIX
# ===============================
OCR_FIX = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6'
}

# ===============================
# REGEX (SOFT VALIDATION)
# ===============================
SOFT_INDIAN_REGEX = re.compile(
    r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$'
)

# ===============================
# HELPERS
# ===============================
def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def normalize_plate(text):
    return ''.join(OCR_FIX.get(c, c) for c in text)

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return thresh

# ===============================
# IMAGE UPLOAD
# ===============================
uploaded = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    frame = np.array(image)

    if st.button("🔍 Run ANPR"):
        with st.spinner("Processing image..."):

            results = model(frame, conf=0.3, verbose=False)[0]

            plate_counter = Counter()
            debug_ocr = []

            if results.boxes is not None:
                h, w, _ = frame.shape

                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)

                    # Safe bounding box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    proc = preprocess_plate(crop)

                    texts = reader.readtext(
                        proc,
                        detail=0,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    )

                    debug_ocr.extend(texts)

                    for t in texts:
                        cleaned = clean_text(t)

                        if not (8 <= len(cleaned) <= 11):
                            continue

                        normalized = normalize_plate(cleaned)

                        if SOFT_INDIAN_REGEX.match(normalized):
                            plate_counter[normalized] += 1

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            final_plate = None
            confidence = 0.0

            if plate_counter:
                final_plate, score = plate_counter.most_common(1)[0]
                confidence = score / sum(plate_counter.values())

            # ===============================
            # DISPLAY
            # ===============================
            st.image(frame, caption="Detected Plates", use_column_width=True)

            st.subheader("🔎 OCR Debug Output")
            st.write(debug_ocr)

            if final_plate:
                st.success(f"🪪 Plate Number: **{final_plate}**")
                st.info(f"Confidence: {confidence:.2f}")
            else:
                st.warning("No valid Indian number plate detected")


