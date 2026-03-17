import os
# Force headless mode to prevent libGL/GUI errors
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import re
from collections import Counter
from PIL import Image
import torch

# ===============================
# CONFIG & UI
# ===============================
st.set_page_config(page_title="ANPR India", layout="wide")
st.title("📸 Indian Number Plate Recognition")

# ===============================
# MODEL LOADER (Cloud Optimized)
# ===============================
@st.cache_resource
def load_models():
    # Use CPU for Streamlit Cloud Free Tier
    use_gpu = torch.cuda.is_available()
    
    # Load YOLO (Ensure best.pt is in the same folder)
    model = YOLO("best.pt")
    
    # Load EasyOCR
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    
    return model, reader, "GPU" if use_gpu else "CPU"

model, reader, device_type = load_models()

# ===============================
# LOGIC HELPERS
# ===============================
OCR_FIX = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
INDIAN_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$')

def clean_and_normalize(text):
    text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return ''.join(OCR_FIX.get(c, c) for c in text)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Enhance contrast for better OCR
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

# ===============================
# SIDEBAR / UPLOAD
# ===============================
st.sidebar.header("Settings")
st.sidebar.write(f"Running on: **{device_type}**")
uploaded = st.file_uploader("Upload Vehicle Image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    frame = np.array(img)
    display_frame = frame.copy()

    if st.button("🚀 Start Recognition"):
        with st.spinner("Analyzing..."):
            # 1. Detection
            results = model(frame, conf=0.4, verbose=False)[0]
            
            plates_found = []
            
            if results.boxes:
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # 2. Crop & Preprocess
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0: continue
                    
                    processed_plate = preprocess(plate_crop)
                    
                    # 3. OCR (Optimized for multiple snippets)
                    ocr_res = reader.readtext(processed_plate, detail=0)
                    raw_text = "".join(ocr_res)
                    final_text = clean_and_normalize(raw_text)
                    
                    if 7 <= len(final_text) <= 11:
                        plates_found.append(final_text)
                        # Draw on UI
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(display_frame, final_text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 4. Display results
            st.image(display_frame, use_container_width=True)
            
            if plates_found:
                st.success(f"### ✅ Detected Plate: {plates_found[0]}")
                if not INDIAN_REGEX.match(plates_found[0]):
                    st.warning("Note: Format doesn't match standard Indian Plate (e.g., MH12AB1234)")
            else:
                st.error("No plate detected. Ensure the plate is clearly visible.")
