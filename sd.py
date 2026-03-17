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
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="ANPR Indian Plates", layout="wide")
st.title("📸 Automatic Number Plate Recognition")
st.markdown("---")

# ===============================
# LOAD MODELS (Optimized for Cloud)
# ===============================
@st.cache_resource
def load_models():
    # Use CPU if GPU is not available (Streamlit Cloud Free Tier)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load YOLO Model
    model = YOLO("best.pt")
    
    # Load EasyOCR Reader
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    return model, reader, device

model, reader, device = load_models()

# ===============================
# OCR UTILITIES
# ===============================
OCR_FIX = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'
}

SOFT_INDIAN_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$')

def clean_text(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

def normalize_plate(text):
    return ''.join(OCR_FIX.get(c, c) for c in text)

def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bilateral filter removes noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    # Use Adaptive Threshold for varying lighting conditions
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

# ===============================
# MAIN UI
# ===============================
col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader("Upload a vehicle image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    frame = np.array(image)
    display_frame = frame.copy()

    with col2:
        st.info(f"Running on: **{device.upper()}**")
        run_btn = st.button("🔍 Run ANPR Scan", use_container_width=True)

    if run_btn:
        with st.spinner("Processing image..."):
            results = model(frame, conf=0.3, verbose=False)[0]
            
            plate_counter = Counter()
            debug_ocr = []

            if results.boxes is not None:
                for box in results.boxes.xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Crop the plate
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue

                    # Process and OCR
                    proc = preprocess_plate(crop)
                    # Use paragraph=True to join multi-line Indian plates
                    ocr_results = reader.readtext(proc, detail=0, paragraph=False)
                    
                    # Combine found snippets into one string
                    raw_text = "".join(ocr_results)
                    cleaned = clean_text(raw_text)
                    normalized = normalize_plate(cleaned)
                    
                    debug_ocr.append(f"Raw: {raw_text} -> Cleaned: {normalized}")

                    # Basic Indian Plate length validation
                    if 7 <= len(normalized) <= 11:
                        plate_counter[normalized] += 1

                    # Draw visual box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(display_frame, normalized, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display Results
            st.image(display_frame, caption="Detection Result", use_column_width=True)

            if plate_counter:
                final_plate, _ = plate_counter.most_common(1)[0]
                if SOFT_INDIAN_REGEX.match(final_plate):
                    st.success(f"### 🪪 Detected Plate: **{final_plate}**")
                else:
                    st.warning(f"Detected: {final_plate} (Pattern doesn't perfectly match Indian Format)")
            else:
                st.error("No valid number plate detected. Try a clearer image.")

            with st.expander("See OCR Debug Logs"):
                st.write(debug_ocr)
