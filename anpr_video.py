import streamlit as st
import cv2
import re
from collections import defaultdict, Counter
import tempfile

# ===============================
# SAFE ML IMPORT (CLOUD GUARD)
# ===============================
ML_AVAILABLE = False

try:
    from ultralytics import YOLO
    import easyocr
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# ===============================
# STREAMLIT CONFIG
# ===============================
st.set_page_config(page_title="ANPR Video", layout="wide")
st.title("🚘 ANPR Video (Indian Plates + Self-Correcting OCR)")

# ===============================
# STOP IF ML NOT AVAILABLE
# ===============================
if not ML_AVAILABLE:
    st.error(
        "🚫 YOLO + OCR cannot run on Streamlit Cloud.\n\n"
        "Reason: Torch is incompatible with Python 3.13.\n\n"
        "✅ This app runs correctly on LOCAL / DOCKER / HUGGING FACE.\n"
        "📌 Deploy this UI here, run inference elsewhere."
    )
    st.stop()

# ===============================
# LOAD MODELS
# ===============================
@st.cache_resource
def load_models():
    model = YOLO("best.pt")
    reader = easyocr.Reader(['en'], gpu=False)
    return model, reader

model, reader = load_models()

# ===============================
# INDIAN STATE CODES
# ===============================
INDIAN_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS",
    "TR","UK","UP","WB",
    "AN","CH","DD","DL","JK","LA","LD","PY"
}

# ===============================
# PLATE CLEANING (INDIAN LOGIC)
# ===============================
def clean_plate(text):
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)

    if len(text) < 8:
        return ""

    state = text[:2].replace('0', 'O')
    if state not in INDIAN_STATE_CODES:
        return ""

    mid = text[2:-4].replace('O', '0').replace('I', '1')
    last = text[-4:].replace('O', '0')

    return state + mid + last

# ===============================
# VIDEO UPLOAD
# ===============================
uploaded = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())
    video_path = tfile.name

    if st.button("▶️ Run ANPR"):
        cap = cv2.VideoCapture(video_path)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out_path = "anpr_output.mp4"
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        plate_votes = defaultdict(list)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1
            progress.progress(min(frame_id / total_frames, 1.0))

            results = model.track(
                frame,
                conf=0.3,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]

            if results.boxes is None or results.boxes.id is None:
                out.write(frame)
                continue

            for box, tid in zip(results.boxes.xyxy, results.boxes.id):
                track_id = int(tid)
                x1, y1, x2, y2 = map(int, box)

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

                texts = reader.readtext(
                    gray,
                    detail=0,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )

                if texts:
                    candidate = max(texts, key=len)
                    cleaned = clean_plate(candidate)

                    if cleaned:
                        plate_votes[track_id].append(cleaned)
                        plate_votes[track_id] = plate_votes[track_id][-20:]

                final_text = ""
                if plate_votes[track_id]:
                    final_text = Counter(
                        plate_votes[track_id]
                    ).most_common(1)[0][0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if final_text:
                    cv2.putText(
                        frame,
                        final_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )

            out.write(frame)

        cap.release()
        out.release()
        progress.empty()

        st.success("✅ Processing complete")
        st.video(out_path)

        with open(out_path, "rb") as f:
            st.download_button(
                "⬇️ Download Output Video",
                f,
                file_name="anpr_output.mp4"
            )
