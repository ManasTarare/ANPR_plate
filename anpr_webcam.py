import cv2
import re
from ultralytics import YOLO
import easyocr
from collections import defaultdict, Counter

# ===============================
# LOAD MODELS
# ===============================
model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# INDIAN STATE CODES
# ===============================
INDIAN_STATE_CODES = {
    "AP","AR","AS","BR","CG","GA","GJ","HR","HP","JH","KA","KL",
    "MP","MH","MN","ML","MZ","NL","OD","PB","RJ","SK","TN","TS",
    "TR","UK","UP","WB","AN","CH","DD","DL","JK","LA","LD","PY"
}

# ===============================
# STRICT PLATE REGEX
# ===============================
STRICT_REGEX = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')

# ===============================
# LOOSE CLEAN (STAGE-1)
# ===============================
def loose_clean(text):
    text = text.upper()
    return re.sub(r'[^A-Z0-9]', '', text)

# ===============================
# STRICT CLEAN (STAGE-2)
# ===============================
def strict_clean(text):
    if len(text) != 10:
        return ""

    t = list(text)

    # State
    t[0] = t[0].replace('0', 'O')
    t[1] = t[1].replace('0', 'O')
    if t[0] + t[1] not in INDIAN_STATE_CODES:
        return ""

    # RTO
    t[2] = t[2].replace('O', '0')
    t[3] = t[3].replace('O', '0')

    # Series
    t[4] = t[4].replace('0', 'O').replace('1', 'I')
    t[5] = t[5].replace('0', 'O').replace('1', 'I')

    # Number
    for i in range(6, 10):
        t[i] = t[i].replace('O', '0')

    plate = ''.join(t)

    if not STRICT_REGEX.match(plate):
        return ""

    return plate

# ===============================
# VIDEO SOURCE
# ===============================
cap = cv2.VideoCapture(0)

# ===============================
# PARAMETERS
# ===============================
FRAME_SKIP = 2
MAX_VOTES = 25
LOCK_THRESHOLD = 0.6   # confidence
FONT = cv2.FONT_HERSHEY_SIMPLEX

# ===============================
# MEMORY
# ===============================
votes = defaultdict(list)
locked_plate = {}

frame_id = 0

print("[INFO] Real-time ANPR started. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    results = model.track(
        frame,
        conf=0.3,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )[0]

    if results.boxes is not None and results.boxes.id is not None:
        for box, tid in zip(results.boxes.xyxy, results.boxes.id):
            track_id = int(tid)
            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ---------------- OCR ----------------
            if frame_id % FRAME_SKIP == 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                gray = cv2.GaussianBlur(gray, (5, 5), 0)
                _, gray = cv2.threshold(gray, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                texts = reader.readtext(
                    gray,
                    detail=0,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )

                if texts:
                    candidate = max(texts, key=len)
                    loose = loose_clean(candidate)

                    if 6 <= len(loose) <= 10:
                        votes[track_id].append(loose)
                        votes[track_id] = votes[track_id][-MAX_VOTES:]

            # ---------------- VOTING ----------------
            final_text = ""
            confidence = 0.0

            if votes[track_id]:
                counter = Counter(votes[track_id])
                best, count = counter.most_common(1)[0]
                confidence = count / sum(counter.values())

                # Try strict validation only after voting
                strict = strict_clean(best)
                if strict:
                    final_text = strict
                    if confidence >= LOCK_THRESHOLD:
                        locked_plate[track_id] = strict
                else:
                    final_text = best  # show loose guess

            if track_id in locked_plate:
                final_text = locked_plate[track_id]

            # ---------------- DRAW ----------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            if final_text:
                label = f"{final_text}  ({confidence:.2f})"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    FONT,
                    0.9,
                    (0,255,0),
                    2
                )

    cv2.imshow("Real-Time ANPR", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] ANPR stopped.")
