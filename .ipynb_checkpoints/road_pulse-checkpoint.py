
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import subprocess
import json
import re
from pathlib import Path
from ultralytics import YOLO

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="RoadPulse", page_icon="🛣️", layout="wide")
st.title("🛣️ RoadPulse")
st.caption("AI Road Condition Intelligence")

MODEL_PATH = "road_pulse_best.pt"

EXPECTED_CLASSES = {
    0: "longitudinal_crack",
    1: "transverse_crack",
    2: "fatigue_crack",
    3: "pothole",
}

DISPLAY_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Fatigue Crack",
    3: "Pothole",
}

DEFECT_PENALTIES = {
    "Longitudinal Crack": 2,
    "Transverse Crack": 3,
    "Fatigue Crack": 8,
    "Pothole": 12,
}

ROUGHNESS_PENALTIES = {
    "Smooth": 0,
    "Moderate": 8,
    "Rough": 15,
    "Severe": 30,
}

# =========================================================
# MODEL
# =========================================================
@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        return None
    return YOLO(MODEL_PATH)

model = load_model()

if model is None:
    st.error("best.pt was not found. Put best.pt in the same repository/folder as this app.")
    st.stop()

st.sidebar.success("Model loaded")
st.sidebar.write("Classes:", model.names)

# Check that the uploaded weights are actually the 4-class RoadPulse model.
if len(model.names) != 4:
    st.error(
        "The loaded best.pt does not appear to be the 4-class RoadPulse model. "
        "Expected 4 road-damage classes."
    )
    st.stop()

# =========================================================
# GPS FROM VIDEO METADATA
# =========================================================
def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None

def extract_video_gps(video_path):
    """
    Attempts to read GPS stored INSIDE the video container using ExifTool.
    Returns (lat, lon, metadata_message).
    This does NOT use the viewer/device location.
    """
    try:
        proc = subprocess.run(
            ["exiftool", "-j", "-n", str(video_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return None, None, "ExifTool could not read the video metadata."

        items = json.loads(proc.stdout)
        if not items:
            return None, None, "No readable metadata found."

        meta = items[0]

        # Most common explicit numeric GPS fields.
        lat = _to_float(meta.get("GPSLatitude"))
        lon = _to_float(meta.get("GPSLongitude"))
        if lat is not None and lon is not None:
            return lat, lon, "Embedded GPS coordinates found in the video."

        # QuickTime/MP4 often stores ISO6709-like strings.
        possible_fields = [
            "GPSCoordinates",
            "GPSPosition",
            "Location",
            "LocationInformation",
            "LocationCreatedGPSCoordinates",
        ]

        for key in possible_fields:
            value = meta.get(key)
            if value is None:
                continue

            text = str(value)

            # ISO6709 example: +26.223500+050.587600/
            m = re.search(
                r"([+-]\d{1,3}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)",
                text,
            )
            if m:
                return float(m.group(1)), float(m.group(2)), f"Embedded GPS found in {key}."

            # Plain "lat lon" or "lat, lon"
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
            if len(nums) >= 2:
                a, b = float(nums[0]), float(nums[1])
                if -90 <= a <= 90 and -180 <= b <= 180:
                    return a, b, f"Embedded GPS found in {key}."

        return None, None, "No embedded GPS coordinates were found in this video."

    except FileNotFoundError:
        return None, None, "ExifTool is not installed on the server."
    except Exception as e:
        return None, None, f"Could not read GPS metadata: {e}"

# =========================================================
# VIDEO ROUGHNESS PROXY
# =========================================================
def camera_motion_value(prev_gray, gray):
    """
    Estimates frame-to-frame global camera motion.
    Used only as a VIDEO-BASED roughness proxy.
    """
    pts0 = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7,
    )

    if pts0 is None or len(pts0) < 10:
        return None

    pts1, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, gray, pts0, None
    )

    if pts1 is None or status is None:
        return None

    good0 = pts0[status.flatten() == 1]
    good1 = pts1[status.flatten() == 1]

    if len(good0) < 10:
        return None

    M, _ = cv2.estimateAffinePartial2D(
        good0,
        good1,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if M is None:
        return None

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    angle = float(np.arctan2(M[1, 0], M[0, 0]))

    h, w = gray.shape
    diagonal = max(np.hypot(w, h), 1.0)

    # Normalize translation by image size and add rotation magnitude.
    return (np.hypot(dx, dy) / diagonal) + abs(angle)

def classify_video_roughness(motion_values):
    """
    Produces a prototype Video Roughness Index (0-100).
    It is NOT IRI and is not an engineering-certified road roughness measure.
    """
    arr = np.asarray([x for x in motion_values if x is not None], dtype=float)

    if len(arr) < 8:
        return None, "Unavailable"

    # Remove slower camera motion (turning/panning) using a moving average.
    window = min(9, len(arr))
    kernel = np.ones(window) / window
    trend = np.convolve(arr, kernel, mode="same")
    residual = np.abs(arr - trend)

    # Robust high-frequency shake statistic.
    raw = float(np.median(residual))

    # Prototype scaling. Must later be calibrated with Bahrain sensor/road labels.
    index = int(np.clip(raw * 5000, 0, 100))

    if index < 20:
        label = "Smooth"
    elif index < 40:
        label = "Moderate"
    elif index < 65:
        label = "Rough"
    else:
        label = "Severe"

    return index, label

# =========================================================
# SIMPLE TRACK-LESS DEDUPLICATION
# =========================================================
def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def match_or_create_tracks(detections, tracks, next_track_id, analyzed_index, frame_diag):
    """
    Very lightweight same-class box matching across nearby frames.
    Avoids the extra ByteTrack/lap dependency and reduces repeated counts.
    """
    assigned = set()
    new_track_ids = []

    for det in detections:
        cls_id = det["cls"]
        box = det["box"]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        best_id = None
        best_score = -1e9

        for tid, tr in tracks.items():
            if tid in assigned:
                continue
            if tr["cls"] != cls_id:
                continue
            if analyzed_index - tr["last_seen"] > 6:
                continue

            old = tr["box"]
            ocx = (old[0] + old[2]) / 2
            ocy = (old[1] + old[3]) / 2

            center_dist = np.hypot(cx - ocx, cy - ocy) / max(frame_diag, 1.0)
            overlap = iou(box, old)

            # Allow either spatial overlap or a reasonably close center.
            if overlap >= 0.10 or center_dist <= 0.12:
                score = overlap - center_dist
                if score > best_score:
                    best_score = score
                    best_id = tid

        if best_id is None:
            best_id = next_track_id
            next_track_id += 1
            tracks[best_id] = {
                "cls": cls_id,
                "box": box,
                "last_seen": analyzed_index,
            }
            new_track_ids.append(best_id)
        else:
            tracks[best_id]["box"] = box
            tracks[best_id]["last_seen"] = analyzed_index

        assigned.add(best_id)

    return next_track_id, new_track_ids

# =========================================================
# HEALTH SCORE
# =========================================================
def road_status(score):
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Fair"
    if score >= 40:
        return "Poor"
    return "Critical"

def health_score(counts, roughness_label):
    score = 100
    for name, count in counts.items():
        score -= DEFECT_PENALTIES[name] * count
    score -= ROUGHNESS_PENALTIES.get(roughness_label, 0)
    return max(0, min(100, int(score)))

# =========================================================
# CONTROLS
# =========================================================
st.sidebar.header("Analysis Settings")

confidence = st.sidebar.slider(
    "Detection confidence",
    min_value=0.01,
    max_value=0.90,
    value=0.05,
    step=0.01,
)

frame_stride = st.sidebar.slider(
    "Analyze every Nth frame",
    min_value=1,
    max_value=10,
    value=2,
    step=1,
)

st.sidebar.caption(
    "Use N=1 for maximum detection sensitivity. "
    "Use N=2 or 3 for a faster cloud demo."
)

road_name = st.sidebar.text_input("Road / Survey Name", "Road Survey")

# =========================================================
# UPLOAD
# =========================================================
uploaded_video = st.file_uploader(
    "Upload a road survey video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is None:
    st.info("Upload a road video to begin.")
    st.stop()

video_bytes = uploaded_video.getvalue()
st.video(video_bytes)

if st.button("🔍 Analyze Road", type="primary", use_container_width=True):

    suffix = Path(uploaded_video.name).suffix or ".mp4"
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_video.write(video_bytes)
    temp_video.close()
    video_path = Path(temp_video.name)

    # -----------------------------------------------------
    # GPS FROM THE VIDEO FILE
    # -----------------------------------------------------
    st.subheader("📍 Video Location")
    latitude, longitude, gps_message = extract_video_gps(video_path)

    if latitude is not None and longitude is not None:
        st.success(gps_message)
        c1, c2 = st.columns(2)
        c1.metric("Latitude", f"{latitude:.6f}")
        c2.metric("Longitude", f"{longitude:.6f}")

        gps_df = pd.DataFrame({"lat": [latitude], "lon": [longitude]})
        st.map(gps_df)
    else:
        st.warning(gps_message)
        st.caption(
            "RoadPulse did not use your current device location. "
            "If the recording does not contain GPS metadata, an exact location "
            "cannot be recovered reliably from the video file alone."
        )

    # -----------------------------------------------------
    # OPEN VIDEO
    # -----------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        st.error("The uploaded video could not be opened.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    counts = {
        "Longitudinal Crack": 0,
        "Transverse Crack": 0,
        "Fatigue Crack": 0,
        "Pothole": 0,
    }

    evidence_frames = []
    motion_values = []

    tracks = {}
    next_track_id = 1

    prev_gray = None
    frame_number = 0
    analyzed_index = 0

    progress = st.progress(0)
    status_box = st.empty()
    status_box.info("Analyzing road damage and video motion...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_number % frame_stride != 0:
            frame_number += 1
            continue

        h, w = frame.shape[:2]
        frame_diag = float(np.hypot(w, h))

        # -------- Roughness proxy from frame-to-frame camera vibration
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mv = camera_motion_value(prev_gray, gray)
            if mv is not None:
                motion_values.append(mv)
        prev_gray = gray

        # -------- Road damage detection
        result = model.predict(
            frame,
            conf=confidence,
            imgsz=640,
            verbose=False,
        )[0]

        detections = []

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.int().cpu().tolist()
            boxes = result.boxes.xyxy.cpu().tolist()

            for cls_id, box in zip(classes, boxes):
                if cls_id not in DISPLAY_NAMES:
                    continue
                detections.append({"cls": cls_id, "box": box})

            next_track_id, new_ids = match_or_create_tracks(
                detections,
                tracks,
                next_track_id,
                analyzed_index,
                frame_diag,
            )

            # New track = a newly observed defect event.
            for tid in new_ids:
                cls_id = tracks[tid]["cls"]
                counts[DISPLAY_NAMES[cls_id]] += 1

            if len(evidence_frames) < 8:
                annotated = result.plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                evidence_frames.append(annotated)

        analyzed_index += 1
        frame_number += 1

        if total_frames > 0:
            progress.progress(min(frame_number / total_frames, 1.0))

    cap.release()
    progress.progress(1.0)
    status_box.success("✅ Analysis complete")

    # -----------------------------------------------------
    # ROUGHNESS
    # -----------------------------------------------------
    roughness_index, roughness_label = classify_video_roughness(motion_values)

    st.subheader("〰️ Video Roughness Estimate")

    if roughness_index is None:
        st.warning("Not enough usable camera-motion data to estimate roughness.")
        roughness_label_for_score = "Smooth"
    else:
        rc1, rc2 = st.columns(2)
        rc1.metric("Video Roughness Index", f"{roughness_index}/100")
        rc2.metric("Estimated Roughness", roughness_label)
        roughness_label_for_score = roughness_label

    st.caption(
        "This is a prototype VIDEO-BASED roughness proxy calculated from high-frequency "
        "camera motion. It is not IRI and is not a certified engineering roughness measurement. "
        "For the final RoadPulse system, phone/vehicle accelerometer and gyroscope data should "
        "be used and calibrated against labeled Bahrain road sections."
    )

    # -----------------------------------------------------
    # ROAD HEALTH
    # -----------------------------------------------------
    score = health_score(counts, roughness_label_for_score)
    condition = road_status(score)
    total_defects = sum(counts.values())

    st.subheader("📊 Road Condition")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Road Health Score", f"{score}/100")
    m2.metric("Condition", condition)
    m3.metric("Unique Defect Events", total_defects)
    m4.metric("Frames Analyzed", analyzed_index)

    defect_df = pd.DataFrame(
        {
            "Damage Type": list(counts.keys()),
            "Detected Events": list(counts.values()),
            "Penalty Each": [DEFECT_PENALTIES[k] for k in counts.keys()],
        }
    )

    st.dataframe(defect_df, use_container_width=True, hide_index=True)

    # -----------------------------------------------------
    # EVIDENCE
    # -----------------------------------------------------
    st.subheader("📸 Detection Evidence")

    if evidence_frames:
        cols = st.columns(2)
        for i, img in enumerate(evidence_frames):
            cols[i % 2].image(
                img,
                caption=f"Detection Evidence {i + 1}",
                use_container_width=True,
            )
    else:
        st.warning(
            "No road damage was detected. Try lowering the confidence threshold "
            "or verify that best.pt is the trained 4-class RoadPulse model."
        )

    # -----------------------------------------------------
    # REPORT
    # -----------------------------------------------------
    report = pd.DataFrame(
        [
            {
                "Road": road_name,
                "Health Score": score,
                "Condition": condition,
                "Video Roughness Index": roughness_index,
                "Roughness Label": roughness_label,
                "Potholes": counts["Pothole"],
                "Longitudinal Cracks": counts["Longitudinal Crack"],
                "Transverse Cracks": counts["Transverse Crack"],
                "Fatigue Cracks": counts["Fatigue Crack"],
                "Latitude": latitude,
                "Longitude": longitude,
                "Video FPS": fps,
                "Frames Analyzed": analyzed_index,
            }
        ]
    )

    st.subheader("🧾 Survey Summary")
    st.dataframe(report, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download RoadPulse Report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="roadpulse_report.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "RoadPulse prototype: trained road-damage detection + embedded video GPS extraction "
    "+ video-motion roughness proxy + Road Health Score.")