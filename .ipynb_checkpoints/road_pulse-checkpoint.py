import streamlit as st
import pandas as pd
import cv2
import tempfile
from pathlib import Path

from ultralytics import YOLO
from streamlit_geolocation import streamlit_geolocation


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="RoadPulse",
    page_icon="🛣️",
    layout="wide"
)

st.title("🛣️ RoadPulse")
st.caption("AI-Powered Road Condition Intelligence")


# =========================================================
# MODEL SETTINGS
# =========================================================

MODEL_PATH = "road_pulse_best.pt"

CLASS_NAMES = {
    0: "Longitudinal Crack",
    1: "Transverse Crack",
    2: "Fatigue Crack",
    3: "Pothole"
}

# Prototype RoadPulse penalties
DEFECT_PENALTIES = {
    "Longitudinal Crack": 2,
    "Transverse Crack": 3,
    "Fatigue Crack": 8,
    "Pothole": 12
}

ROUGHNESS_PENALTIES = {
    "Smooth": 0,
    "Rough": 15,
    "Severely Rough": 30
}


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not Path(MODEL_PATH).exists():
        return None

    return YOLO(MODEL_PATH)


model = load_model()


# =========================================================
# HEALTH SCORE
# =========================================================

def calculate_health_score(counts, roughness):

    score = 100

    for defect, count in counts.items():
        score -= count * DEFECT_PENALTIES[defect]

    score -= ROUGHNESS_PENALTIES[roughness]

    return max(0, min(100, score))


def get_status(score):

    if score >= 80:
        return "Good"

    elif score >= 60:
        return "Fair"

    elif score >= 40:
        return "Poor"

    else:
        return "Critical"


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Survey Information")

road_name = st.sidebar.text_input(
    "Road / Street Name",
    placeholder="Example: Road 101"
)

roughness = st.sidebar.selectbox(
    "Road Roughness",
    [
        "Smooth",
        "Rough",
        "Severely Rough"
    ]
)

confidence = st.sidebar.slider(
    "Detection Confidence",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

st.sidebar.caption(
    "Roughness is manual in this MVP. "
    "Later it will come from phone accelerometer/gyroscope data."
)


# =========================================================
# GPS
# =========================================================

st.header("📍 Survey Location")

location = streamlit_geolocation()

latitude = None
longitude = None

if location and location.get("latitude") is not None:

    latitude = location["latitude"]
    longitude = location["longitude"]

    st.success("GPS location received")

    gps1, gps2 = st.columns(2)

    gps1.metric(
        "Latitude",
        f"{latitude:.6f}"
    )

    gps2.metric(
        "Longitude",
        f"{longitude:.6f}"
    )

    gps_df = pd.DataFrame({
        "lat": [latitude],
        "lon": [longitude]
    })

    st.map(gps_df)

else:

    st.info(
        "Press the location button and allow location access."
    )


st.caption(
    "GPS currently represents the device running RoadPulse. "
    "It is not extracted from the uploaded video."
)


# =========================================================
# VIDEO UPLOAD
# =========================================================

st.divider()

st.header("🎥 Road Survey Analysis")

uploaded_video = st.file_uploader(
    "Upload road survey video",
    type=["mp4", "mov", "avi", "mkv"]
)


if uploaded_video is not None:

    video_bytes = uploaded_video.getvalue()

    st.video(video_bytes)

    st.write(
        "**Video:**",
        uploaded_video.name
    )


# =========================================================
# ANALYZE VIDEO
# =========================================================

    if st.button(
        "🔍 Analyze Road",
        type="primary",
        use_container_width=True
    ):

        if model is None:

            st.error(
                "best.pt was not found. "
                "Put your trained model beside app.py."
            )

            st.stop()


        # -------------------------------------------------
        # Save uploaded video temporarily
        # -------------------------------------------------

        suffix = Path(uploaded_video.name).suffix

        if not suffix:
            suffix = ".mp4"

        temp_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        )

        temp_file.write(video_bytes)
        temp_file.close()


        # -------------------------------------------------
        # Open video
        # -------------------------------------------------

        cap = cv2.VideoCapture(temp_file.name)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )


        # -------------------------------------------------
        # Defect counters
        # -------------------------------------------------

        counts = {
            "Longitudinal Crack": 0,
            "Transverse Crack": 0,
            "Fatigue Crack": 0,
            "Pothole": 0
        }

        # Used to avoid counting same tracked defect repeatedly
        seen_tracks = set()

        evidence_frames = []

        frame_number = 0
        analyzed_frames = 0

        FRAME_STRIDE = 5

        progress = st.progress(0)

        message = st.empty()

        message.info(
            "RoadPulse is analyzing the video..."
        )


        # -------------------------------------------------
        # VIDEO LOOP
        # -------------------------------------------------

        while True:

            success, frame = cap.read()

            if not success:
                break


            # Only analyze every 5th frame for speed
            if frame_number % FRAME_STRIDE != 0:

                frame_number += 1
                continue


            # -------------------------------------------------
            # YOLO + TRACKING
            # -------------------------------------------------

            result = model.track(
                frame,
                persist=True,
                conf=confidence,
                imgsz=512,
                tracker="bytetrack.yaml",
                verbose=False
            )[0]


            boxes = result.boxes


            if boxes is not None and len(boxes) > 0:

                class_ids = (
                    boxes.cls
                    .int()
                    .cpu()
                    .tolist()
                )


                # Tracking IDs
                if boxes.id is not None:

                    track_ids = (
                        boxes.id
                        .int()
                        .cpu()
                        .tolist()
                    )

                else:

                    track_ids = [None] * len(class_ids)


                # -------------------------------------------------
                # COUNT UNIQUE TRACKED DEFECTS
                # -------------------------------------------------

                for i, class_id in enumerate(class_ids):

                    if class_id not in CLASS_NAMES:
                        continue

                    defect_name = CLASS_NAMES[class_id]

                    track_id = track_ids[i]


                    if track_id is not None:

                        unique_key = (
                            class_id,
                            track_id
                        )

                        if unique_key not in seen_tracks:

                            seen_tracks.add(unique_key)

                            counts[defect_name] += 1

                    else:

                        # fallback if tracking ID isn't available
                        counts[defect_name] += 1


                # -------------------------------------------------
                # SAVE A FEW EVIDENCE FRAMES
                # -------------------------------------------------

                if len(evidence_frames) < 6:

                    annotated = result.plot()

                    annotated = cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    )

                    evidence_frames.append(
                        annotated
                    )


            analyzed_frames += 1
            frame_number += 1


            # -------------------------------------------------
            # UPDATE PROGRESS BAR
            # -------------------------------------------------

            if total_frames > 0:

                percent = min(
                    frame_number / total_frames,
                    1.0
                )

                progress.progress(percent)


        cap.release()

        progress.progress(1.0)

        message.success(
            "✅ Road analysis completed"
        )


        # =====================================================
        # HEALTH SCORE
        # =====================================================

        health_score = calculate_health_score(
            counts,
            roughness
        )

        status = get_status(
            health_score
        )

        total_defects = sum(
            counts.values()
        )


        # =====================================================
        # RESULTS
        # =====================================================

        st.divider()

        st.header("📊 Road Condition Results")


        metric1, metric2, metric3, metric4 = st.columns(4)


        metric1.metric(
            "Road Health Score",
            f"{health_score}/100"
        )


        metric2.metric(
            "Road Status",
            status
        )


        metric3.metric(
            "Detected Defects",
            total_defects
        )


        metric4.metric(
            "Frames Analyzed",
            analyzed_frames
        )


        # =====================================================
        # DEFECT BREAKDOWN
        # =====================================================

        st.subheader("Detected Damage")


        defect_df = pd.DataFrame({

            "Damage Type": [
                "Longitudinal Crack",
                "Transverse Crack",
                "Fatigue Crack",
                "Pothole"
            ],

            "Detected": [
                counts["Longitudinal Crack"],
                counts["Transverse Crack"],
                counts["Fatigue Crack"],
                counts["Pothole"]
            ],

            "Health Penalty Each": [
                2,
                3,
                8,
                12
            ]
        })


        st.dataframe(
            defect_df,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================
        # ROAD SUMMARY
        # =====================================================

        st.subheader("RoadPulse Assessment")


        summary = pd.DataFrame({

            "Road": [
                road_name if road_name else "Unnamed Road"
            ],

            "Health Score": [
                health_score
            ],

            "Status": [
                status
            ],

            "Potholes": [
                counts["Pothole"]
            ],

            "Longitudinal Cracks": [
                counts["Longitudinal Crack"]
            ],

            "Transverse Cracks": [
                counts["Transverse Crack"]
            ],

            "Fatigue Cracks": [
                counts["Fatigue Crack"]
            ],

            "Roughness": [
                roughness
            ],

            "Latitude": [
                latitude
            ],

            "Longitude": [
                longitude
            ]
        })


        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )


        # =====================================================
        # EVIDENCE IMAGES
        # =====================================================

        st.subheader("📸 Detection Evidence")


        if evidence_frames:

            columns = st.columns(2)

            for i, image in enumerate(evidence_frames):

                columns[i % 2].image(
                    image,
                    caption=f"Detection Evidence {i + 1}",
                    use_container_width=True
                )

        else:

            st.info(
                "No road damage was detected in the sampled frames."
            )


        # =====================================================
        # DOWNLOAD REPORT
        # =====================================================

        csv = summary.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "⬇️ Download RoadPulse Report",
            data=csv,
            file_name="roadpulse_report.csv",
            mime="text/csv"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "RoadPulse prototype — AI road damage detection, "
    "GPS mapping and road health assessment."
)