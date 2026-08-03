import base64
import io

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

from predict import predict_all_probabilities

st.set_page_config(page_title="Emotion Engine", page_icon="🎭", layout="wide")

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
BG = "#0F1115"
PANEL = "#171A21"
PANEL_2 = "#1D212B"
BORDER = "#2A2E38"
TEXT = "#ECEAE3"
MUTED = "#7D8290"
NEUTRAL_ACCENT = "#8A93A6"

EMOTION_COLORS = {
    "happy": "#F2B84B",
    "joy": "#F2B84B",
    "sad": "#5C8AEA",
    "sadness": "#5C8AEA",
    "angry": "#E85C4A",
    "anger": "#E85C4A",
    "fear": "#9B6BF2",
    "afraid": "#9B6BF2",
    "surprise": "#4ECDC4",
    "surprised": "#4ECDC4",
    "disgust": "#7FB069",
    "neutral": NEUTRAL_ACCENT,
}


def emotion_color(label: str) -> str:
    return EMOTION_COLORS.get(label.lower().strip(), NEUTRAL_ACCENT)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500;600&display=swap');

        html, body, .stApp {{
            background-color: {BG} !important;
            color: {TEXT};
            font-family: 'Inter', sans-serif;
        }}

        .block-container {{
            padding-top: 2.5rem;
            max-width: 1180px;
        }}

        /* ---- Header ---- */
        .eyebrow {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {MUTED};
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 0.4rem;
        }}
        .dot {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: {NEUTRAL_ACCENT};
            display: inline-block;
        }}
        .dot.live {{
            background: #E85C4A;
            box-shadow: 0 0 0 0 rgba(232,92,74,0.6);
            animation: pulse 1.8s infinite;
        }}
        @keyframes pulse {{
            0%   {{ box-shadow: 0 0 0 0 rgba(232,92,74,0.55); }}
            70%  {{ box-shadow: 0 0 0 8px rgba(232,92,74,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(232,92,74,0); }}
        }}
        h1.title {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 2.6rem;
            letter-spacing: -0.01em;
            margin: 0 0 0.3rem 0;
            color: {TEXT};
        }}
        .subtitle {{
            color: {MUTED};
            font-size: 0.98rem;
            margin-bottom: 2rem;
            max-width: 640px;
        }}

        /* ---- Tabs ---- */
        div[data-testid="stTabs"] button[role="tab"] {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            letter-spacing: 0.04em;
            color: {MUTED};
            background: transparent;
            border-bottom: 2px solid {BORDER};
        }}
        div[data-testid="stTabs"] button[aria-selected="true"] {{
            color: {TEXT};
            border-bottom: 2px solid {TEXT};
        }}

        /* ---- Viewfinder frame (bracket corners) ---- */
        .viewfinder {{
            position: relative;
            border: 1px solid {BORDER};
            background: {PANEL};
            border-radius: 4px;
            padding: 14px;
        }}
        .viewfinder::before, .viewfinder::after {{
            content: "";
            position: absolute;
            width: 18px;
            height: 18px;
        }}
        .viewfinder::before {{
            top: -1px; left: -1px;
            border-top: 2px solid var(--vf-accent, {NEUTRAL_ACCENT});
            border-left: 2px solid var(--vf-accent, {NEUTRAL_ACCENT});
        }}
        .viewfinder::after {{
            bottom: -1px; right: -1px;
            border-bottom: 2px solid var(--vf-accent, {NEUTRAL_ACCENT});
            border-right: 2px solid var(--vf-accent, {NEUTRAL_ACCENT});
        }}
        .viewfinder img {{
            width: 100%;
            border-radius: 2px;
            display: block;
        }}
        .vf-label {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            color: {MUTED};
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        /* ---- Panel / cards ---- */
        .panel {{
            border: 1px solid {BORDER};
            background: {PANEL};
            border-radius: 4px;
            padding: 18px 20px;
        }}
        .panel + .panel {{ margin-top: 12px; }}

        .face-heading {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 1.05rem;
            margin-bottom: 2px;
        }}
        .face-conf {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
            color: {MUTED};
            margin-bottom: 14px;
        }}

        /* ---- Probability bars ---- */
        .prob-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .prob-label {{
            width: 82px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            color: {MUTED};
            flex-shrink: 0;
        }}
        .prob-track {{
            flex: 1;
            height: 6px;
            background: {PANEL_2};
            border: 1px solid {BORDER};
            border-radius: 3px;
            overflow: hidden;
        }}
        .prob-fill {{
            height: 100%;
            border-radius: 3px;
        }}
        .prob-value {{
            width: 48px;
            text-align: right;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: {TEXT};
            flex-shrink: 0;
        }}

        /* ---- Empty / status states ---- */
        .status-line {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            color: {MUTED};
            border: 1px dashed {BORDER};
            border-radius: 4px;
            padding: 14px 16px;
        }}

        /* ---- File uploader ---- */
        div[data-testid="stFileUploader"] section {{
            background: {PANEL};
            border: 1px dashed {BORDER};
            border-radius: 4px;
        }}

        /* ---- webrtc container ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            border-color: {BORDER} !important;
            background: {PANEL};
            border-radius: 4px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(mode: str):
    live = mode == "webcam"
    dot_class = "dot live" if live else "dot"
    label = "LIVE FEED &middot; WEBRTC" if live else "STATIC ANALYSIS &middot; IMAGE"
    st.markdown(
        f"""
        <div class="eyebrow"><span class="{dot_class}"></span>COMPUTER VISION &nbsp;/&nbsp; {label}</div>
        <h1 class="title">Emotion Engine</h1>
        <div class="subtitle">Face detection via Haar cascade, expression classification via a fine-tuned
        ResNet50 head. Upload a still frame or run it live against your webcam.</div>
        """,
        unsafe_allow_html=True,
    )


def render_prob_bars(probs: dict) -> str:
    rows = []
    for emo, p in probs.items():
        color = emotion_color(emo)
        rows.append(
            f"""
            <div class="prob-row">
                <div class="prob-label">{emo}</div>
                <div class="prob-track"><div class="prob-fill" style="width:{p*100:.1f}%; background:{color};"></div></div>
                <div class="prob-value">{p*100:.1f}%</div>
            </div>
            """
        )
    return "".join(rows)


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------
@st.cache_resource
def get_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


face_cascade = get_face_cascade()


def detect_and_annotate(img_bgr, box_color_bgr=(74, 92, 232)):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    results = []
    for (x, y, w, h) in faces:
        pad_w = int(0.1 * w)
        pad_h = int(0.1 * h)
        y1 = max(0, y - pad_h)
        y2 = min(gray.shape[0], y + h + pad_h)
        x1 = max(0, x - pad_w)
        x2 = min(gray.shape[1], x + w + pad_w)

        face_crop = gray[y1:y2, x1:x2]
        face_crop = cv2.equalizeHist(face_crop)

        probs = predict_all_probabilities(face_crop)
        top_emotion, top_conf = next(iter(probs.items()))

        results.append({
            "box": (x, y, w, h),
            "emotion": top_emotion,
            "confidence": top_conf,
            "probs": probs,
        })

        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), box_color_bgr, 2)
        cv2.putText(
            img_bgr, f"{top_emotion} ({top_conf:.0%})", (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color_bgr, 2
        )

    return img_bgr, results


def bgr_to_data_uri(img_bgr) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
inject_css()

tab_upload, tab_webcam = st.tabs(["UPLOAD", "LIVE WEBCAM"])

# ---------------- Upload tab ----------------
with tab_upload:
    render_header("upload")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        annotated_bgr, results = detect_and_annotate(img_bgr.copy())
        dominant_color = emotion_color(results[0]["emotion"]) if results else NEUTRAL_ACCENT

        col_frame, col_panel = st.columns([1.2, 1])

        with col_frame:
            st.markdown('<div class="vf-label">Analyzed frame</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="viewfinder" style="--vf-accent:{dominant_color};">
                    <img src="{bgr_to_data_uri(annotated_bgr)}" />
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_panel:
            if not results:
                st.markdown(
                    '<div class="status-line">NO FACES DETECTED &mdash; try a closer, front-facing shot.</div>',
                    unsafe_allow_html=True,
                )
            else:
                for i, r in enumerate(results):
                    st.markdown(
                        f"""
                        <div class="panel">
                            <div class="face-heading">Face {i+1} &middot; {r['emotion'].capitalize()}</div>
                            <div class="face-conf">confidence {r['confidence']*100:.1f}%</div>
                            {render_prob_bars(r['probs'])}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------- Webcam tab ----------------
with tab_webcam:
    render_header("webcam")

    RTC_CONFIGURATION = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    class EmotionVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            annotated, _ = detect_and_annotate(img)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    with st.container(border=True):
        webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionVideoProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
        )