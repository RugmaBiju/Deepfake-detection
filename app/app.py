
import os
import sys

import streamlit as st
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, APP_DIR)

from detector import predict_image
from video_detector import predict_video


st.set_page_config(
    page_title="TrueLens | Media Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #111a2e;
        --panel-light: #17233b;
        --border: #263550;
        --text: #f3f6fc;
        --muted: #9aabc4;
        --accent: #8b9cff;
        --accent-2: #53d6bd;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 50% -15%, #1c2c50 0%, transparent 48%),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, .brand {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
    }

    .hero {
        text-align: center;
        padding: 1rem 0 2rem;
    }

    .eyebrow {
        display: inline-block;
        color: #b8c3ff;
        background: rgba(139, 156, 255, 0.10);
        border: 1px solid rgba(139, 156, 255, 0.25);
        border-radius: 999px;
        padding: 6px 12px;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .brand {
        font-size: clamp(2.8rem, 6vw, 4.4rem);
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.06em;
        margin: 0;
    }

    .hero-copy {
        color: var(--muted);
        font-size: 1.08rem;
        margin: 0.8rem auto 0;
        max-width: 580px;
    }

    .section-kicker {
        color: #b8c3ff;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .upload-panel, .result-panel {
        background: linear-gradient(145deg, rgba(23, 35, 59, 0.94), rgba(15, 24, 42, 0.96));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.5rem;
    }

    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.25rem;
    }

    .panel-copy {
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 1rem;
    }

    [data-testid="stFileUploader"] {
        background: rgba(7, 13, 27, 0.55);
        border: 1px dashed #425477;
        border-radius: 14px;
        padding: 0.65rem;
    }

    [data-testid="stFileUploader"] section {
        border: 0;
    }

    .result-card {
        background: rgba(7, 13, 27, 0.42);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.15rem;
        height: 100%;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.85rem;
    }

    .result-value {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.25rem 0;
    }

    .result-note {
        color: #8193b0;
        font-size: 0.75rem;
    }

    .stButton > button[kind="primary"] {
        width: 100%;
        min-height: 3rem;
        border-radius: 12px;
        border: 0;
        color: #09111f;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.07);
        color: #09111f;
    }

    [data-testid="stMetric"] {
        background: rgba(7, 13, 27, 0.42);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem 1.2rem;
    }

    [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--text);
    }

    hr {
        border-color: var(--border);
        margin: 2rem 0;
    }

    .footer {
        text-align: center;
        color: #7485a1;
        font-size: 0.8rem;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Media authenticity analysis</div>
        <h1 class="brand">TrueLens</h1>
        <p class="hero-copy">
            Analyze images and videos using AI-powered authenticity detection,
            with detailed model probability estimates.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


left, right = st.columns([0.92, 1.08], gap="large")


with left:

    st.markdown('<div class="upload-panel">', unsafe_allow_html=True)

    st.markdown(
        '<div class="section-kicker">01 · Select media</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-title">Upload your media</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel-copy">Choose an image or video to analyze.</div>',
        unsafe_allow_html=True,
    )

    media_type = st.radio(
        "Media type",
        ["Image", "Video"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if media_type == "Image":

        uploaded_file = st.file_uploader(
            "Drop an image here or browse files",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed",
            key="image_upload",
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            if st.button("Analyze image", type="primary"):

                with st.spinner("Analyzing image with TrueLens models..."):

                    try:
                        results = predict_image(image)
                        st.session_state["image_results"] = results

                    except Exception as e:
                        st.error("Image analysis failed.")
                        st.exception(e)

    else:

        uploaded_file = st.file_uploader(
            "Drop a video here or browse files",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            key="video_upload",
        )

        if uploaded_file is not None:

            st.video(uploaded_file)

            if st.button("Analyze video", type="primary"):

                with st.spinner("Extracting frames and analyzing video..."):

                    try:
                        results = predict_video(uploaded_file)
                        st.session_state["video_results"] = results

                    except Exception as e:
                        st.error("Video analysis failed.")
                        st.exception(e)

    st.markdown("</div>", unsafe_allow_html=True)


with right:

    st.markdown(
        '<div class="section-kicker">02 · Analysis</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Detection results")

    if media_type == "Image":
        results = st.session_state.get("image_results")
    else:
        results = st.session_state.get("video_results")

    if results:

        prediction = results["prediction"]

        if media_type == "Image":
            confidence = results["fusion_probability"]
            probability_label = "Fusion probability"
        else:
            confidence = results["fusion_probability"]
            probability_label = "B0 + DINOv2 Fusion probability"

        if prediction in ["AI-Generated", "Manipulated"]:
            st.error(f"{prediction} media detected")
        else:
            st.success(f"Media classified as {prediction}")

        st.metric(
            probability_label,
            f"{confidence * 100:.2f}%",
        )

        st.markdown("#### Model breakdown")

        if media_type == "Image":

            col1, col2, col3 = st.columns(3)

            model_results = [
                ("EfficientNet-B0", results["b0_probability"]),
                ("DINOv2", results["dino_probability"]),
                ("Weighted fusion", confidence),
            ]

            for column, (name, probability) in zip(
                (col1, col2, col3),
                model_results,
            ):

                with column:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">{name}</div>
                            <div class="result-value">
                                {probability * 100:.1f}%
                            </div>
                            <div class="result-note">
                                Positive-class probability
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with st.expander("How the estimate is calculated"):

                st.write(
                    f"EfficientNet-B0 probability: "
                    f"{results['b0_probability'] * 100:.2f}%"
                )

                st.write(
                    f"DINOv2 probability: "
                    f"{results['dino_probability'] * 100:.2f}%"
                )

                st.write(
                    f"Combined probability: "
                    f"{confidence * 100:.2f}%"
                )

                st.write(
                    "Fusion weights: 20% EfficientNet-B0 "
                    "and 80% DINOv2."
                )

                st.caption(
                    f"Inference device: {results['device']}"
                )

        
        else:

            col1, col2, col3 = st.columns(3)

            model_results = [
                ("EfficientNet-B0", results["b0_probability"]),
                ("DINOv2", results["dino_probability"]),
                ("B0 + DINOv2 Fusion", results["fusion_probability"]),
            ]

            for column, (name, probability) in zip(
                (col1, col2, col3),
                model_results,
            ):

                with column:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-label">{name}</div>
                            <div class="result-value">
                                {probability * 100:.1f}%
                            </div>
                            <div class="result-note">
                                Positive-class probability
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.metric(
                "Frames analyzed",
                results["frames_analyzed"],
            )

            with st.expander("How the estimate is calculated"):

                st.write(
                    f"EfficientNet-B0 probability: "
                    f"{results['b0_probability'] * 100:.2f}%"
                )

                st.write(
                    f"DINOv2 probability: "
                    f"{results['dino_probability'] * 100:.2f}%"
                )

                st.write(
                    f"B0 + DINOv2 Fusion probability: "
                    f"{results['fusion_probability'] * 100:.2f}%"
                )

                st.write(
                    "Fusion formula: "
                    "35% EfficientNet-B0 + 65% DINOv2."
                )

                st.write(
                    "Both models independently extract features "
                    "from 10 evenly sampled video frames. "
                    "Their video-level probabilities are then combined."
                )

                st.caption(
                    f"Inference device: {results['device']}"
                )

    else:

        st.info(
            "Your analysis will appear here after you upload "
            "and analyze your media."
        )

        st.caption("Select Image or Video to begin.")


st.markdown("---")

st.markdown(
    """
    <div class="footer">
        TrueLens · Research prototype · AI-generated media detection
    </div>
    """,
    unsafe_allow_html=True,
)
