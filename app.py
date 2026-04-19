"""
IMDB Sentiment Classifier — Streamlit App
==========================================
A local Streamlit application that classifies movie review sentiment
(Positive / Negative) using a pre-trained SimpleRNN model.

Usage:
    streamlit run app.py
"""

import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MODEL_PATH_KERAS = "Simple_RNN_IMDB.keras"
MODEL_PATH_H5 = "Simple_RNN_IMDB.h5"
MAX_SEQUENCE_LEN = 500
VOCAB_SIZE = 10_000
POSITIVE_THRESHOLD = 0.5

# ──────────────────────────────────────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🎬 IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
)


# ──────────────────────────────────────────────────────────────────────────────
# Cached loaders (run once per session)
# ──────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model…")
def load_sentiment_model() -> tf.keras.Model:
    """Load the pre-trained SimpleRNN model from disk."""
    if os.path.exists(MODEL_PATH_KERAS):
        return load_model(MODEL_PATH_KERAS)
    if os.path.exists(MODEL_PATH_H5):
        return load_model(MODEL_PATH_H5)
    raise FileNotFoundError(
        f"Model file not found. Expected '{MODEL_PATH_KERAS}' or "
        f"'{MODEL_PATH_H5}' in the project root directory."
    )


@st.cache_resource(show_spinner="Loading word index…")
def load_word_index() -> dict:
    """Download and return the IMDB word-to-index mapping."""
    return imdb.get_word_index()


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing & inference helpers
# ──────────────────────────────────────────────────────────────────────────────


def preprocess_text(text: str, word_index: dict) -> np.ndarray:
    """
    Convert a raw review string into a padded integer sequence.

    Args:
        text:       Raw review text from the user.
        word_index: IMDB word-to-index mapping.

    Returns:
        Padded numpy array of shape (1, MAX_SEQUENCE_LEN).
    """
    words = text.lower().split()
    # +3 offset: 0=padding, 1=start, 2=unknown, 3=unused
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=MAX_SEQUENCE_LEN)
    return padded


def predict_sentiment(
    review: str, model: tf.keras.Model, word_index: dict
) -> tuple[str, float]:
    """
    Predict sentiment for a movie review.

    Args:
        review:     Raw review string.
        model:      Loaded Keras model.
        word_index: IMDB word-to-index mapping.

    Returns:
        Tuple of (sentiment_label, confidence_score).
    """
    processed = preprocess_text(review, word_index)
    score: float = float(model.predict(processed, verbose=0)[0][0])
    label = "Positive" if score >= POSITIVE_THRESHOLD else "Negative"
    return label, score


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────


def render_result(label: str, score: float) -> None:
    """Render the sentiment result card."""
    is_positive = label == "Positive"
    emoji = "✅" if is_positive else "❌"
    color = "#28a745" if is_positive else "#dc3545"
    confidence_pct = score * 100 if is_positive else (1 - score) * 100

    st.markdown(
        f"""
        <div style="
            background-color: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            margin-top: 20px;
        ">
            <h2 style="color: {color}; margin: 0;">{emoji} {label}</h2>
            <p style="color: #555; margin-top: 8px; font-size: 1rem;">
                Confidence: <strong>{confidence_pct:.2f}%</strong>
                &nbsp;|&nbsp; Raw score: <code>{score:.4f}</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🎬 IMDB Movie Review Sentiment Analyzer")
    st.markdown(
        "Enter a movie review below and the **SimpleRNN model** will classify "
        "it as **Positive** or **Negative** with a confidence score."
    )
    st.divider()

    # ── Load resources ────────────────────────────────────────────────────────
    try:
        model = load_sentiment_model()
        word_index = load_word_index()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    # ── Input form ────────────────────────────────────────────────────────────
    review_text = st.text_area(
        label="Your Movie Review",
        placeholder=(
            "e.g., The movie was absolutely fantastic! The acting was superb "
            "and the plot kept me on the edge of my seat throughout…"
        ),
        height=160,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        classify_btn = st.button("🔍 Classify Sentiment", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.rerun()

    # ── Prediction ────────────────────────────────────────────────────────────
    if classify_btn:
        if not review_text.strip():
            st.warning("⚠️ Please enter a review before classifying.")
        else:
            with st.spinner("Analyzing sentiment…"):
                label, score = predict_sentiment(review_text, model, word_index)
            render_result(label, score)

    # ── Sidebar info ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown(
            """
            **Model:** SimpleRNN (Keras / TensorFlow)  
            **Dataset:** IMDB Movie Reviews (50k samples)  
            **Vocabulary:** Top 10,000 words  
            **Max sequence length:** 500 tokens  
            **Validation accuracy:** ~77%

            ---
            **Architecture**
            ```
            Embedding(10000, 128)
                ↓
            SimpleRNN(128, relu)
                ↓
            Dense(1, sigmoid)
            ```
            ---
            **Tip:** For best results, write reviews in plain English
            with at least 20–30 words.
            """
        )


if __name__ == "__main__":
    main()
