import streamlit as st
import pickle
import numpy as np

from src.preprocess import clean_text


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_artifacts():
    with open("data/cleaned/tfidf_bigram.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("data/cleaned/logreg_model.pkl", "rb") as f:
        model = pickle.load(f)

    return vectorizer, model


vectorizer, model = load_artifacts()


# -----------------------------
# UI
# -----------------------------
st.title("🎬 Movie Review Sentiment Analyzer")
st.caption("Paste a movie review and see whether it sounds positive or negative.")


review = st.text_area(
    "Enter a movie review",
    placeholder="I expected this movie to be great, but it was painfully slow...",
    height=180
)

analyze = st.button("Analyze Sentiment 🎯")


# -----------------------------
# Prediction logic
# -----------------------------
if analyze and review.strip():

    # Preprocess
    cleaned = clean_text(review)

    # Vectorize
    X = vectorizer.transform([cleaned])

    # Predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    confidence = np.max(prob)

    # Display result
    st.divider()
    if confidence < 0.6:
        st.warning("😐 Uncertain / Mixed Review")
    elif pred == 1:
        st.success("😄 Positive Review")
    else:
        st.error("😡 Negative Review")

    st.progress(confidence)
    st.caption(f"Model confidence: {confidence:.2%}")

elif analyze:
    st.warning("Please enter a review first.")
    

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Built using classic NLP techniques (TF-IDF) and Logistic Regression on the IMDb dataset. Longer reviews usually lead to more confident predictions."
)
