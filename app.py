import re
from pathlib import Path

import joblib
import nltk
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="BOW Sentiment Analyzer",
    page_icon="📝",
    layout="centered",
)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 42px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            color: #6b7280;
            font-size: 17px;
            margin-bottom: 25px;
        }

        .card {
            padding: 18px;
            border-radius: 14px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            margin: 15px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# NLTK resources
# ---------------------------------------------------------
@st.cache_resource
def load_nltk_resources():
    for package in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(package, quiet=True)

    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    return set(stopwords.words("english")), WordNetLemmatizer()


stop_words, lemmatizer = load_nltk_resources()


# ---------------------------------------------------------
# SAME preprocessing used in your BOW notebook
# ---------------------------------------------------------
def preprocess_text(text):
    text = text.lower()

    text = re.sub(r"[^A-Za-z\s]", "", text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


# ---------------------------------------------------------
# Load BOW artifact
# ---------------------------------------------------------
@st.cache_resource
def load_artifact():
    model_path = Path("bow_sentiment_model.pkl")

    if not model_path.exists():
        st.error(
            "bow_sentiment_model.pkl was not found. "
            "Place it in the same folder as app.py."
        )
        st.stop()

    return joblib.load(model_path)


artifact = load_artifact()

vectorizer = artifact["vectorizer"]
model = artifact["model"]
label_encoder = artifact["label_encoder"]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="title">📝 BOW Sentiment Analyzer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Bag of Words + Logistic Regression"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
        <b>Model:</b> CountVectorizer(max_features=5000)
        <br>
        <b>Classifier:</b> Logistic Regression
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Input
# ---------------------------------------------------------
review = st.text_area(
    "Enter a movie review",
    height=190,
    placeholder="Example: The movie was fantastic and the acting was brilliant!",
)


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a review.")

    else:
        cleaned_review = preprocess_text(review)

        transformed_text = vectorizer.transform([cleaned_review])

        prediction = model.predict(transformed_text)[0]

        predicted_label = label_encoder.inverse_transform([prediction])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed_text)[0]
            confidence = probabilities[prediction] * 100
        else:
            probabilities = None
            confidence = None

        if predicted_label.lower() == "positive":
            st.success(f"### 😊 POSITIVE")

        else:
            st.error(f"### 😞 NEGATIVE")

        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

            positive_index = label_encoder.transform(["positive"])[0]
            negative_index = label_encoder.transform(["negative"])[0]

            col1, col2 = st.columns(2)

            with col1:
                st.write(
                    f"**Positive:** "
                    f"{probabilities[positive_index] * 100:.2f}%"
                )

            with col2:
                st.write(
                    f"**Negative:** "
                    f"{probabilities[negative_index] * 100:.2f}%"
                )

            st.progress(min(confidence / 100, 1.0))


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("NLP Sentiment Analysis • Bag of Words • Scikit-learn • Streamlit")
