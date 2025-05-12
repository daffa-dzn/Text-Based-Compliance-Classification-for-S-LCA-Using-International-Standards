import streamlit as st
import pandas as pd
import os
import nltk
import pickle

from nltk.corpus import stopwords
from utils.pdf_utils import extract_text_from_pdf
from utils.preprocessing import preprocess_text, find_keywords_with_context

# === NLTK Setup ===
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Only download if missing
def safe_download(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)
        
resources = [
    "corpora/stopwords",
    "tokenizers/punkt",
    "corpora/wordnet",
    "corpora/omw-1.4"
]

for res in resources:
    safe_download(res)

# === Load ML Model ===
model_path = "model_xgb.sav"
model = pickle.load(open(model_path, "rb"))

# === Streamlit UI ===
st.set_page_config(page_title="S-LCA Compliance Classifier", layout="wide")
st.title("Text-Based Compliance Classification for S-LCA")

# Sidebar: File upload
st.sidebar.header("Step 1: Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Extract raw text
    raw_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Step 2: Extracted Text Preview")
    st.write(raw_text[:1000] + "...")  # Preview only

    # Preprocess
    st.subheader("Step 3: Preprocessing & Lemmatization")
    cleaned_full_text, cleaned_sentences = preprocess_text(raw_text)
    st.success("Preprocessing complete.")

    # Keyword config (can expand)
    topics = ["freedom of association", "collective bargaining"]

    # Find sentences with keywords
    results = find_keywords_with_context(cleaned_sentences, topics)

    if results:
        df = pd.DataFrame(results, columns=["lemmatize"])
        st.subheader("Step 4: Sentences with Keywords")
        st.dataframe(df, use_container_width=True)

        # Predict button
        if st.button("Step 5: Predict Compliance"):
            df["Performance Score"] = model.predict(df["lemmatize"])
            st.subheader("Step 6: Prediction Results")
            st.dataframe(df, use_container_width=True)
    else:
        st.warning("No keyword-related sentences found.")
else:
    st.info("Upload a PDF file from the sidebar to begin.")
