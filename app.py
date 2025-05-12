import streamlit as st
import pandas as pd
import pickle
from utils.pdf_utils import extract_text_from_pdf
from utils.preprocessing import preprocess_text, find_keywords_with_context
import nltk
import os

# 1. Create a local nltk_data dir (once)
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)

# 2. Tell NLTK to use it
nltk.data.path.append(nltk_data_path)

# 3. Download resources to that local path
nltk.download("stopwords", download_dir=nltk_data_path)
nltk.download("punkt", download_dir=nltk_data_path)
nltk.download("wordnet", download_dir=nltk_data_path)
nltk.download("omw-1.4", download_dir=nltk_data_path)

# Load model
model = pickle.load(open("model_xgb.sav", "rb"))

# UI - Sidebar
st.sidebar.title("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# UI - Main
st.title("S-LCA Text-Based Compliance Classification")

if uploaded_file:
    raw_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Text Preview")
    st.write(raw_text[:1000] + "...")

    preprocessed_text, cleaned_sentences = preprocess_text(raw_text)

    # Define your S-LCA topics
    topics = ["freedom of association", "collective bargaining"]  # Add more as needed

    results = find_keywords_with_context(cleaned_sentences, topics)

    if results:
        df = pd.DataFrame(results, columns=["lemmatize"])
        st.subheader("Sentences with Keywords")
        st.dataframe(df)

        if st.button("Predict Compliance"):
            predictions = model.predict(df["lemmatize"])
            df["Performance Score"] = predictions
            st.subheader("Prediction Results")
            st.dataframe(df)
    else:
        st.warning("No relevant sentences with specified keywords were found.")
