import streamlit as st
import pandas as pd
import tempfile
from process import main  # Import main function from processing module
from match import database_match # Import matching function from match module
import pdfplumber
import re
from dotenv import load_dotenv
import os
import openai
from openai import OpenAI
import json
import faiss
import numpy as np
from typing import Optional, List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.title("📄 Arthur Weber PDF Processing App")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Run processing function
    with st.spinner("Processing PDF with OpenAI..."):
        try:
            df = main(temp_pdf_path)
            st.success("Item extraction complete!")
    
        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Run database matching function   
    with st.spinner("Matching items with database..."):
        try:
            matched_df = database_match(df)
            st.success("Item matching complete! Your results are ready below.")
            st.dataframe(matched_df)
    
            # Download button
            csv = matched_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("Download results as CSV", csv, "output.csv", "text/csv")
        except Exception as e:
            st.error(f"An error occurred: {e}")