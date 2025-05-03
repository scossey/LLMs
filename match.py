import openai
import faiss
import numpy as np
import pandas as pd
import re
from dotenv import load_dotenv
import os
import json
from typing import Optional, List, Dict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

load_dotenv()  # Load variables from .env into the environment
openai_api_key = os.getenv("OPENAI_API_KEY")  # Get the API key
openai.api_key = openai_api_key

nltk.download('stopwords')
nltk.download('punkt')
german_stopwords = set(stopwords.words('german'))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.strip()
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in german_stopwords]
    processed_text = " ".join(filtered_words)
    return processed_text

def embed_text(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed a list of texts using OpenAI text-embedding-3-small model in batches.

    Args:
        texts (list[str]): List of input texts to embed.
        batch_size (int): How many texts to send per batch (default 100).

    Returns:
        np.ndarray: All embeddings as a (n_samples, embedding_dim) array.
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings = [data.embedding for data in response.data]
        all_embeddings.extend(embeddings)

    return np.array(all_embeddings).astype("float32")
    
# Build the FAISS index
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product (cosine similarity after normalization)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

# finds top db matches

def find_matches(extracted_items: List[str], index: faiss.IndexFlatIP, database_items: List[str], threshold: float = 0.40, top_k: int = 5):
    """
    Finds matches for a batch of query strings.

    Args:
        extracted_items: List of extracted items from tender documents.
        index: FAISS index.
        database_items: List of catalog descriptions.
        threshold: Minimum similarity score to consider a match.
        top_k: How many candidates to retrieve per query.

    Returns:
        List of lists of matches. Each inner list corresponds to a query.
    """
    # Embed all queries
    processed_items = [preprocess_text(item) for item in extracted_items]
    query_embeddings = embed_text(processed_items)
    faiss.normalize_L2(query_embeddings)

    # Search all queries
    distances, indices = index.search(query_embeddings, top_k)

    batch_matches = []

    # Process each query separately
    for query_idx in range(len(extracted_items)):
        matches = []
        for candidate_idx, score in zip(indices[query_idx], distances[query_idx]):
            if score >= threshold:
                matches.append({
                    "matched_item": database_items[candidate_idx],
                    "similarity_score": float(score)
                })
        batch_matches.append(matches if matches else None)

    return batch_matches

def verify_match_llm(extracted_item: str, candidate_item: str) -> bool:
    client = openai.OpenAI()
    prompt = f"""
    You are verifying potential product matches based on their descriptions ONLY.

    Disregard any pre-calculated similarity scores between the following items. Your decision should be based solely on the textual information provided in the "Extracted Tender Item" and the "Database Catalog Item".

    Characteristics such as material and size indicators are very important. Any indicators of primary usage context and any mentioned standards (SN, SDR) should also be taken into consideration. Determine if the descriptions indicate the same type of product based on these key attributes.

    Extracted Tender Item:
    "{extracted_item}"

    Database Catalog Item:
    "{candidate_item}"

    Answer:
    - "Yes" if, based on the textual descriptions alone, they strongly suggest the same type of product, considering potential variations in description.
    - "No" only if the textual descriptions clearly indicate fundamentally different products or if critical specifications demonstrably conflict.

    Respond only with "Yes" or "No".
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"Error during verification: {e}")
        return False
 
# need to loop through the list of candidate matches for each item
# can change to keep all llm "yes" candidates
def find_verified_match(extracted_item: str, candidates: List[Dict]) -> Optional[Dict]:
    """
    Loop through candidate matches and verify each one using LLM.
    
    Args:
        extracted_item: The tender item extracted from document.
        candidates: List of candidate dicts with "matched_item" and "similarity_score".

    Returns:
        The first verified match (dict) or None if no verified match.
    """
    for candidate in candidates:
        catalog_item = candidate["matched_item"]
        if verify_match_llm(extracted_item, catalog_item):
            return candidate  # Return the first verified match
    return None  # No verified matches

def find_verified_matches(extracted_items: List[str], batch_matches: List[List[Dict]]) -> List[Optional[Dict]]:
    """
    For a list of extracted items and their candidate matches,
    verify each and find the first valid match for each query.

    Args:
        extracted_items: List of tender items extracted.
        candidates_batch: List of candidate match lists, one list per tender item. Output from find_matches()

    Returns:
        List of verified matches (one per tender item), or None if no verified match.
    """
    verified_matches = []

    for extracted_item, candidates in zip(extracted_items, batch_matches):
        if candidates:
            verified_match = find_verified_match(extracted_item, candidates)
            verified_matches.append(verified_match)
        else:
            verified_matches.append(None)

    return verified_matches


def database_match(extracted_items_df: pd.DataFrame) -> pd.DataFrame:

    # load embedded database item descriptions from file
    with open("database_embeddings_processed.json", "r") as f:
        loaded_data = json.load(f)

    db_embeddings = np.array([item['embedding'] for item in loaded_data]).astype("float32")
    db_items = [item['text'] for item in loaded_data]
    faiss_index = build_faiss_index(db_embeddings)

    # merge columns to give more item context
    extracted_items_df["item_section"] = extracted_items_df["item"] + " " + extracted_items_df["section"]

    matches = find_matches(extracted_items_df["item_section"], faiss_index, db_items)
    llm_verified_matches = find_verified_matches(extracted_items_df["item_section"], matches)

    # add matches to extacted items df

    matched_items = []
    similarity_scores = []

    for match in llm_verified_matches:
        if match is not None:
            matched_items.append(match["matched_item"])
            similarity_scores.append(match["similarity_score"])
        else:
            matched_items.append(None)
            similarity_scores.append(None)

    # Now: Add to the DataFrame
    extracted_items_df["matched_catalog_item"] = matched_items
    extracted_items_df["similarity_score"] = similarity_scores

    return extracted_items_df




