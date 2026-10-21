import os
import json
import numpy as np
import faiss
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Where we save the index and chunk metadata
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.json"


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """
    Loads the embedding model on first use (downloads on first run, cached after).
    Lazy so that importing this module doesn't load the model.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks: List[Dict[str, str]]) -> None:
    """
    Embeds all chunks and saves them to a FAISS index on disk.
    Only needs to run once, or when your documents change.
    """
    texts = [chunk["text"] for chunk in chunks]

    print("Embedding chunks... (this may take a moment on first run)")
    # Normalize to unit length so inner product == cosine similarity
    embeddings = get_model().encode(texts, show_progress_bar=True, normalize_embeddings=True)

    # FAISS expects float32 arrays
    embeddings = np.array(embeddings).astype("float32")

    # Flat (exact search) inner-product index; on unit vectors this is cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save the FAISS index to disk
    faiss.write_index(index, INDEX_PATH)

    # Save the chunk metadata so we can look up text by index later
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)

    print(f"Saved index with {index.ntotal} vectors to '{INDEX_PATH}'")


def load_index():
    """
    Loads the saved FAISS index and chunk metadata from disk.
    Returns (index, chunks).
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Index not found. Run `python build_index.py` first.")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    return index, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string so it can be compared against the index.
    Normalized the same way as the indexed chunks.
    """
    embedding = get_model().encode([query], normalize_embeddings=True)
    return np.array(embedding).astype("float32")
