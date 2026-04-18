import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Where we save the index and chunk metadata
INDEX_PATH = "data/index.faiss"
CHUNKS_PATH = "data/chunks.json"

# Load the embedding model once (downloads on first run, cached after)
model = SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks: List[Dict[str, str]]) -> None:
    """
    Embeds all chunks and saves them to a FAISS index on disk.
    Only needs to run once, or when your documents change.
    """
    texts = [chunk["text"] for chunk in chunks]

    print("Embedding chunks... (this may take a moment on first run)")
    embeddings = model.encode(texts, show_progress_bar=True)

    # FAISS expects float32 arrays
    embeddings = np.array(embeddings).astype("float32")

    # Create a flat (exact search) FAISS index using cosine-like L2 distance
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
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
        raise FileNotFoundError("Index not found. Run build_index() first.")

    index = faiss.read_index(INDEX_PATH)

    with open(CHUNKS_PATH, "r") as f:
        chunks = json.load(f)

    return index, chunks


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single query string so it can be compared against the index.
    """
    embedding = model.encode([query])
    return np.array(embedding).astype("float32")
