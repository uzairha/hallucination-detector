from functools import lru_cache
from typing import List, Dict
from utils.embedder import load_index, embed_query


@lru_cache(maxsize=1)
def _get_index():
    """
    Loads the FAISS index and chunks once and reuses them for every claim.
    A missing index raises FileNotFoundError, which lru_cache doesn't cache,
    so the next call will try again after you run build_index.py.
    """
    return load_index()


def retrieve(claim: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Finds the top_k most relevant chunks for a given claim.
    Returns a list of chunk dicts with an added 'score' key.
    """
    index, chunks = _get_index()

    # Embed the claim into a vector
    query_vector = embed_query(claim)

    # Search the FAISS index — returns cosine similarities and positions
    scores, indices = index.search(query_vector, k=min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 if there aren't enough results
            continue

        chunk = chunks[idx].copy()
        chunk["score"] = float(score)  # cosine similarity: higher = more similar
        results.append(chunk)

    return results
