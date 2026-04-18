from typing import List, Dict
from utils.embedder import load_index, embed_query


def retrieve(claim: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Finds the top_k most relevant chunks for a given claim.
    Returns a list of chunk dicts with an added 'score' key.
    """
    index, chunks = load_index()

    # Embed the claim into a vector
    query_vector = embed_query(claim)

    # Search the FAISS index — returns distances and positions
    distances, indices = index.search(query_vector, k=min(top_k, len(chunks)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 if there aren't enough results
            continue

        chunk = chunks[idx].copy()
        chunk["score"] = float(dist)  # lower = more similar
        results.append(chunk)

    return results
