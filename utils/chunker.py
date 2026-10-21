from typing import List, Dict


def chunk_documents(
    documents: List[Dict[str, str]],
    chunk_size: int = 200,
    overlap: int = 40
) -> List[Dict[str, str]]:
    """
    Splits each document into smaller overlapping chunks of words.
    Returns a list of dicts with 'filename', 'chunk_id', and 'text'.
    """
    # Each step must move forward, otherwise the loop below never ends
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError(
            f"Need chunk_size > 0 and 0 <= overlap < chunk_size (got {chunk_size}, {overlap})"
        )

    all_chunks = []

    for doc in documents:
        words = doc["text"].split()
        total_words = len(words)
        chunk_id = 0
        start = 0

        while start < total_words:
            end = start + chunk_size

            # Slice out this chunk's words and rejoin into a string
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            all_chunks.append({
                "filename": doc["filename"],
                "chunk_id": chunk_id,
                "text": chunk_text
            })

            chunk_id += 1

            # This chunk reached the end of the doc; another one would only repeat its tail
            if end >= total_words:
                break

            # Move forward by (chunk_size - overlap) so chunks share some words
            start += chunk_size - overlap

    print(f"Created {len(all_chunks)} chunk(s) from {len(documents)} document(s)")
    return all_chunks
