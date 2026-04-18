import os
from typing import List, Dict


def load_documents(docs_dir: str = "data/docs") -> List[Dict[str, str]]:
    """
    Reads all .txt files from the given folder.
    Returns a list of dicts, each with 'filename' and 'text'.
    """
    documents = []

    # Check the folder actually exists before trying to read it
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"Could not find document folder: {docs_dir}")

    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)

            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()

            # Skip files that are empty
            if text:
                documents.append({"filename": filename, "text": text})

    print(f"Loaded {len(documents)} document(s) from '{docs_dir}'")
    return documents
