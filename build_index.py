from utils.loader import load_documents
from utils.chunker import chunk_documents
from utils.embedder import build_index


def main() -> None:
    """
    Rebuilds the FAISS index from scratch: load docs -> chunk -> embed + save.
    Run this whenever the files in data/docs/ change.
    """
    documents = load_documents()
    chunks = chunk_documents(documents)
    build_index(chunks)


if __name__ == "__main__":
    main()
