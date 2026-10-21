# AI Hallucination Detector

A Streamlit app that fact-checks AI-generated answers against a curated CS and AI knowledge base. Paste any AI response, and the app extracts factual claims, retrieves relevant evidence via semantic search, and verdicts each claim as **Supported**, **Weakly Supported**, or **Unsupported**.

## How It Works

1. **Claim Extraction** — GPT-4o-mini parses the AI answer into discrete factual claims
2. **Semantic Retrieval** — Each claim is embedded and matched against a FAISS index built from the knowledge base
3. **Verification** — GPT-4o-mini evaluates each claim against retrieved evidence
4. **Scoring** — A hallucination score (0–1) is computed and a verdict is rendered

## Knowledge Base

Covers 19 core CS and AI topics: algorithms, data structures, databases, operating systems, networking, software engineering, testing, Git and CI/CD, cloud and DevOps, web development, cybersecurity, machine learning, deep learning, transformers, LLMs, embeddings, vector databases, RAG, and AI safety. Stored as `.txt` files in `data/docs/`, with references for each file in `data/docs/SOURCES.md`.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Build the FAISS index (first run, or after changing data/docs/)
python build_index.py

# Run the app
streamlit run app.py
```

## Project Structure

```
hallucination-detector/
├── app.py                  # Streamlit UI
├── build_index.py          # Rebuilds the FAISS index from data/docs/
├── utils/
│   ├── loader.py           # Reads .txt files from data/docs/
│   ├── chunker.py          # Splits docs into overlapping word chunks
│   ├── embedder.py         # Embeds chunks, builds FAISS index
│   ├── retriever.py        # Searches FAISS for relevant chunks
│   ├── claims.py           # Extracts factual claims via LLM
│   ├── verifier.py         # Verdicts each claim against evidence
│   └── scorer.py           # Computes overall hallucination score
├── data/
│   ├── docs/               # Knowledge base .txt files
│   ├── chunks.json         # Chunked text cache
│   └── index.faiss         # FAISS vector index
└── requirements.txt
```

## Tech Stack

- [Streamlit](https://streamlit.io) — UI
- [OpenAI GPT-4o-mini](https://platform.openai.com) — Claim extraction & verification
- [sentence-transformers](https://www.sbert.net) — Text embeddings (`all-MiniLM-L6-v2`)
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
