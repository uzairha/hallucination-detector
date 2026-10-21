from utils.chunker import chunk_documents


def make_doc(n_words, filename="doc.txt"):
    return {"filename": filename, "text": " ".join(f"w{i}" for i in range(n_words))}


def test_chunk_size_and_overlap():
    chunks = chunk_documents([make_doc(25)], chunk_size=10, overlap=3)

    # Windows start every 10 - 3 = 7 words: 0, 7, 14, 21
    assert [c["text"].split()[0] for c in chunks] == ["w0", "w7", "w14", "w21"]
    assert [len(c["text"].split()) for c in chunks] == [10, 10, 10, 4]
    assert [c["chunk_id"] for c in chunks] == [0, 1, 2, 3]
    assert all(c["filename"] == "doc.txt" for c in chunks)

    # Each chunk shares its last `overlap` words with the start of the next one
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev["text"].split()[-3:] == nxt["text"].split()[:3]


def test_short_doc_gives_one_chunk():
    chunks = chunk_documents([make_doc(5)], chunk_size=10, overlap=3)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "w0 w1 w2 w3 w4"
    assert chunks[0]["chunk_id"] == 0


def test_empty_doc_gives_no_chunks():
    assert chunk_documents([{"filename": "empty.txt", "text": ""}]) == []


def test_chunk_ids_restart_per_document():
    chunks = chunk_documents([make_doc(5, "a.txt"), make_doc(5, "b.txt")], chunk_size=10, overlap=3)
    assert [(c["filename"], c["chunk_id"]) for c in chunks] == [("a.txt", 0), ("b.txt", 0)]
