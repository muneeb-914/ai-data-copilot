"""
retriever.py
------------
Responsibility: Embed the user's query, search the FAISS index,
and return the most relevant document chunks.

Flow:
    1. Takes a plain text query string.
    2. Embeds it using embedder.embed_text() — same model used for chunks.
    3. Searches the FAISS index for the closest vectors (L2 distance).
    4. Returns the top-k matching chunk dicts with their similarity scores.

Returns a list of result dicts:
    {
        "filename":    "report.pdf",
        "filetype":    "pdf",
        "chunk_index": 2,
        "text":        "relevant chunk text...",
        "score":       0.312    # L2 distance — lower means more similar
    }
"""

from __future__ import annotations

import numpy as np
import faiss

from rag.embedder import embed_text


# ── DEFAULT ───────────────────────────────────────────────────────────────────

DEFAULT_TOP_K = 5   # number of chunks to return per query


# ── RETRIEVE ──────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    index: faiss.Index,
    metadata: list[dict],
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a given query.

    Args:
        query:    The user's question or search string.
        index:    FAISS index built by vectordb.build_index().
        metadata: Parallel metadata list from vectordb (same order as index).
        top_k:    Number of top results to return. Capped at total chunk count.

    Returns:
        List of chunk dicts sorted by relevance (most relevant first),
        each with an added 'score' key (L2 distance — lower = more similar).

    Raises:
        ValueError: if query is empty.
    """
    if not query or not query.strip():
        raise ValueError("[Retriever] Query cannot be empty.")

    if index.ntotal == 0:
        print("[Retriever] Index is empty. No results.")
        return []

    # Cap top_k so we never ask FAISS for more results than exist
    top_k = min(top_k, index.ntotal)

    # Embed the query — must use the same model used for chunks
    print(f"[Retriever] Embedding query: '{query[:80]}'")
    query_vector = np.array([embed_text(query)], dtype=np.float32)

    # Search the FAISS index
    # distances: shape (1, top_k) — L2 distances to each result
    # indices:   shape (1, top_k) — positions in the metadata list
    distances, indices = index.search(query_vector, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            # FAISS returns -1 when fewer results exist than top_k
            continue
        chunk = metadata[idx]
        result = {**chunk, "score": float(dist)}
        results.append(result)

    print(f"[Retriever] Found {len(results)} result(s) for query.")
    return results