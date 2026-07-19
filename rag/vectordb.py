"""
vectordb.py
-----------
Responsibility: Build and manage a local FAISS vector index
for efficient similarity search over embedded document chunks.

Why FAISS?
    - Facebook AI Similarity Search — industry standard for local vector search.
    - No server, no cloud, runs entirely on your machine.
    - IndexFlatL2 does exact nearest-neighbor search using L2 (Euclidean) distance.
      Simple and accurate for our scale (hundreds to low thousands of chunks).

What it stores:
    - FAISS index: the actual vectors for similarity search.
    - Metadata list: the chunk dicts (filename, filetype, chunk_index, text)
      stored in parallel so we can return full chunk info after a search.

Saves to disk:
    - rag/faiss_index/index.faiss  — the FAISS index binary
    - rag/faiss_index/metadata.pkl — the parallel metadata list
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np


# ── PATHS ─────────────────────────────────────────────────────────────────────

INDEX_DIR      = Path("rag/faiss_index")
INDEX_PATH     = INDEX_DIR / "index.faiss"
METADATA_PATH  = INDEX_DIR / "metadata.pkl"


# ── BUILD INDEX ───────────────────────────────────────────────────────────────

def build_index(embedded_chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """
    Build a FAISS index from a list of embedded chunk dicts.

    Args:
        embedded_chunks: List of chunk dicts with an 'embedding' key
                         (output of embedder.embed_chunks()).

    Returns:
        Tuple of (faiss_index, metadata_list) where:
            - faiss_index    : the FAISS IndexFlatL2 object
            - metadata_list  : list of chunk dicts without the 'embedding' key
                               (stored separately for retrieval)

    Raises:
        ValueError: if embedded_chunks is empty.
    """
    if not embedded_chunks:
        raise ValueError("[VectorDB] Cannot build index from empty chunk list.")

    # Extract vectors as a float32 numpy matrix (FAISS requirement)
    vectors = np.array(
        [chunk["embedding"] for chunk in embedded_chunks],
        dtype=np.float32
    )

    dim   = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    # Store metadata without embeddings (saves memory — FAISS holds the vectors)
    metadata = [
        {k: v for k, v in chunk.items() if k != "embedding"}
        for chunk in embedded_chunks
    ]

    print(f"[VectorDB] Index built — {index.ntotal} vector(s), dim={dim}.")
    return index, metadata


# ── SAVE ──────────────────────────────────────────────────────────────────────

def save_index(index: faiss.Index, metadata: list[dict]) -> None:
    """
    Save the FAISS index and metadata to disk.

    Args:
        index:    FAISS index object.
        metadata: Parallel list of chunk dicts (without embeddings).
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[VectorDB] Saved index  → {INDEX_PATH}")
    print(f"[VectorDB] Saved metadata → {METADATA_PATH}")


# ── LOAD ──────────────────────────────────────────────────────────────────────

def load_index() -> tuple[faiss.Index, list[dict]]:
    """
    Load a previously saved FAISS index and metadata from disk.

    Returns:
        Tuple of (faiss_index, metadata_list).

    Raises:
        FileNotFoundError: if index or metadata files do not exist.
    """
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"[VectorDB] Index not found at '{INDEX_PATH}'. "
            "Run the RAG pipeline first to build the index."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"[VectorDB] Metadata not found at '{METADATA_PATH}'. "
            "Run the RAG pipeline first to build the index."
        )

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)

    print(f"[VectorDB] Loaded index — {index.ntotal} vector(s).")
    return index, metadata


# ── INDEX EXISTS ──────────────────────────────────────────────────────────────

def index_exists() -> bool:
    """
    Check whether a saved index already exists on disk.
    Used by pipeline.py to decide whether to rebuild or reuse.

    Returns:
        True if both index and metadata files exist, False otherwise.
    """
    return INDEX_PATH.exists() and METADATA_PATH.exists()


# ── CLEAR INDEX ───────────────────────────────────────────────────────────────

def clear_index() -> None:
    """
    Delete the saved index and metadata files from disk.
    Used when the knowledge/ folder changes and the index needs rebuilding.
    """
    if INDEX_PATH.exists():
        os.remove(INDEX_PATH)
        print(f"[VectorDB] Deleted {INDEX_PATH}")

    if METADATA_PATH.exists():
        os.remove(METADATA_PATH)
        print(f"[VectorDB] Deleted {METADATA_PATH}")

    print("[VectorDB] Index cleared.")