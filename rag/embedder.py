"""
embedder.py
-----------
Responsibility: Generate semantic embeddings for each text chunk
using fastembed (lightweight, no PyTorch required).

Why fastembed?
    - No PyTorch dependency — much smaller install.
    - Uses ONNX Runtime under the hood (fast and efficient).
    - 'BAAI/bge-small-en-v1.5' produces 384-dim embeddings,
      same dimension as all-MiniLM-L6-v2 from sentence-transformers.

Returns:
    A list of dicts — original chunk dict + an 'embedding' key:
    {
        "filename":    "report.pdf",
        "filetype":    "pdf",
        "chunk_index": 0,
        "text":        "chunk text...",
        "embedding":   [0.123, -0.456, ...]   # 384 floats
    }
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding


# ── MODEL ─────────────────────────────────────────────────────────────────────

MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Loaded once at module level — avoids reloading the model on every call.
_model: TextEmbedding | None = None


def get_model() -> TextEmbedding:
    """
    Load and return the embedding model.
    Uses a module-level singleton so the model is only loaded once per session.
    """
    global _model
    if _model is None:
        print(f"[Embedder] Loading model '{MODEL_NAME}'...")
        _model = TextEmbedding(model_name=MODEL_NAME)
        print(f"[Embedder] Model loaded successfully.")
    return _model


# ── EMBED TEXT ────────────────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Generate a semantic embedding for a single string.
    Used for embedding user queries at retrieval time.

    Args:
        text: Any string to embed.

    Returns:
        List of floats (384 dimensions).
    """
    model = get_model()
    # fastembed.embed() returns a generator — we take the first (and only) result
    vector = list(model.embed([text]))[0]
    return vector.tolist()


# ── EMBED CHUNKS ──────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate semantic embeddings for a list of chunk dicts from splitter.py.
    Embeddings are generated in one batched call for efficiency.

    Args:
        chunks: List of chunk dicts with at least a 'text' key.

    Returns:
        Same list of dicts with an 'embedding' key added to each.
        Original chunk dicts are not mutated — new dicts are returned.
    """
    if not chunks:
        print("[Embedder] No chunks to embed. Returning empty.")
        return []

    model  = get_model()
    texts  = [chunk["text"] for chunk in chunks]

    print(f"[Embedder] Embedding {len(texts)} chunk(s)...")

    # fastembed returns a generator — convert to list
    vectors = list(model.embed(texts))

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunk = {**chunk, "embedding": vector.tolist()}
        embedded_chunks.append(embedded_chunk)

    print(f"[Embedder] Done. {len(embedded_chunks)} chunk(s) embedded.")
    return embedded_chunks