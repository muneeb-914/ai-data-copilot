"""
splitter.py
-----------
Responsibility: Split raw document text into smaller, overlapping chunks
suitable for embedding and retrieval.

Why chunking?
    Embedding models have a token limit (~512 tokens for most sentence-transformers).
    Full documents are too large to embed as one unit.
    Smaller chunks = more precise retrieval.

Why overlap?
    If a key sentence sits at the boundary of two chunks, overlap ensures
    it appears fully in at least one of them.

Returns a list of chunk dicts:
    {
        "filename":    "report.pdf",
        "filetype":    "pdf",
        "chunk_index": 0,
        "text":        "chunk text here..."
    }
"""

from __future__ import annotations


# ── DEFAULTS ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE    = 500   # characters per chunk
DEFAULT_CHUNK_OVERLAP = 100   # characters of overlap between consecutive chunks


# ── CORE SPLITTER ─────────────────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split a plain string into overlapping chunks by character count.

    Strategy:
        - Slide a window of `chunk_size` characters across the text.
        - Each next chunk starts `chunk_size - chunk_overlap` characters
          after the previous one.
        - Strips whitespace from each chunk and drops empty ones.

    Args:
        text:          The raw text to split.
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of non-empty text strings.

    Raises:
        ValueError: if chunk_overlap >= chunk_size (would cause infinite loop).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    if not text or not text.strip():
        return []

    chunks = []
    step   = chunk_size - chunk_overlap
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# ── DOCUMENT SPLITTER ─────────────────────────────────────────────────────────

def split_document(
    document: dict,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split a single loaded document (from loader.py) into chunks.

    Args:
        document:      Dict with keys: filename, filetype, content.
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of chunk dicts:
        {
            "filename":    str,
            "filetype":    str,
            "chunk_index": int,   # 0-based position within this document
            "text":        str
        }
    """
    filename = document.get("filename", "unknown")
    filetype = document.get("filetype", "unknown")
    content  = document.get("content", "")

    raw_chunks = split_text(content, chunk_size, chunk_overlap)

    chunks = [
        {
            "filename":    filename,
            "filetype":    filetype,
            "chunk_index": i,
            "text":        chunk,
        }
        for i, chunk in enumerate(raw_chunks)
    ]

    print(
        f"[Splitter] '{filename}' → {len(chunks)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks


def split_all_documents(
    documents: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split every document in a list into chunks.

    Args:
        documents:     List of document dicts from loader.load_all_documents().
        chunk_size:    Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        Flat list of all chunks across all documents.
        Each chunk carries its source filename for citation purposes.
    """
    if not documents:
        print("[Splitter] No documents to split. Returning empty.")
        return []

    all_chunks = []
    for doc in documents:
        chunks = split_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    print(f"[Splitter] Done. {len(all_chunks)} total chunk(s) from {len(documents)} document(s).")
    return all_chunks