"""
pipeline.py
-----------
Responsibility: Wire the entire RAG Knowledge Layer into a single
callable service that any agent can use.

Two main functions:
    1. build_rag_pipeline()  — loads documents, splits, embeds, indexes.
       Called once when documents are uploaded or knowledge/ changes.

    2. query_rag_pipeline()  — takes a question, retrieves relevant chunks.
       Called by agents (Chat Agent, Insight Agent) at query time.

This is NOT an agent. It is a shared service.

Full flow:
    knowledge/ folder
        → loader.load_all_documents()
        → splitter.split_all_documents()
        → embedder.embed_chunks()
        → vectordb.build_index()
        → vectordb.save_index()

    Query time:
        user question
        → retriever.retrieve()
        → top-k relevant chunks with scores
"""

from __future__ import annotations

import faiss

from rag.loader    import load_all_documents
from rag.splitter  import split_all_documents
from rag.embedder  import embed_chunks
from rag.vectordb  import build_index, save_index, load_index, index_exists, clear_index
from rag.retriever import retrieve, DEFAULT_TOP_K


# ── MODULE-LEVEL STATE ────────────────────────────────────────────────────────

# Held in memory after first load — avoids hitting disk on every query
_index:    faiss.Index | None = None
_metadata: list[dict]  | None = None


# ── BUILD ─────────────────────────────────────────────────────────────────────

def build_rag_pipeline(
    knowledge_dir: str = "knowledge",
    chunk_size: int    = 500,
    chunk_overlap: int = 100,
    force_rebuild: bool = False,
) -> dict:
    """
    Build the RAG Knowledge Layer from documents in the knowledge/ folder.

    Steps:
        1. Check if index already exists (skip rebuild unless forced).
        2. Load all documents from knowledge/.
        3. Split into chunks.
        4. Generate embeddings.
        5. Build FAISS index.
        6. Save index and metadata to disk.
        7. Cache in module-level state for immediate querying.

    Args:
        knowledge_dir:  Path to the folder containing knowledge documents.
        chunk_size:     Characters per chunk (passed to splitter).
        chunk_overlap:  Overlap between chunks (passed to splitter).
        force_rebuild:  If True, rebuilds even if an index already exists.
                        Use this when documents are added or removed.

    Returns:
        Status dict:
        {
            "status":       "built" | "loaded" | "empty",
            "documents":    int,   # number of documents loaded
            "chunks":       int,   # number of chunks created
            "index_total":  int,   # number of vectors in the index
        }
    """
    global _index, _metadata

    # ── Reuse existing index if available and not forced ──────────────────────
    if index_exists() and not force_rebuild:
        print("[Pipeline] Index already exists. Loading from disk...")
        _index, _metadata = load_index()
        return {
            "status":      "loaded",
            "documents":   0,
            "chunks":      len(_metadata),
            "index_total": _index.ntotal,
        }

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    print("[Pipeline] Step 1/4 — Loading documents...")
    documents = load_all_documents(knowledge_dir)

    if not documents:
        print("[Pipeline] No documents found. RAG pipeline is inactive.")
        _index    = None
        _metadata = None
        return {
            "status":      "empty",
            "documents":   0,
            "chunks":      0,
            "index_total": 0,
        }

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    print("[Pipeline] Step 2/4 — Splitting into chunks...")
    chunks = split_all_documents(documents, chunk_size, chunk_overlap)

    # ── Step 3: Embed ─────────────────────────────────────────────────────────
    print("[Pipeline] Step 3/4 — Generating embeddings...")
    embedded_chunks = embed_chunks(chunks)

    # ── Step 4: Index ─────────────────────────────────────────────────────────
    print("[Pipeline] Step 4/4 — Building and saving FAISS index...")
    _index, _metadata = build_index(embedded_chunks)
    save_index(_index, _metadata)

    print(f"[Pipeline] ✅ RAG pipeline ready — {_index.ntotal} vectors indexed.")
    return {
        "status":      "built",
        "documents":   len(documents),
        "chunks":      len(chunks),
        "index_total": _index.ntotal,
    }


# ── QUERY ─────────────────────────────────────────────────────────────────────

def query_rag_pipeline(
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks for a user query.
    Called by agents at query time.

    Loads index from disk if not already in memory.

    Args:
        query: The user's question or search string.
        top_k: Number of top chunks to return.

    Returns:
        List of chunk dicts sorted by relevance (most relevant first).
        Each dict contains: filename, filetype, chunk_index, text, score.
        Returns empty list if no index exists or knowledge/ is empty.
    """
    global _index, _metadata

    # ── Load index into memory if not already cached ──────────────────────────
    if _index is None or _metadata is None:
        if not index_exists():
            print("[Pipeline] No index found. Run build_rag_pipeline() first.")
            return []
        _index, _metadata = load_index()

    return retrieve(query, _index, _metadata, top_k)


# ── REBUILD ───────────────────────────────────────────────────────────────────

def rebuild_rag_pipeline(knowledge_dir: str = "knowledge") -> dict:
    """
    Force a full rebuild of the RAG index.
    Use this when documents are added, removed, or updated in knowledge/.

    Args:
        knowledge_dir: Path to knowledge folder.

    Returns:
        Same status dict as build_rag_pipeline().
    """
    global _index, _metadata

    print("[Pipeline] Clearing existing index for rebuild...")
    clear_index()
    _index    = None
    _metadata = None

    return build_rag_pipeline(knowledge_dir, force_rebuild=True)


# ── STATUS ────────────────────────────────────────────────────────────────────

def get_pipeline_status() -> dict:
    """
    Return the current state of the RAG pipeline.
    Used by app.py to show pipeline status in the Streamlit UI.

    Returns:
        {
            "ready":       bool,  # True if index is loaded and queryable
            "index_total": int,   # number of vectors currently indexed
            "in_memory":   bool,  # True if index is cached in memory
        }
    """
    if _index is not None and _metadata is not None:
        return {
            "ready":       True,
            "index_total": _index.ntotal,
            "in_memory":   True,
        }

    if index_exists():
        return {
            "ready":       True,
            "index_total": -1,   # exists on disk but not loaded yet
            "in_memory":   False,
        }

    return {
        "ready":       False,
        "index_total": 0,
        "in_memory":   False,
    }