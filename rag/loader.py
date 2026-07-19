"""
loader.py
---------
Responsibility: Read raw text content from documents in the knowledge/ folder.
Supports: PDF, DOCX, TXT, MD

Returns a list of dicts, one per document:
    {
        "filename": "report.pdf",
        "filetype": "pdf",
        "content":  "full extracted text..."
    }
"""

import os
from pathlib import Path


# ── PDF ──────────────────────────────────────────────────────────────────────

def _load_pdf(filepath: Path) -> str:
    """Extract text from all pages of a PDF file."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required for PDF support. Run: pip install pypdf")

    reader = PdfReader(str(filepath))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


# ── DOCX ─────────────────────────────────────────────────────────────────────

def _load_docx(filepath: Path) -> str:
    """Extract text from a Word document paragraph by paragraph."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Run: pip install python-docx")

    doc = Document(str(filepath))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# ── TXT / MD ─────────────────────────────────────────────────────────────────

def _load_text(filepath: Path) -> str:
    """Read plain text or markdown files as-is."""
    return filepath.read_text(encoding="utf-8", errors="ignore").strip()


# ── DISPATCHER ───────────────────────────────────────────────────────────────

_LOADERS = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".txt":  _load_text,
    ".md":   _load_text,
}

SUPPORTED_EXTENSIONS = set(_LOADERS.keys())


def load_document(filepath: str | Path) -> dict:
    """
    Load a single document and return its content as a dict.

    Args:
        filepath: Path to the document file.

    Returns:
        {
            "filename": str,
            "filetype": str,   # "pdf" | "docx" | "txt" | "md"
            "content":  str    # full extracted text
        }

    Raises:
        ValueError: if the file extension is not supported.
        FileNotFoundError: if the file does not exist.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Document not found: {filepath}")

    ext = filepath.suffix.lower()
    if ext not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    content = _LOADERS[ext](filepath)

    return {
        "filename": filepath.name,
        "filetype": ext.lstrip("."),
        "content":  content,
    }


def load_all_documents(knowledge_dir: str | Path = "knowledge") -> list[dict]:
    """
    Load every supported document from the knowledge/ folder.

    Args:
        knowledge_dir: Path to the folder containing knowledge documents.
                       Defaults to 'knowledge' (relative to project root).

    Returns:
        List of document dicts (same structure as load_document).
        Skips unsupported files silently.
        Returns empty list if folder is empty or does not exist.
    """
    knowledge_dir = Path(knowledge_dir)

    if not knowledge_dir.exists():
        print(f"[Loader] knowledge/ folder not found at '{knowledge_dir}'. Returning empty.")
        return []

    documents = []
    skipped  = []

    for filepath in sorted(knowledge_dir.iterdir()):
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            skipped.append(filepath.name)
            continue

        try:
            doc = load_document(filepath)
            documents.append(doc)
            print(f"[Loader] ✅ Loaded '{doc['filename']}' ({doc['filetype'].upper()}) "
                  f"— {len(doc['content'])} chars")
        except Exception as e:
            print(f"[Loader] ❌ Failed to load '{filepath.name}': {e}")

    if skipped:
        print(f"[Loader] ⚠️  Skipped unsupported files: {', '.join(skipped)}")

    print(f"[Loader] Done. {len(documents)} document(s) loaded.")
    return documents