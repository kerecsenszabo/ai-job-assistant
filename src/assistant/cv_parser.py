"""Parse and chunk a CV PDF into text segments."""

from pathlib import Path

import pdfplumber

DATA_DIR = Path(__file__).parent.parent.parent / "data"

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks


def extract_text(pdf_path: Path) -> str:
    """Extract full text from a PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"CV not found at {pdf_path}\n" "→ Copy your CV PDF to data/cv.pdf"
        )
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n\n".join(pages)
    if not text.strip():
        raise ValueError(
            f"No readable text in {pdf_path}. Use a text-based PDF; "
            "scanned PDFs need OCR before import."
        )
    return text


def read_document(path: Path) -> str:
    """Read a text-based PDF or UTF-8 text document."""
    if path.suffix.lower() == ".pdf":
        return extract_text(path)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a PDF or TXT document: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Document is empty: {path}")
    return text


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError(
            "Chunk size must be positive and overlap smaller than chunk size."
        )
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]
