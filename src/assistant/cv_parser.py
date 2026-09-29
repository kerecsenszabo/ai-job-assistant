"""Parse and chunk a CV PDF into text segments."""

from pathlib import Path
import pdfplumber
from rich.console import Console

console = Console()

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CV_PATH = DATA_DIR / "cv.pdf"

CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 100  # overlap between consecutive chunks


def extract_text(pdf_path: Path = CV_PATH) -> str:
    """Extract full text from a PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"CV not found at {pdf_path}\n" "→ Copy your CV PDF to data/cv.pdf"
        )
    with pdfplumber.open(pdf_path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return "\n\n".join(pages)


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def load_cv_chunks(pdf_path: Path = CV_PATH) -> list[str]:
    """Extract and chunk the CV. Returns a list of text chunks."""
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    return chunks


if __name__ == "__main__":
    console.print("[bold cyan]Parsing CV...[/bold cyan]")
    chunks = load_cv_chunks()
    console.print(f"[green]✓ Extracted {len(chunks)} chunks from CV[/green]\n")
    for i, chunk in enumerate(chunks[:3], 1):
        console.rule(f"Chunk {i}")
        console.print(chunk)
