"""Index a CV PDF or JSON and job descriptions for local chat."""

from pathlib import Path

from assistant.cv_generator import load_cv
from assistant.cv_parser import DATA_DIR, chunk_text, read_document


def read_cv(path: Path) -> str:
    """Read a PDF CV or serialize a structured CV as searchable text."""
    if path.suffix.lower() == ".json":
        return load_cv(path).model_dump_json(indent=2)
    if path.suffix.lower() == ".pdf":
        return read_document(path)
    raise ValueError(f"CV must be a PDF or JSON file: {path}")


def index_documents(cv: Path, jobs: list[Path] | None = None) -> None:
    """Replace the chat index with a CV and explicitly selected or default jobs."""
    jobs = (
        jobs
        if jobs is not None
        else sorted(
            path
            for path in (DATA_DIR / "job_descriptions").glob("*")
            if path.suffix.lower() in {".txt", ".pdf"} and path.is_file()
        )
    )
    # Validate all inputs before replacing the existing index.
    documents = [(cv.name, chunk_text(read_cv(cv)))]
    documents += [(job.name, chunk_text(read_document(job))) for job in jobs]

    from assistant.vector_store import clear_db, ingest

    print("Replacing the chat index with the supplied CV and job descriptions.")
    clear_db()
    for index, (name, chunks) in enumerate(documents):
        source = f"cv:{name}" if index == 0 else f"jd:{index}:{name}"
        ingest(chunks, source=source)
    print(f"Indexed {len(documents)} documents.")
