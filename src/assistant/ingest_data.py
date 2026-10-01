"""Index a CV PDF and job descriptions for local chat."""

from pathlib import Path

from assistant.cv_parser import DATA_DIR, chunk_text, read_document


def index_documents(cv: Path, jobs: list[Path] | None = None) -> None:
    """Replace the chat index with a CV and explicitly selected or default jobs."""
    if cv.suffix.lower() != ".pdf":
        raise ValueError("The chat source CV must be a PDF.")
    jobs = jobs if jobs is not None else sorted(
        path for path in (DATA_DIR / "job_descriptions").glob("*")
        if path.suffix.lower() in {".txt", ".pdf"} and path.is_file()
    )
    # Validate all inputs before replacing the existing index.
    documents = [(cv.name, chunk_text(read_document(cv)))]
    documents += [(job.name, chunk_text(read_document(job))) for job in jobs]

    from assistant.vector_store import clear_db, ingest

    print("Replacing the chat index with the supplied CV and job descriptions.")
    clear_db()
    for index, (name, chunks) in enumerate(documents):
        source = f"cv:{name}" if index == 0 else f"jd:{index}:{name}"
        ingest(chunks, source=source)
    print(f"Indexed {len(documents)} documents.")
