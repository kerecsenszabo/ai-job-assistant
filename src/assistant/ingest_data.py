#!/usr/bin/env python
"""
Ingest all data sources into ChromaDB vector database.

This script loads data from:
- data/cv.pdf → career experience and skills
- data/cover_letters/ → writing samples
- data/messages/ → LinkedIn message corpus
- data/job_descriptions/ → target job postings

Run with:
    uv run python ingest_data.py
"""

from pathlib import Path
from assistant.cv_parser import load_cv_chunks
from assistant.vector_store import ingest, clear_db


def ingest_cv() -> int:
    """Load and ingest CV from data/cv.pdf."""
    print("\n📄 Ingesting CV...")
    try:
        chunks = load_cv_chunks()
        ingest(chunks, source="cv")
        print(f"✓ CV: {len(chunks)} chunks ingested")
        return len(chunks)
    except FileNotFoundError:
        print("✗ CV not found at data/cv.pdf")
        return 0


def ingest_cover_letters() -> int:
    """Load and ingest cover letters from data/cover_letters/."""
    print("\n📝 Ingesting cover letters...")
    cover_letters_dir = Path("data/cover_letters")
    if not cover_letters_dir.exists():
        print("✗ No cover_letters directory found")
        return 0

    total_chunks = 0
    for letter_file in cover_letters_dir.glob("*.txt"):
        try:
            text = letter_file.read_text()
            # Simple chunking for text files (can be improved)
            chunks = [text[i : i + 500] for i in range(0, len(text), 400)]
            ingest(chunks, source=f"cover_letter:{letter_file.stem}")
            total_chunks += len(chunks)
            print(f"  ✓ {letter_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ {letter_file.name}: {e}")

    if total_chunks > 0:
        print(f"✓ Cover letters: {total_chunks} chunks ingested")
    return total_chunks


def ingest_messages() -> int:
    """Load and ingest LinkedIn messages from data/messages/."""
    print("\n💬 Ingesting messages...")
    messages_dir = Path("data/messages")
    if not messages_dir.exists():
        print("✗ No messages directory found")
        return 0

    total_chunks = 0
    for message_file in messages_dir.glob("*.txt"):
        try:
            text = message_file.read_text()
            # Split by newline for message files
            chunks = [m.strip() for m in text.split("\n") if m.strip()]
            ingest(chunks, source=f"message:{message_file.stem}")
            total_chunks += len(chunks)
            print(f"  ✓ {message_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ {message_file.name}: {e}")

    if total_chunks > 0:
        print(f"✓ Messages: {total_chunks} chunks ingested")
    return total_chunks


def ingest_job_descriptions() -> int:
    """Load and ingest job descriptions from data/job_descriptions/."""
    print("\n🎯 Ingesting job descriptions...")
    jd_dir = Path("data/job_descriptions")
    if not jd_dir.exists():
        print("✗ No job_descriptions directory found")
        return 0

    total_chunks = 0
    for jd_file in jd_dir.glob("*.txt"):
        try:
            text = jd_file.read_text()
            chunks = [text[i : i + 500] for i in range(0, len(text), 400)]
            ingest(chunks, source=f"jd:{jd_file.stem}")
            total_chunks += len(chunks)
            print(f"  ✓ {jd_file.name}: {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ {jd_file.name}: {e}")

    if total_chunks > 0:
        print(f"✓ Job descriptions: {total_chunks} chunks ingested")
    return total_chunks


def main():
    """Ingest all available data sources."""
    print("=" * 60)
    print("AI Job Assistant — Data Ingestion Pipeline")
    print("=" * 60)

    # Clear existing data
    print("\n🗑️  Clearing ChromaDB...")
    clear_db()

    total = 0
    total += ingest_cv()
    total += ingest_cover_letters()
    total += ingest_messages()
    total += ingest_job_descriptions()

    print("\n" + "=" * 60)
    print(f"✓ Done! Total: {total} chunks ingested into ChromaDB")
    print("=" * 60)


if __name__ == "__main__":
    main()
