import chromadb
from pathlib import Path
from assistant.embeddings import embed

DB_PATH = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION_NAME = "career_docs"


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)


def ingest(chunks: list[str], source: str = "cv") -> None:
    """Embed and store chunks in ChromaDB."""
    collection = get_collection()
    vectors = embed(chunks)
    ids = [f"{source}-{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": source}] * len(chunks),
    )
    print(f"Ingested {len(chunks)} chunks from '{source}'")


def search(query: str, top_k: int = 5, source_filter: str | None = None) -> list[str]:
    """Semantic search — returns top_k relevant chunks."""
    collection = get_collection()
    query_vector = embed([query])[0]
    where = {"source": source_filter} if source_filter else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )
    return results["documents"][0]


def clear_db() -> None:
    """Clear all documents from the ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(DB_PATH))
    client.delete_collection(name=COLLECTION_NAME)
    print(f"Cleared ChromaDB collection '{COLLECTION_NAME}'")
