from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 80MB, fast, good quality

_model = None   # lazy-load singleton

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings into vectors."""
    model = get_model()
    return model.encode(texts, show_progress_bar=True).tolist()
