import json

import httpx
import pytest

from assistant import cv_generator


def test_cache_identity_includes_model_digest_and_inference_settings(monkeypatch):
    llm = cv_generator.local_llm("granite4.2:3b")
    requested = []

    def tags(url, *, timeout):
        requested.append((url, timeout))
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"models": [{"name": "granite4.2:3b", "digest": "weights-v1"}]},
        )

    monkeypatch.setattr(cv_generator.httpx, "get", tags)
    identity = json.loads(cv_generator.local_model_identity(llm))
    assert identity == {
        "model": "granite4.2:3b",
        "digest": "weights-v1",
        "temperature": 0,
        "num_ctx": cv_generator.CONTEXT_TOKENS,
        "reasoning": False,
    }
    assert requested == [("http://localhost:11434/api/tags", 5)]


def test_unidentifiable_model_does_not_reuse_cache(monkeypatch):
    monkeypatch.setattr(
        cv_generator.httpx,
        "get",
        lambda url, timeout: httpx.Response(
            200,
            request=httpx.Request("GET", url),
            json={"models": []},
        ),
    )
    with pytest.raises(ValueError, match="Cannot identify"):
        cv_generator.local_model_identity(cv_generator.local_llm("missing:3b"))


def test_model_metadata_service_errors_propagate(monkeypatch):
    def unavailable(url, *, timeout):
        raise httpx.ConnectError("Ollama unavailable")

    monkeypatch.setattr(cv_generator.httpx, "get", unavailable)
    with pytest.raises(httpx.ConnectError, match="unavailable"):
        cv_generator.local_model_identity(cv_generator.local_llm("granite4.2:3b"))
