import hashlib
import json

import pytest

from assistant import cv_tailoring, job_requirements
from assistant.cv_tailoring import ParsedJob
from assistant.job_requirements import get_parsed_job


@pytest.fixture
def parser(monkeypatch):
    calls = []

    def parse(llm, description):
        calls.append((description, llm))
        return ParsedJob(
            requirements=[
                {
                    "id": "requirement/0",
                    "text": description,
                    "quote": description,
                    "importance": "required",
                    "criteria": [
                        {
                            "id": "requirement/0/criterion/0",
                            "text": description,
                            "quote": description,
                            "kind": "technology",
                            "options": ["Python"],
                            "operator": "all",
                            "production": True,
                            "experience_required": True,
                            "proficiency": "advanced",
                        }
                    ],
                }
            ]
        )

    monkeypatch.setattr(cv_tailoring, "parse_job", parse)
    return calls


def test_cache_hit_skips_parser_and_preserves_full_job(tmp_path, parser):
    llm = object()
    parsed, fingerprint, cached = get_parsed_job(
        "  Python required\n", llm, cache_dir=tmp_path
    )
    assert not cached
    assert fingerprint == hashlib.sha256(b"Python required").hexdigest()
    reused, reused_fingerprint, cached = get_parsed_job(
        "Python required", object(), cache_dir=tmp_path
    )
    assert cached
    assert reused == parsed
    assert reused_fingerprint == fingerprint
    assert parser == [("Python required", llm)]
    record = json.loads((tmp_path / f"{fingerprint}.json").read_text())
    assert set(record) == {
        "cache_version",
        "schema_hash",
        "description_hash",
        "parsed_job",
    }
    assert record["cache_version"] == job_requirements.CACHE_VERSION
    assert record["parsed_job"] == parsed.model_dump(mode="json")
    assert "evidence" not in record and "cv" not in record
    assert list(tmp_path.glob("*.tmp")) == []


def test_recovery_warnings_survive_job_cache(tmp_path, monkeypatch):
    warning = "Job analysis: discarded an unquoted option."
    monkeypatch.setattr(
        cv_tailoring,
        "parse_job",
        lambda llm, description: ParsedJob(requirements=[], warnings=[warning]),
    )
    parsed, _, cached = get_parsed_job("Python", object(), cache_dir=tmp_path)
    assert not cached
    reused, _, cached = get_parsed_job("Python", object(), cache_dir=tmp_path)
    assert cached
    assert parsed.warnings == reused.warnings == [warning]


def test_changed_description_gets_its_own_analysis(tmp_path, parser):
    first, first_hash, _ = get_parsed_job(
        "Python required", object(), cache_dir=tmp_path
    )
    second, second_hash, cached = get_parsed_job(
        "Python preferred", object(), cache_dir=tmp_path
    )
    assert not cached
    assert first_hash != second_hash
    assert first != second
    assert len(parser) == 2
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_fingerprint_preserves_internal_whitespace(tmp_path, parser):
    first = "Python required\nProduction experience"
    second = "Python required  Production experience"
    _, first_hash, _ = get_parsed_job(first, object(), cache_dir=tmp_path)
    _, second_hash, cached = get_parsed_job(second, object(), cache_dir=tmp_path)
    assert first_hash != second_hash
    assert not cached
    assert [call[0] for call in parser] == [first, second]


def test_refresh_reparses_and_overwrites_cache(tmp_path, parser):
    _, fingerprint, _ = get_parsed_job("Python required", object(), cache_dir=tmp_path)
    cache_path = tmp_path / f"{fingerprint}.json"
    cache_path.write_text("corrupt")
    parsed, refreshed_hash, cached = get_parsed_job(
        "Python required", object(), cache_dir=tmp_path, refresh=True
    )
    assert not cached
    assert refreshed_hash == fingerprint
    assert len(parser) == 2
    assert json.loads(cache_path.read_text())["parsed_job"] == parsed.model_dump(
        mode="json"
    )


@pytest.mark.parametrize("refresh", [False, True])
def test_no_cache_directory_always_parses_without_writes(parser, refresh):
    for _ in range(2):
        parsed, fingerprint, cached = get_parsed_job(
            " Python required ", object(), cache_dir=None, refresh=refresh
        )
        assert parsed.requirements[0].text == "Python required"
        assert fingerprint == hashlib.sha256(b"Python required").hexdigest()
        assert not cached
    assert len(parser) == 2


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "missing_version",
        "schema",
        "hash",
        "malformed",
        "unicode",
        "parsed_job",
        "extra_cv",
        "parsed_job_cv",
    ],
)
def test_invalid_cache_requires_explicit_refresh(tmp_path, parser, damage):
    _, fingerprint, _ = get_parsed_job("Python required", object(), cache_dir=tmp_path)
    cache_path = tmp_path / f"{fingerprint}.json"
    record = json.loads(cache_path.read_text())
    if damage == "version":
        record["cache_version"] += 1
    elif damage == "missing_version":
        del record["cache_version"]
    elif damage == "schema":
        record["schema_hash"] = "0" * 64
    elif damage == "hash":
        record["description_hash"] = "0" * 64
    elif damage == "parsed_job":
        record["parsed_job"]["requirements"][0]["importance"] = "invalid"
    elif damage == "extra_cv":
        record["cv"] = {"name": "Candidate"}
    elif damage == "parsed_job_cv":
        record["parsed_job"]["cv"] = {"name": "Candidate"}
    if damage == "malformed":
        cache_path.write_text("{invalid")
    elif damage == "unicode":
        cache_path.write_bytes(b"\xff")
    else:
        cache_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="--refresh-job-analysis"):
        get_parsed_job("Python required", object(), cache_dir=tmp_path)
    assert len(parser) == 1


def test_atomic_write_failure_preserves_old_cache_and_cleans_exact_file(
    tmp_path, parser, monkeypatch
):
    _, fingerprint, _ = get_parsed_job("Python required", object(), cache_dir=tmp_path)
    cache_path = tmp_path / f"{fingerprint}.json"
    original = cache_path.read_bytes()
    unrelated = tmp_path / ".unrelated.tmp"
    unrelated.write_text("leave this alone")

    def fail_replace(source, destination):
        assert source.parent == tmp_path
        assert destination == cache_path
        raise OSError("replace failed")

    monkeypatch.setattr(job_requirements.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        get_parsed_job("Python required", object(), cache_dir=tmp_path, refresh=True)
    assert cache_path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == [unrelated]


def test_parse_failure_is_not_cached(tmp_path, monkeypatch):
    def fail_parse(llm, description):
        raise cv_tailoring.ModelOutputError("invalid response")

    monkeypatch.setattr(cv_tailoring, "parse_job", fail_parse)
    cache_dir = tmp_path / "cache"
    with pytest.raises(cv_tailoring.ModelOutputError, match="invalid response"):
        get_parsed_job("Python required", object(), cache_dir=cache_dir)
    assert not cache_dir.exists()


def test_creates_missing_cache_directory(tmp_path, parser):
    cache_dir = tmp_path / "nested" / "job-requirements"
    _, fingerprint, cached = get_parsed_job(
        "Python required", object(), cache_dir=cache_dir
    )
    assert not cached
    assert (cache_dir / f"{fingerprint}.json").is_file()
