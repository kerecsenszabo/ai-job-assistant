"""Persist validated job analyses independently of candidate CVs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from assistant.cv_tailoring import ParsedJob

CACHE_VERSION = 5


class _CacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cache_version: Literal[5]
    schema_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    description_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    parsed_job: dict[str, object]


def atomic_write_json(path: Path, value: dict[str, object]) -> None:
    """Replace a JSON file atomically, cleaning only our own staging file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.stem}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=False, indent=2)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def get_parsed_job(
    description: str,
    llm: Runnable,
    *,
    cache_dir: Path | None,
    refresh: bool = False,
) -> tuple[ParsedJob, str, bool]:
    """Reuse a job-only parse, or explicitly refresh it with the supplied model."""
    from assistant.cv_tailoring import ParsedJob, parse_job

    description = description.strip()
    fingerprint = hashlib.sha256(description.encode("utf-8")).hexdigest()
    if cache_dir is None:
        return parse_job(llm, description), fingerprint, False

    schema_hash = hashlib.sha256(
        json.dumps(
            {"version": CACHE_VERSION, "schema": ParsedJob.model_json_schema()},
            sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{fingerprint}.json"
    if not refresh:
        try:
            cached = cache_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            cached = None
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Invalid job analysis cache {cache_path}; "
                "rerun with --refresh-job-analysis to replace it."
            ) from exc
        if cached is not None:
            try:
                record = _CacheRecord.model_validate_json(cached)
                if record.description_hash != fingerprint:
                    raise ValueError("Job description hash does not match.")
                if record.schema_hash != schema_hash:
                    raise ValueError("Parsed-job schema has changed.")
                parsed = ParsedJob.model_validate(record.parsed_job, strict=True)
            except (ValidationError, ValueError) as exc:
                raise ValueError(
                    f"Invalid or stale job analysis cache {cache_path}; "
                    "rerun with --refresh-job-analysis to replace it."
                ) from exc
            return parsed, fingerprint, True

    parsed = parse_job(llm, description)
    record = _CacheRecord(
        cache_version=CACHE_VERSION,
        schema_hash=schema_hash,
        description_hash=fingerprint,
        parsed_job=parsed.model_dump(mode="json"),
    )
    atomic_write_json(cache_path, record.model_dump(mode="json"))
    return parsed, fingerprint, False
