"""Validated matching cache; keep its personal explanations in private output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

from assistant.job_requirements import atomic_write_json

if TYPE_CHECKING:
    from assistant.cv_generator import CV
    from assistant.cv_tailoring import (
        EvidenceItem, ParsedJob, RequirementMatch, ScoringRubric,
    )

CACHE_VERSION = 1
ANALYSIS_VERSION = "matching-v2"


class _CacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cache_version: Literal[1]
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    matches: dict[str, object]


def get_cached_matches(
    cv: CV,
    description: str,
    job: ParsedJob,
    evidence: list[EvidenceItem],
    llm: Runnable,
    *,
    cache_dir: Path | None,
    model_identity: str | None,
    rubric: ScoringRubric,
    refresh: bool = False,
) -> tuple[list[RequirementMatch], str, bool]:
    """Return complete matches, their fingerprint, and whether disk was reused.

    Model identity must include actual model weights and inference settings.
    Cache directories must be private and gitignored: explanations are personal.
    """
    from assistant.cv_generator import CV
    from assistant.cv_tailoring import (
        EvidenceItem, Matches, ParsedJob, ScoringRubric,
        checked_matches, enforce_match_rules, match_job,
    )

    if cache_dir is not None and (
        model_identity is None or not model_identity.strip()
    ):
        raise ValueError(
            "Matching cache requires model_identity containing the model name, "
            "actual model digest, and inference settings; "
            "supply it or disable caching with cache_dir=None."
        )
    payload = {
        "cache_version": CACHE_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "cv": cv.model_dump(mode="json"),
        "description": description.strip(),
        "job": job.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "model_identity": model_identity,
        "rubric": rubric.model_dump(mode="json"),
        "schemas": {
            model.__name__: model.model_json_schema()
            for model in (CV, ParsedJob, EvidenceItem, Matches, ScoringRubric)
        },
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    cache_path = cache_dir / f"{fingerprint}.json" if cache_dir is not None else None
    if cache_path is not None and not refresh:
        try:
            cached = cache_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            cached = None
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Invalid matching cache {cache_path}; "
                "rerun with --refresh-matching to replace it."
            ) from exc
        if cached is not None:
            try:
                record = _CacheRecord.model_validate_json(cached)
                if record.fingerprint != fingerprint:
                    raise ValueError("Matching fingerprint does not match.")
                stored = Matches.model_validate(record.matches, strict=True)
                validated = checked_matches(stored, job.requirements, evidence)
                corrected = enforce_match_rules(job.requirements, validated, evidence)
                # Rule-adjustment history is not idempotent; decisions must be.
                for original, current in zip(validated, corrected):
                    if original.model_dump(exclude={"rule_adjustments"}) != current.model_dump(
                        exclude={"rule_adjustments"}
                    ):
                        raise ValueError("Cached decisions violate deterministic match rules.")
                if validated != stored.matches:
                    raise ValueError("Cached matches require structural normalization.")
            except ValueError as exc:
                raise ValueError(
                    f"Invalid matching cache {cache_path}; "
                    "rerun with --refresh-matching to replace it."
                ) from exc
            return validated, fingerprint, True

    proposed = match_job(llm, job, evidence) if job.requirements else []
    validated = checked_matches(
        Matches.model_validate({"matches": proposed}, strict=True),
        job.requirements, evidence,
    )
    completed = enforce_match_rules(job.requirements, validated, evidence)
    completed = checked_matches(
        Matches(matches=completed), job.requirements, evidence,
    )
    if cache_path is not None:
        record = _CacheRecord(
            cache_version=CACHE_VERSION, fingerprint=fingerprint,
            matches=Matches(matches=completed).model_dump(mode="json"),
        )
        atomic_write_json(cache_path, record.model_dump(mode="json"))
    return completed, fingerprint, False
