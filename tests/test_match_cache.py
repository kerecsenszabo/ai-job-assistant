import json
from unittest.mock import Mock

import pytest

from assistant import cv_tailoring, job_requirements, match_cache
from assistant.cv_generator import CV
from assistant.cv_tailoring import (
    CriterionMatch, EvidenceItem, ParsedJob, Requirement, RequirementCriterion,
    RequirementMatch, ScoringRubric,
)
from assistant.match_cache import get_cached_matches


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    cv = CV(name="Private Candidate", email="private@example.com", skills=["Python"])
    evidence = [EvidenceItem(id="skills/0", section="skills", text="Python")]
    job = ParsedJob(requirements=[
        Requirement(
            id="requirement/0", text="Python", quote="Python", importance="required",
            criteria=[RequirementCriterion(
                id="criterion/0", text="Python", quote="Python",
                kind="technology", options=["Python"],
            )],
        )
    ])
    matcher = Mock(return_value=[
        RequirementMatch(
            requirement_id="requirement/0", status="direct",
            evidence_ids=["skills/0"], explanation="Supported by the cited skill.",
            criteria_matches=[CriterionMatch(
                criterion_id="criterion/0", status="direct",
                evidence_ids=["skills/0"], explanation="Python listed in source.",
            )],
        )
    ])
    monkeypatch.setattr(cv_tailoring, "match_job", matcher)
    return {
        "cv": cv, "description": "Python", "job": job, "evidence": evidence,
        "llm": object(), "cache_dir": tmp_path / "matching",
        "model_identity": "fake-v1", "rubric": ScoringRubric(),
    }, matcher


def test_miss_then_hit_preserves_matches_without_calls(inputs):
    args, matcher = inputs
    first, fingerprint, hit = get_cached_matches(**args)
    assert not hit
    args["description"] = " \nPython  "
    second, second_fingerprint, hit = get_cached_matches(**args)
    assert hit
    assert first == second
    assert fingerprint == second_fingerprint
    matcher.assert_called_once_with(args["llm"], args["job"], args["evidence"])
    assert not list(args["cache_dir"].glob("*.tmp"))


@pytest.mark.parametrize("change", [
    "cv", "description", "job", "criterion", "evidence", "model", "rubric", "analysis", "schema",
])
def test_full_inputs_invalidate_cache(inputs, monkeypatch, change):
    args, matcher = inputs
    _, original, _ = get_cached_matches(**args)
    if change == "cv":
        args["cv"] = args["cv"].model_copy(update={"email": "changed@example.com"})
    elif change == "description":
        args["description"] = "Python preferred"
    elif change == "job":
        args["job"] = args["job"].model_copy(deep=True)
        args["job"].requirements[0].importance = "preferred"
    elif change == "criterion":
        args["job"] = args["job"].model_copy(deep=True)
        args["job"].requirements[0].criteria = [
            RequirementCriterion(
                id="criterion/0", text="Python", quote="Python",
                kind="technology", options=["Python"], experience_required=True,
            )
        ]
    elif change == "evidence":
        args["evidence"] = [args["evidence"][0].model_copy(update={"context": "Changed"})]
    elif change == "model":
        args["model_identity"] = "fake-v2"
    elif change == "rubric":
        args["rubric"] = ScoringRubric(partial_credit=0.25)
    elif change == "analysis":
        monkeypatch.setattr(match_cache, "ANALYSIS_VERSION", "matching-v3")
    elif change == "schema":
        original_schema = cv_tailoring.Matches.model_json_schema()
        monkeypatch.setattr(cv_tailoring.Matches, "model_json_schema", lambda: {**original_schema, "title": "Changed"})
    _, fingerprint, hit = get_cached_matches(**args)
    assert fingerprint != original
    assert not hit
    assert matcher.call_count == 2


def test_refresh_replaces_corrupt_record(inputs):
    args, matcher = inputs
    _, fingerprint, _ = get_cached_matches(**args)
    path = args["cache_dir"] / f"{fingerprint}.json"
    path.write_text("corrupt")
    _, refreshed, hit = get_cached_matches(**args, refresh=True)
    assert refreshed == fingerprint
    assert not hit
    assert matcher.call_count == 2
    assert json.loads(path.read_text())["fingerprint"] == fingerprint


@pytest.mark.parametrize("damage", [
    "version", "fingerprint", "malformed", "unicode", "extra",
    "requirement_id", "evidence_id", "missing", "duplicate", "status",
])
def test_corrupt_record_requires_refresh_without_model_call(inputs, damage):
    args, matcher = inputs
    _, fingerprint, _ = get_cached_matches(**args)
    path = args["cache_dir"] / f"{fingerprint}.json"
    record = json.loads(path.read_text())
    if damage == "version":
        record["cache_version"] = 2
    elif damage == "fingerprint":
        record["fingerprint"] = "0" * 64
    elif damage == "extra":
        record["cv"] = {}
    elif damage == "requirement_id":
        record["matches"]["matches"][0]["requirement_id"] = "nonexistent"
    elif damage == "evidence_id":
        record["matches"]["matches"][0]["evidence_ids"] = ["nonexistent"]
    elif damage == "missing":
        record["matches"]["matches"] = []
    elif damage == "duplicate":
        record["matches"]["matches"] *= 2
    elif damage == "status":
        record["matches"]["matches"][0]["status"] = "invalid"
    if damage == "malformed":
        path.write_text("{invalid")
    elif damage == "unicode":
        path.write_bytes(b"\xff")
    else:
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="--refresh-matching"):
        get_cached_matches(**args)
    assert matcher.call_count == 1


@pytest.mark.parametrize("identity", [None, "", " \n "])
def test_missing_identity_requires_explicit_identity(inputs, identity):
    args, matcher = inputs
    args["model_identity"] = identity
    with pytest.raises(ValueError, match="model_identity.*digest"):
        get_cached_matches(**args)
    matcher.assert_not_called()
    assert not args["cache_dir"].exists()


@pytest.mark.parametrize("refresh", [False, True])
def test_no_cache_needs_no_identity_and_always_computes(inputs, refresh):
    args, matcher = inputs
    directory = args["cache_dir"]
    args.update(cache_dir=None, model_identity=None)
    for _ in range(2):
        _, _, hit = get_cached_matches(**args, refresh=refresh)
        assert not hit
    assert matcher.call_count == 2
    assert not directory.exists()


@pytest.mark.parametrize("failure", [RuntimeError("service unavailable"), ValueError("bad output")])
def test_failures_are_not_swallowed_or_written(inputs, failure):
    args, matcher = inputs
    matcher.side_effect = failure
    with pytest.raises(type(failure), match=str(failure)):
        get_cached_matches(**args)
    assert not args["cache_dir"].exists()


def test_invalid_fresh_matches_are_not_written(inputs):
    args, matcher = inputs
    matcher.return_value[0].evidence_ids = ["nonexistent"]
    with pytest.raises(ValueError, match="nonexistent CV evidence"):
        get_cached_matches(**args)
    assert not args["cache_dir"].exists()


def test_empty_requirements_never_call_matcher(inputs):
    args, matcher = inputs
    args["job"] = ParsedJob(requirements=[])
    assert get_cached_matches(**args)[0] == []
    assert get_cached_matches(**args)[2]
    matcher.assert_not_called()


def test_requirements_without_criteria_are_rejected_before_cache_lookup(inputs):
    args, matcher = inputs
    args["job"].requirements[0].criteria = []
    with pytest.raises(ValueError, match="--refresh-job-analysis"):
        get_cached_matches(**args)
    matcher.assert_not_called()


def test_envelope_contains_no_full_source_documents(inputs):
    args, _ = inputs
    _, fingerprint, _ = get_cached_matches(**args)
    contents = (args["cache_dir"] / f"{fingerprint}.json").read_text()
    record = json.loads(contents)
    assert set(record) == {"cache_version", "fingerprint", "matches"}
    assert args["cv"].name not in contents
    assert args["cv"].email not in contents
    assert "description" not in record
    assert "evidence" not in record


@pytest.mark.parametrize("kind, option", [("technology", "Kubernetes"), ("language", "German")])
def test_cached_positive_missing_explicit_criterion_is_rejected(inputs, kind, option):
    args, matcher = inputs
    args["job"].requirements[0].criteria = [
        RequirementCriterion(id="criterion/0", text=option, quote=option, kind=kind, options=[option])
    ]
    matcher.return_value[0].criteria_matches = [
        CriterionMatch(criterion_id="criterion/0", status="direct", evidence_ids=["skills/0"], explanation="Incorrect positive")
    ]
    completed, fingerprint, _ = get_cached_matches(**args)
    assert completed[0].status == "not_evidenced"
    # A legitimate guarded result remains reusable despite adjustment history.
    assert get_cached_matches(**args)[2]
    path = args["cache_dir"] / f"{fingerprint}.json"
    record = json.loads(path.read_text())
    match = record["matches"]["matches"][0]
    match.update(status="direct", evidence_ids=["skills/0"])
    match["criteria_matches"][0].update(status="direct", evidence_ids=["skills/0"])
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="--refresh-matching"):
        get_cached_matches(**args)
    assert matcher.call_count == 1


def test_explicit_match_hit_preserves_original_rule_adjustments(inputs):
    args, matcher = inputs
    args["job"].requirements[0].criteria = [
        RequirementCriterion(
            id="criterion/0", text="Python", quote="Python",
            kind="technology", options=["Python"],
        )
    ]
    matcher.return_value[0] = matcher.return_value[0].model_copy(update={
        "status": "unclear",
        "evidence_ids": [],
        "criteria_matches": [
            CriterionMatch(
                criterion_id="criterion/0", status="direct",
                evidence_ids=["skills/0"], explanation="Python is explicit.",
            )
        ],
    })
    completed, fingerprint, hit = get_cached_matches(**args)
    assert not hit
    assert completed[0].status == "direct"
    assert completed[0].rule_adjustments
    reused, reused_fingerprint, hit = get_cached_matches(**args)
    assert hit
    assert reused_fingerprint == fingerprint
    assert reused == completed
    assert reused[0].rule_adjustments == completed[0].rule_adjustments
    matcher.assert_called_once()


def test_shared_atomic_writer_preserves_old_record_and_cleans_own_file(inputs, monkeypatch):
    args, matcher = inputs
    _, fingerprint, _ = get_cached_matches(**args)
    path = args["cache_dir"] / f"{fingerprint}.json"
    original = path.read_bytes()
    unrelated = args["cache_dir"] / ".unrelated.tmp"
    unrelated.write_text("keep")

    def fail_replace(source, destination):
        assert source.parent == path.parent
        assert destination == path
        raise OSError("replace failed")

    monkeypatch.setattr(job_requirements.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        get_cached_matches(**args, refresh=True)
    assert path.read_bytes() == original
    assert list(args["cache_dir"].glob("*.tmp")) == [unrelated]
    assert matcher.call_count == 2
