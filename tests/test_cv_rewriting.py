import json

import pytest
from langchain_core.runnables import RunnableLambda

from assistant.cv_generator import CV, Experience, TailorDiagnostics
from assistant.cv_tailoring import TailoringReport, rewrite, source_evidence

IDS = [f"experience/0/bullets/{i}" for i in range(3)]
ORIGINALS = [
    "Built Python pipelines.",
    "Maintained SQL reports.",
    "Client A: Supported a prototype.",
]
CHANGED = [
    "Developed Python pipelines.",
    "Updated SQL reports.",
    "Client A: supported a prototype.",
]


@pytest.fixture
def setup():
    cv = CV(
        name="Example",
        email="example@example.com",
        summary="I build pipelines.",
        skills=["Python", "SQL"],
        experience=[
            Experience(
                company="Example Co", role="Engineer", dates="2024", bullets=ORIGINALS
            )
        ],
    )
    evidence = source_evidence(cv)
    report = TailoringReport(evidence=evidence)
    selected = [item for item in evidence if item.section == "experience"]
    return cv, report, selected


def proposal(texts=CHANGED):
    return {
        "bullets": {
            sid: {"source_ids": [sid], "text": text} for sid, text in zip(IDS, texts)
        },
        "summary_sentences": [],
    }


def verdicts(ids=IDS):
    return {
        "verdicts": {
            sid: {"status": "supported", "reason": "Supported by cited source."}
            for sid in ids
        }
    }


def run(setup, responses, description=None):
    remaining = iter(responses)
    calls = []

    def respond(prompt, **kwargs):
        calls.append((prompt, kwargs))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    llm = RunnableLambda(respond)
    cv, report, selected = setup
    diagnostics = TailorDiagnostics()
    result = rewrite(cv, llm, report, selected, description, diagnostics)
    return result, report, diagnostics, calls


def test_omitted_bullet_preserves_only_that_original(setup):
    draft = proposal()
    del draft["bullets"][IDS[1]]
    result, report, _, calls = run(setup, [draft, verdicts([IDS[0], IDS[2]])])
    assert result.experience[0].bullets == [CHANGED[0], ORIGINALS[1], CHANGED[2]]
    assert len(report.rewrites) == 3
    assert report.invalid_rewrite_response == json.dumps(draft)
    assert "Missing bullet" in next(
        a.reason for a in report.rewrites if a.target_id == IDS[1]
    )
    assert calls[1][1]["format"]["properties"]["verdicts"]["required"] == [
        IDS[0],
        IDS[2],
    ]


@pytest.mark.parametrize("citation", [[IDS[1]], [IDS[0], IDS[1]], ["missing"]])
def test_wrong_citation_rejects_only_affected_target(setup, citation):
    draft = proposal()
    draft["bullets"][IDS[0]]["source_ids"] = citation
    result, report, _, calls = run(setup, [draft, verdicts(IDS[1:])])
    assert result.experience[0].bullets == [ORIGINALS[0], *CHANGED[1:]]
    assert report.invalid_rewrite_response
    assert IDS[0] not in calls[1][1]["format"]["properties"]["verdicts"]["properties"]


def test_unkeyed_legacy_draft_cannot_export(setup):
    old_draft = {
        "sentences": [
            {"id": sid, "target_id": sid, "source_ids": [sid], "text": text}
            for sid, text in zip(IDS, CHANGED)
        ]
    }
    result, report, _, calls = run(setup, [old_draft])
    assert result.experience[0].bullets == ORIGINALS
    assert report.invalid_rewrite_response == json.dumps(old_draft)
    assert all(a.status == "unclear" for a in report.rewrites)
    assert len(calls) == 1


def test_duplicate_json_target_key_degrades_only_that_target(setup):
    draft = json.dumps(proposal())
    duplicated = draft.replace(
        '"bullets": {',
        '"bullets": {'
        + json.dumps(IDS[0])
        + ": "
        + json.dumps(proposal()["bullets"][IDS[0]])
        + ", ",
        1,
    )
    result, report, _, _ = run(setup, [duplicated, verdicts(IDS[1:])])
    assert result.experience[0].bullets == [ORIGINALS[0], *CHANGED[1:]]
    assert report.invalid_rewrite_response == duplicated


def test_missing_review_verdict_blocks_only_affected_changed_candidate(setup):
    result, report, _, _ = run(setup, [proposal(), verdicts([IDS[0], IDS[2]])])
    assert result.experience[0].bullets == [CHANGED[0], ORIGINALS[1], CHANGED[2]]
    assert report.invalid_review_response
    assert any("Missing review verdict" in a.reason for a in report.rewrites)


@pytest.mark.parametrize(
    "invalid",
    [
        {"source_ids": [IDS[0]], "text": ""},
        {"source_ids": [IDS[0]], "text": CHANGED[0], "target_id": IDS[1]},
        42,
    ],
)
def test_malformed_keyed_entry_does_not_invalidate_other_targets(setup, invalid):
    draft = proposal()
    draft["bullets"][IDS[0]] = invalid
    result, report, _, _ = run(setup, [draft, verdicts(IDS[1:])])
    assert result.experience[0].bullets == [ORIGINALS[0], *CHANGED[1:]]
    assert report.invalid_rewrite_response == json.dumps(draft)
    assert len(report.rewrites) == 3


def test_duplicate_json_review_key_degrades_only_that_candidate(setup):
    response = json.dumps(verdicts())
    response = response.replace(
        '"verdicts": {',
        '"verdicts": {'
        + json.dumps(IDS[0])
        + ": "
        + json.dumps(verdicts()["verdicts"][IDS[0]])
        + ", ",
        1,
    )
    result, report, _, _ = run(setup, [proposal(), response])
    assert result.experience[0].bullets == [ORIGINALS[0], *CHANGED[1:]]
    assert report.invalid_review_response == response


def test_legacy_list_review_cannot_export_changed_candidates(setup):
    review = {
        "verdicts": [
            {"id": sid, "status": "supported", "reason": "Supported."} for sid in IDS
        ]
    }
    review["verdicts"].append(dict(review["verdicts"][0]))
    result, report, _, _ = run(setup, [proposal(), review])
    assert result.experience[0].bullets == ORIGINALS
    assert report.invalid_review_response


def test_unchanged_bullets_need_no_review(setup):
    result, report, _, calls = run(setup, [proposal(ORIGINALS)])
    assert result.experience[0].bullets == ORIGINALS
    assert len(calls) == 1
    assert all(
        a.status == "accepted" and a.reason == "Unchanged source."
        for a in report.rewrites
    )


def test_mechanical_rejections_happen_before_review_and_can_skip_it(setup):
    draft = proposal(
        ["Built 999 Python pipelines.", ORIGINALS[1], "Supported a prototype."]
    )
    result, report, _, calls = run(setup, [draft])
    assert result.experience[0].bullets == ORIGINALS
    assert len(calls) == 1
    assert {a.status for a in report.rewrites} == {"accepted", "rejected"}
    assert any("context prefix" in a.reason for a in report.rewrites)
    assert report.invalid_rewrite_response == json.dumps(draft)


def test_only_changed_mechanically_valid_entries_are_reviewed(setup):
    draft = proposal([CHANGED[0], ORIGINALS[1], "Supported a prototype."])
    result, _, _, calls = run(setup, [draft, verdicts([IDS[0]])])
    assert result.experience[0].bullets == [CHANGED[0], *ORIGINALS[1:]]
    assert calls[1][1]["format"]["properties"]["verdicts"]["required"] == [IDS[0]]


def test_dynamic_schemas_require_exact_targets_and_citations(setup):
    _, _, _, calls = run(setup, [proposal(), verdicts()], description="Python role")
    schema = calls[0][1]["format"]
    assert schema["required"] == ["bullets", "summary_sentences"]
    bullets = schema["properties"]["bullets"]
    assert bullets["required"] == IDS
    assert bullets["additionalProperties"] is False
    assert set(bullets["properties"]) == set(IDS)
    for sid, value in bullets["properties"].items():
        assert value["required"] == ["source_ids", "text"]
        sources = value["properties"]["source_ids"]
        assert sources["maxItems"] == 1
        assert sources["items"]["enum"] == [sid]
    summaries = schema["properties"]["summary_sentences"]
    assert summaries["maxItems"] == 4
    assert set(summaries["items"]["properties"]["source_ids"]["items"]["enum"]) == {
        *IDS,
        "summary/0",
    }
    review = calls[1][1]["format"]["properties"]["verdicts"]
    assert review["required"] == IDS
    assert review["additionalProperties"] is False


@pytest.mark.parametrize("description", [None, "Python role"])
def test_prompt_requests_actual_experience_rewriting_in_both_modes(setup, description):
    result, _, _, calls = run(setup, [proposal(), verdicts()], description=description)
    instruction = calls[0][0].to_messages()[0].content
    assert "Rewrite every selected work-experience bullet" in instruction
    assert "Use direct action verbs" in instruction
    assert "preserving every factual detail" in instruction
    assert "never invent impact or metrics" in instruction
    if description is None:
        assert "Use general-purpose wording" in instruction
    else:
        assert "Emphasize facts relevant to the job" in instruction
    assert result.experience[0].bullets == CHANGED


@pytest.mark.parametrize("failure", ["unsupported", "missing", "bad_source"])
def test_any_summary_sentence_failure_retains_whole_source_summary(setup, failure):
    draft = proposal()
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I develop Python pipelines.",
        },
        {"id": "summary/1", "source_ids": [IDS[1]], "text": "I maintain SQL reports."},
    ]
    review = verdicts([*IDS, "summary/0", "summary/1"])
    if failure == "unsupported":
        review["verdicts"]["summary/1"]["status"] = "unsupported"
    elif failure == "missing":
        del review["verdicts"]["summary/1"]
    else:
        draft["summary_sentences"][1]["source_ids"] = ["unknown"]
        del review["verdicts"]["summary/1"]
    result, report, diagnostics, _ = run(
        setup, [draft, review], description="Python role"
    )
    assert result.summary == setup[0].summary
    assert result.experience[0].bullets == CHANGED
    summary_audits = [a for a in report.rewrites if a.target_id == "summary"]
    assert len(summary_audits) == 2
    assert all(
        a.exported == result.summary and a.status != "accepted" for a in summary_audits
    )
    assert diagnostics.summary_fallback


def test_fully_supported_summary_is_exported(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I develop Python pipelines.",
        }
    ]
    result, _, diagnostics, calls = run(
        setup, [draft, verdicts(["summary/0"])], description="Python role"
    )
    assert result.summary == "I develop Python pipelines."
    assert not diagnostics.summary_fallback
    assert calls[1][1]["format"]["properties"]["verdicts"]["required"] == ["summary/0"]


@pytest.mark.parametrize(
    "text",
    [
        "The candidate develops Python pipelines.",
        "I build pipelines. The candidate also develops Python pipelines.",
    ],
)
def test_supported_summary_with_changed_voice_keeps_source(setup, text):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {"id": "summary/0", "source_ids": [IDS[0]], "text": text}
    ]
    result, report, diagnostics, _ = run(
        setup, [draft, verdicts(["summary/0"])], description="Python role"
    )
    assert result.summary == setup[0].summary
    assert diagnostics.summary_fallback
    audit = next(a for a in report.rewrites if a.target_id == "summary")
    assert audit.status == "rejected"
    assert audit.exported == setup[0].summary
    assert "first-person voice" in audit.reason


def test_summary_sentences_can_share_first_person_voice(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {"id": "summary/0", "source_ids": [IDS[0]], "text": "I build pipelines."},
        {"id": "summary/1", "source_ids": [IDS[1]], "text": "SQL reports are my focus."},
    ]
    result, _, diagnostics, _ = run(
        setup, [draft, verdicts(["summary/0", "summary/1"])],
        description="Python role",
    )
    assert result.summary == "I build pipelines. SQL reports are my focus."
    assert not diagnostics.summary_fallback


def test_mechanically_rejected_summary_gets_focused_retry(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I have 9+ years of Python experience.",
        },
    ]
    corrected = {
        "summary_sentences": [
            {
                "id": "summary/0",
                "source_ids": [IDS[0]],
                "text": "I build Python pipelines.",
            },
        ]
    }
    result, report, diagnostics, calls = run(
        setup, [draft, corrected, verdicts(["summary/0"])], description="Python role"
    )
    assert result.summary == "I build Python pipelines."
    assert not diagnostics.summary_fallback
    assert [
        audit.status for audit in report.rewrites if audit.target_id == "summary"
    ] == [
        "rejected",
        "accepted",
    ]
    assert calls[1][1]["format"]["properties"]["summary_sentences"]["maxItems"] == 1
    assert (
        calls[1][1]["format"]["properties"]["summary_sentences"]["items"]["properties"][
            "text"
        ]["pattern"]
        == r"^[^0-9]*\S[^0-9]*$"
    )
    assert calls[2][1]["format"]["properties"]["verdicts"]["required"] == ["summary/0"]


def test_supported_third_person_retry_keeps_source_summary(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I have 9+ years of experience.",
        },
    ]
    retry = {
        "summary_sentences": [
            {
                "id": "summary/0",
                "source_ids": [IDS[0]],
                "text": "The candidate builds Python pipelines.",
            },
        ]
    }
    result, report, diagnostics, calls = run(
        setup, [draft, retry], description="Python role"
    )
    assert result.summary == setup[0].summary
    assert diagnostics.summary_fallback
    assert len(calls) == 2
    assert [
        audit.status for audit in report.rewrites if audit.target_id == "summary"
    ] == ["rejected", "rejected"]
    assert "first-person voice" in report.rewrites[-1].reason
    assert report.rewrites[-1].exported == setup[0].summary


def test_retry_rejection_keeps_original_summary_and_bullets(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I have 9+ years of experience.",
        },
    ]
    retry = {
        "summary_sentences": [
            {
                "id": "summary/0",
                "source_ids": [IDS[0]],
                "text": "I build 99 Python pipelines.",
            },
        ]
    }
    result, report, diagnostics, calls = run(
        setup, [draft, retry], description="Python role"
    )
    assert result.summary == setup[0].summary
    assert result.experience[0].bullets == ORIGINALS
    assert diagnostics.summary_fallback
    assert len(calls) == 2
    assert all(
        audit.status == "rejected" and audit.exported == setup[0].summary
        for audit in report.rewrites
        if audit.target_id == "summary"
    )


def test_retry_unchanged_source_needs_no_review(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I have 9+ years of experience.",
        },
    ]
    retry = {
        "summary_sentences": [
            {"id": "summary/0", "source_ids": ["summary/0"], "text": setup[0].summary},
        ]
    }
    result, _, diagnostics, calls = run(
        setup, [draft, retry], description="Python role"
    )
    assert result.summary == setup[0].summary
    assert not diagnostics.summary_fallback
    assert len(calls) == 2


def test_retry_review_rejection_still_falls_back(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I have 9+ years of experience.",
        },
    ]
    retry = {
        "summary_sentences": [
            {
                "id": "summary/0",
                "source_ids": [IDS[0]],
                "text": "I develop Python pipelines.",
            },
        ]
    }
    review = verdicts(["summary/0"])
    review["verdicts"]["summary/0"]["status"] = "unsupported"
    result, report, diagnostics, _ = run(
        setup, [draft, retry, review], description="Python role"
    )
    assert result.summary == setup[0].summary
    assert diagnostics.summary_fallback
    assert [
        audit.status for audit in report.rewrites if audit.target_id == "summary"
    ] == [
        "rejected",
        "rejected",
    ]


def test_retry_supersedes_initially_accepted_summary_sentence(setup):
    draft = proposal(ORIGINALS)
    draft["summary_sentences"] = [
        {
            "id": "summary/0",
            "source_ids": [IDS[0]],
            "text": "I develop Python pipelines.",
        },
        {
            "id": "summary/1",
            "source_ids": [IDS[1]],
            "text": "I have 9+ years of SQL experience.",
        },
    ]
    retry = {
        "summary_sentences": [
            {
                "id": "summary/0",
                "source_ids": [IDS[0]],
                "text": "I build Python pipelines.",
            },
        ]
    }
    result, report, diagnostics, _ = run(
        setup,
        [draft, verdicts(["summary/0"]), retry, verdicts(["summary/0"])],
        description="Python role",
    )
    assert result.summary == "I build Python pipelines."
    assert not diagnostics.summary_fallback
    audits = [audit for audit in report.rewrites if audit.target_id == "summary"]
    assert [audit.status for audit in audits] == ["rejected", "unclear", "accepted"]
    assert audits[1].exported == setup[0].summary


@pytest.mark.parametrize("response", ["not JSON", "[]", '{"bullets": null}', "{}"])
def test_unusable_draft_retains_all_with_visible_warning(setup, response):
    result, report, _, calls = run(setup, [response], description="Python role")
    assert result.experience[0].bullets == ORIGINALS
    assert result.summary == setup[0].summary
    assert report.invalid_rewrite_response == response
    assert report.warnings
    assert len(report.rewrites) == 4
    assert len(calls) == 1


def test_invalid_review_json_keeps_changed_originals(setup):
    result, report, _, _ = run(setup, [proposal(), "not JSON"])
    assert result.experience[0].bullets == ORIGINALS
    assert report.invalid_review_response == "not JSON"


@pytest.mark.parametrize("stage", ["draft", "review"])
def test_transport_errors_propagate(setup, stage):
    responses = [RuntimeError("transport failed")]
    if stage == "review":
        responses.insert(0, proposal())
    with pytest.raises(RuntimeError, match="transport failed"):
        run(setup, responses)
