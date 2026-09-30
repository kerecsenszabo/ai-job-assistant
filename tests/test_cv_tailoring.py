import json

import pytest
from langchain_core.runnables import RunnableLambda

from assistant.cv_generator import CV, Experience, TailorDiagnostics, tailor_cv
from assistant.cv_tailoring import (
    DraftSentence,
    EvidenceItem,
    Matches,
    Requirement,
    RequirementMatch,
    ScoringRubric,
    TailoringReport,
    checked_matches,
    mechanical_rejection,
    parse_job,
    polish_with_report,
    score_matches,
    select_evidence,
    source_evidence,
    tailor_with_report,
)


@pytest.fixture
def cv():
    return CV(
        name="Example Engineer",
        email="example@example.com",
        summary="I build data pipelines.",
        skills=["Python", "SQL", "AWS"],
        ai_native=["Reviewed AI-assisted code."],
        experience=[
            Experience(
                company="Current Co",
                role="Engineer",
                dates="2024 - Present",
                bullets=["Maintained reports.", "Built Python ETL pipelines."],
            ),
            Experience(
                company="Previous Co",
                role="Developer",
                dates="2020 - 2023",
                bullets=["Supported an internal prototype.", "Wrote SQL reports."],
            ),
        ],
        education=[{"institution": "Example University", "degree": "BSc"}],
        certifications=[{"name": "Example Certification"}],
        publications=[{"title": "Example Paper"}],
    )


def fake_llm(responses):
    remaining = iter(responses)
    calls = []

    def respond(prompt, **kwargs):
        calls.append((prompt, kwargs))
        response = next(remaining)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    return RunnableLambda(respond), calls


def requirements():
    return [
        Requirement(
            id="requirement/0", text="Python", quote="Python",
            importance="required",
        ),
        Requirement(
            id="requirement/1", text="Kubernetes", quote="Kubernetes",
            importance="preferred",
        ),
    ]


def matches():
    return [
        RequirementMatch(
            requirement_id="requirement/0", status="direct",
            evidence_ids=["skills/0", "experience/0/bullets/1"],
            explanation="Python pipelines explicitly described.",
        ),
        RequirementMatch(
            requirement_id="requirement/1", status="not_evidenced",
            evidence_ids=[], explanation="Not evidenced in this CV.",
        ),
    ]


def draft(bullet="Developed Python ETL pipelines.", summary="I build Python ETL pipelines."):
    return {"sentences": [
        {"id": "b0", "target_id": "experience/0/bullets/1",
         "source_ids": ["experience/0/bullets/1"], "text": bullet},
        {"id": "b1", "target_id": "experience/1/bullets/0",
         "source_ids": ["experience/1/bullets/0"],
         "text": "Supported an internal prototype."},
        {"id": "s0", "target_id": "summary",
         "source_ids": ["experience/0/bullets/1"], "text": summary},
    ]}


def review(bullet_status="supported", summary_status="supported"):
    return {"verdicts": [
        {"id": "b0", "status": bullet_status, "reason": "Evidence review."},
        {"id": "b1", "status": "supported", "reason": "Unchanged source."},
        {"id": "s0", "status": summary_status, "reason": "Evidence review."},
    ]}


def pipeline_responses(proposal=None, verdict=None):
    job = {"requirements": [item.model_dump() for item in requirements()]}
    mapping = {"matches": [item.model_dump() for item in matches()]}
    return [job, mapping, mapping, proposal or draft(), verdict or review()]


def test_evidence_paths_cover_all_sections_and_preserve_source(cv):
    evidence = source_evidence(cv)
    assert len({item.id for item in evidence}) == len(evidence)
    by_id = {item.id: item for item in evidence}
    assert by_id["experience/0/bullets/1"].text == "Built Python ETL pipelines."
    assert "Current Co" in by_id["experience/0/bullets/1"].context
    assert by_id["experience/0"].section == "role"
    assert {"summary", "skills", "ai_native", "role", "experience",
            "education", "publications", "certifications"} == {
                item.section for item in evidence
            }


def test_pipeline_rewrites_supported_facts_and_keeps_source_immutable(cv):
    original = cv.model_dump_json()
    llm, calls = fake_llm(pipeline_responses())
    diagnostics = TailorDiagnostics()
    result, report = tailor_with_report(
        cv, "Python required; Kubernetes preferred.", llm, diagnostics
    )
    assert result.skills == ["Python"]
    assert result.ai_native == []
    assert result.experience[0].bullets == ["Developed Python ETL pipelines."]
    assert result.experience[1].bullets == ["Supported an internal prototype."]
    assert result.summary == "I build Python ETL pipelines."
    assert report.match_percent == 75
    assert report.must_have_percent == 100
    assert report.contextual_evidence_ids == ["experience/1/bullets/0"]
    assert all(item.status == "accepted" for item in report.rewrites)
    assert diagnostics.report is report
    assert diagnostics.summary_attempts == 1
    assert cv.model_dump_json() == original
    for section in ("name", "email", "education", "certifications", "publications"):
        assert getattr(result, section) == getattr(cv, section)
    assert [(r.company, r.role, r.dates) for r in result.experience] == [
        (r.company, r.role, r.dates) for r in cv.experience
    ]
    assert len(calls) == 5
    assert all("format" in kwargs for _, kwargs in calls)
    match_schema = calls[1][1]["format"]["$defs"]["RequirementMatch"]["properties"]
    assert match_schema["requirement_id"]["enum"] == [
        "requirement/0", "requirement/1"
    ]
    assert set(match_schema["evidence_ids"]["items"]["enum"]) == {
        item.id for item in report.evidence
    }


@pytest.mark.parametrize("bullet,status", [
    ("Led enterprise-wide Python ETL development.", "unsupported"),
    ("Built production pipelines.", "unsupported"),
    ("Developed Python ETL pipelines.", "unclear"),
    ("Built Python ETL pipelines improving throughput by 90%.", "supported"),
    ("Built Python ETL pipelines on Kubernetes.", "supported"),
    # A skill elsewhere in the CV is not evidence of using it in this role.
    ("Built Python ETL pipelines on AWS.", "supported"),
])
def test_unsafe_or_uncertain_bullets_retain_original(cv, bullet, status):
    llm, _ = fake_llm(pipeline_responses(draft(bullet=bullet), review(status)))
    result, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert result.experience[0].bullets == ["Built Python ETL pipelines."]
    audit = next(item for item in report.rewrites
                 if item.target_id == "experience/0/bullets/1")
    assert audit.status != "accepted"
    assert audit.exported == audit.original
    assert report.warnings
    assert report.match_percent == 75


def test_whole_summary_falls_back_when_one_sentence_is_unsupported(cv):
    proposal = draft()
    proposal["sentences"].append({
        "id": "s1", "target_id": "summary",
        "source_ids": ["experience/1/bullets/0"],
        "text": "I led production deployment.",
    })
    verdict = review()
    verdict["verdicts"].append({
        "id": "s1", "status": "unsupported", "reason": "Prototype, not production."
    })
    llm, _ = fake_llm(pipeline_responses(proposal, verdict))
    diagnostics = TailorDiagnostics()
    result, report = tailor_with_report(cv, "Python and Kubernetes", llm, diagnostics)
    assert result.summary == cv.summary
    assert diagnostics.summary_fallback
    summary_audits = [item for item in report.rewrites if item.target_id == "summary"]
    assert all(item.exported == cv.summary for item in summary_audits)
    assert not any(item.status == "accepted" for item in summary_audits)


@pytest.mark.parametrize("proposal", [
    "not json",
    {"sentences": []},
    {"sentences": [
        {"id": "b0", "target_id": "experience/0/bullets/1",
         "source_ids": ["skills/2"], "text": "Built AWS pipelines."},
    ]},
    {"sentences": [
        {"id": "s0", "target_id": "summary",
         "source_ids": ["nonexistent"], "text": "I build pipelines."},
    ]},
])
def test_invalid_draft_keeps_selected_originals_and_reports_failure(cv, proposal):
    responses = pipeline_responses()
    responses[3] = proposal
    llm, _ = fake_llm(responses[:4])
    result, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert result.summary == cv.summary
    assert result.experience[0].bullets == ["Built Python ETL pipelines."]
    assert report.rewrites
    assert all(item.status == "unclear" for item in report.rewrites)
    assert report.warnings
    assert report.invalid_rewrite_response


@pytest.mark.parametrize("verdict", [
    "invalid", {"verdicts": []},
    {"verdicts": [{"id": "wrong", "status": "supported", "reason": "OK"}]},
])
def test_invalid_review_cannot_authorize_any_rewrite(cv, verdict):
    responses = pipeline_responses()
    responses[4] = verdict
    llm, _ = fake_llm(responses)
    result, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert result.experience[0].bullets == ["Built Python ETL pipelines."]
    assert result.summary == cv.summary
    assert all(item.status == "unclear" for item in report.rewrites)
    assert report.invalid_review_response


def test_transport_errors_propagate_instead_of_success_shaped_fallback(cv):
    responses = pipeline_responses()
    responses[3] = ConnectionError("Ollama unavailable")
    llm, _ = fake_llm(responses)
    diagnostics = TailorDiagnostics()
    with pytest.raises(ConnectionError, match="unavailable"):
        tailor_with_report(cv, "Python and Kubernetes", llm, diagnostics)
    assert diagnostics.report.match_percent == 75


def test_unassessable_job_returns_no_percentage_and_no_rewrite(cv):
    llm, calls = fake_llm([{"requirements": []}])
    result, report = tailor_with_report(cv, "Welcome to our company.", llm)
    assert report.match_percent is None
    assert report.must_have_percent is None
    assert report.warnings
    assert result == cv
    assert len(calls) == 1


def test_sparse_cv_and_unresolved_eligibility():
    cv = CV(name="Example")
    job = {"requirements": [{
        "id": "r", "text": "Work authorization", "quote": "Work authorization",
        "importance": "eligibility",
    }]}
    mapping = {"matches": [{
        "requirement_id": "requirement/0", "status": "not_evidenced",
        "evidence_ids": [], "explanation": "Not evidenced in the CV.",
    }]}
    llm, _ = fake_llm([job, mapping, mapping, {"sentences": []}, {"verdicts": []}])
    result, report = tailor_with_report(cv, "Work authorization required.", llm)
    assert result.experience == []
    assert report.match_percent == 0
    assert report.must_have_percent == 0
    assert report.unresolved_eligibility_ids == ["requirement/0"]
    assert report.warnings


def test_requirement_quotes_and_deduplication():
    items = [item.model_dump() for item in requirements()]
    items.append(items[0])
    llm, _ = fake_llm([{"requirements": items}])
    assert len(parse_job(llm, "Python and Kubernetes").requirements) == 2
    llm, _ = fake_llm([{"requirements": items}])
    with pytest.raises(ValueError, match="source quote"):
        parse_job(llm, "SQL only")


def test_requirement_quotes_allow_wrapped_hyphenated_names():
    llm, _ = fake_llm([{"requirements": [{
        "id": "r", "text": "scikit-learn in production",
        "quote": "Experience with scikit-learn in production.",
        "importance": "required",
    }]}])
    job = parse_job(llm, "Experience with scikit-\nlearn in production.")
    assert job.requirements[0].text == "scikit-learn in production"


@pytest.mark.parametrize("change", [
    {"requirement_id": "nonexistent"},
    {"evidence_ids": ["nonexistent"]},
    {"evidence_ids": []},
    {"evidence_ids": ["skills/0", "skills/0"]},
])
def test_invalid_match_provenance_is_rejected(cv, change):
    items = matches()
    items[0] = items[0].model_copy(update=change)
    with pytest.raises(ValueError):
        checked_matches(Matches(matches=items), requirements(), source_evidence(cv))


def test_negative_match_citations_are_inspected_not_supporting_evidence(cv):
    mapping = matches()
    mapping[1].evidence_ids = ["skills/0"]
    result = checked_matches(
        Matches(matches=mapping), requirements(), source_evidence(cv)
    )
    assert result[1].evidence_ids == []
    assert result[1].inspected_evidence_ids == ["skills/0"]
    assert score_matches(requirements(), result, ScoringRubric()) == (75, 100)
    report = TailoringReport(
        requirements=requirements(), matches=result, evidence=source_evidence(cv)
    )
    assert "skills/0" in [item.id for item in select_evidence(report)]
    assert mapping[1].evidence_ids == ["skills/0"]


def test_invalid_matching_response_is_retried_with_feedback(cv):
    responses = pipeline_responses()
    invalid = {"matches": [item.model_dump() for item in matches()]}
    invalid["matches"][0]["evidence_ids"] = ["invented"]
    llm, calls = fake_llm([responses[0], invalid, *responses[1:]])
    _, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert report.match_percent == 75
    assert len(calls) == 6
    assert "previous response was invalid" in calls[2][0].to_string()


def test_repeated_invalid_matching_cannot_export(cv):
    responses = pipeline_responses()
    invalid = {"matches": []}
    llm, _ = fake_llm([responses[0], invalid, invalid])
    with pytest.raises(ValueError, match="every requirement"):
        tailor_with_report(cv, "Python and Kubernetes", llm)


def test_match_review_downgrade_changes_score_before_rewriting(cv):
    responses = pipeline_responses()
    responses[2]["matches"] = [
        item.model_dump() for item in matches()
    ]
    responses[2]["matches"][0]["status"] = "partial"
    llm, _ = fake_llm(responses)
    _, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert report.match_percent == 37.5
    assert report.must_have_percent == 50


def test_scoring_is_configurable_and_handles_uncertainty():
    mapping = matches()
    mapping[0].status = "partial"
    assert score_matches(requirements(), mapping, ScoringRubric()) == (37.5, 50)
    rubric = ScoringRubric(required_weight=1, partial_credit=0.25)
    assert score_matches(requirements(), mapping, rubric) == (12.5, 25)
    mapping[0].status = "unclear"
    assert score_matches(requirements(), mapping, rubric) == (0, 0)
    assert score_matches([], [], rubric) == (None, None)


def test_client_context_is_retained_when_anchor_is_not_relevant():
    cv = CV(name="Example", experience=[Experience(
        company="Consultancy", role="Engineer", dates="2020",
        bullets=["Client A: Built a prototype.", "Maintained Python pipelines.",
                 "Client B: Wrote SQL reports."],
    )])
    mapping = matches()[:1]
    mapping[0].evidence_ids = ["experience/0/bullets/1"]
    report = TailoringReport(
        evidence=source_evidence(cv), requirements=requirements()[:1], matches=mapping
    )
    selected = select_evidence(report)
    assert [item.text for item in selected] == cv.experience[0].bullets[:2]
    assert report.contextual_evidence_ids == ["experience/0/bullets/0"]
    source = selected[0]
    proposal = DraftSentence(
        id="b", target_id=source.id, source_ids=[source.id], text="Built a prototype."
    )
    assert "prefix" in mechanical_rejection(proposal, [source])


def test_selection_caps_without_topping_up_unrelated_items(cv):
    cv.skills = [f"Skill {index}" for index in range(20)]
    mapping = matches()
    mapping[0].evidence_ids = [f"skills/{index}" for index in range(20)]
    report = TailoringReport(
        evidence=source_evidence(cv), requirements=requirements(), matches=mapping
    )
    selected = select_evidence(report)
    assert len([item for item in selected if item.section == "skills"]) == 10
    assert len([item for item in selected if item.section == "experience"]) == 2
    assert not any(item.section == "ai_native" for item in selected)


def test_general_polish_preserves_structure_and_rejects_invention(cv):
    selected = [item for item in source_evidence(cv) if item.section == "experience"]
    proposal = {"sentences": [
        {"id": str(index), "target_id": item.id, "source_ids": [item.id],
         "text": "Led a production team." if index == 0 else item.text}
        for index, item in enumerate(selected)
    ]}
    verdict = {"verdicts": [
        {"id": str(index), "status": "unsupported" if index == 0 else "supported",
         "reason": "Evidence review."}
        for index in range(len(selected))
    ]}
    llm, _ = fake_llm([proposal, verdict])
    result, report = polish_with_report(cv, llm)
    assert result == cv
    assert report.rewrites[0].status == "rejected"
    assert report.warnings
    assert report.match_percent is None


def test_cv_return_api_exposes_report_through_diagnostics(cv):
    llm, _ = fake_llm(pipeline_responses())
    diagnostics = TailorDiagnostics()
    result = tailor_cv(cv, "Python and Kubernetes", llm=llm, diagnostics=diagnostics)
    assert isinstance(result, CV)
    assert diagnostics.report.match_percent == 75


def test_named_terms_must_match_tokens_not_substrings():
    source = EvidenceItem(id="b", section="experience", text="Worked with Sparkling.")
    sentence = DraftSentence(id="s", target_id="b", source_ids=["b"], text="Used Spark.")
    assert "Spark" in mechanical_rejection(sentence, [source])
