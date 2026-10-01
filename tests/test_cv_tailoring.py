import json
import re

import pytest
from langchain_core.runnables import RunnableLambda

from assistant.cv_generator import CV, Experience, TailorDiagnostics, tailor_cv
from assistant.cv_tailoring import (
    DraftSentence,
    EvidenceItem,
    Matches,
    ModelOutputError,
    ParsedJob,
    Requirement,
    RequirementCriterion,
    RequirementMatch,
    CriterionMatch,
    ScoringRubric,
    TailoringReport,
    assessable_job_lines,
    checked_matches,
    enforce_match_rules,
    explicit_criterion_match,
    job_description_chunks,
    job_extraction_schema,
    match_job,
    mechanical_rejection,
    parse_job,
    prepare_job_description,
    polish_with_report,
    request,
    score_matches,
    select_evidence,
    source_evidence,
    source_option,
    source_option_pattern,
    tailor_with_report,
    unkey_semantic_decisions,
)


@pytest.fixture
def cv():
    return CV(
        name="Example Engineer",
        email="example@example.com",
        summary="I build data pipelines.",
        skills=["Python", "SQL", "AWS"],
        ai_native=["Reviewed AI-assisted code."],
        languages=[
            {"name": "English", "proficiency": "Native"},
            {"name": "Hungarian", "proficiency": "Native"},
            {"name": "German", "proficiency": "Basic"},
        ],
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
            importance="required", criteria=[RequirementCriterion(
                id="requirement/0/criterion/0", text="Python", quote="Python",
                kind="technology", options=["Python"],
            )],
        ),
        Requirement(
            id="requirement/1", text="Kubernetes", quote="Kubernetes",
            importance="preferred", criteria=[RequirementCriterion(
                id="requirement/1/criterion/0", text="Kubernetes", quote="Kubernetes",
                kind="technology", options=["Kubernetes"],
            )],
        ),
    ]


def matches():
    return [
        RequirementMatch(
            requirement_id="requirement/0", status="direct",
            evidence_ids=["skills/0", "experience/0/bullets/1"],
            explanation="Python pipelines explicitly described.",
            criteria_matches=[CriterionMatch(
                criterion_id="requirement/0/criterion/0", status="direct",
                evidence_ids=["skills/0", "experience/0/bullets/1"],
                explanation="Python explicitly described.",
            )],
        ),
        RequirementMatch(
            requirement_id="requirement/1", status="not_evidenced",
            evidence_ids=[], explanation="Not evidenced in this CV.",
            criteria_matches=[CriterionMatch(
                criterion_id="requirement/1/criterion/0", status="not_evidenced",
                evidence_ids=[], explanation="Not evidenced in this CV.",
            )],
        ),
    ]


def draft(bullet="Developed Python ETL pipelines.", summary="I build Python ETL pipelines."):
    return {
        "bullets": {
            "experience/0/bullets/0": {
                "source_ids": ["experience/0/bullets/0"],
                "text": "Maintained reports.",
            },
            "experience/0/bullets/1": {
                "source_ids": ["experience/0/bullets/1"], "text": bullet,
            },
            "experience/1/bullets/0": {
                "source_ids": ["experience/1/bullets/0"],
                "text": "Supported an internal prototype.",
            },
            "experience/1/bullets/1": {
                "source_ids": ["experience/1/bullets/1"],
                "text": "Wrote SQL reports.",
            },
        },
        "summary_sentences": [{
            "id": "summary/0", "source_ids": ["experience/0/bullets/1"],
            "text": summary,
        }],
    }


def review(bullet_status="supported", summary_status="supported"):
    return {"verdicts": {
        "experience/0/bullets/1": {
            "status": bullet_status, "reason": "Evidence review.",
        },
        "summary/0": {"status": summary_status, "reason": "Evidence review."},
    }}


def pipeline_responses(proposal=None, verdict=None):
    job = {"requirements": [item.model_dump() for item in requirements()]}
    return [job, proposal if proposal is not None else draft(),
            verdict if verdict is not None else review()]


def test_evidence_paths_cover_all_sections_and_preserve_source(cv):
    evidence = source_evidence(cv)
    assert len({item.id for item in evidence}) == len(evidence)
    by_id = {item.id: item for item in evidence}
    assert by_id["experience/0/bullets/1"].text == "Built Python ETL pipelines."
    assert "Current Co" in by_id["experience/0/bullets/1"].context
    assert by_id["experience/0"].section == "role"
    assert {"summary", "skills", "ai_native", "languages", "role", "experience",
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
    assert result.experience[0].bullets == [
        "Maintained reports.", "Developed Python ETL pipelines.",
    ]
    assert result.experience[1].bullets == cv.experience[1].bullets
    assert result.summary == "I build Python ETL pipelines."
    assert report.match_percent == 75
    assert report.must_have_percent == 100
    assert report.contextual_evidence_ids == [
        "experience/0/bullets/0",
        "experience/1/bullets/0", "experience/1/bullets/1",
    ]
    assert all(item.status == "accepted" for item in report.rewrites)
    assert diagnostics.report is report
    assert cv.model_dump_json() == original
    for section in ("name", "email", "education", "certifications", "publications", "languages"):
        assert getattr(result, section) == getattr(cv, section)
    assert [(r.company, r.role, r.dates) for r in result.experience] == [
        (r.company, r.role, r.dates) for r in cv.experience
    ]
    assert len(calls) == 3
    assert all("format" in kwargs for _, kwargs in calls)
    assert calls[1][1]["format"]["properties"]["bullets"]["required"] == [
        "experience/0/bullets/0", "experience/0/bullets/1",
        "experience/1/bullets/0", "experience/1/bullets/1",
    ]


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
    assert result.experience[0].bullets == cv.experience[0].bullets
    audit = next(item for item in report.rewrites
                 if item.target_id == "experience/0/bullets/1")
    assert audit.status != "accepted"
    assert audit.exported == audit.original
    assert report.warnings
    assert report.match_percent == 75


def test_whole_summary_falls_back_when_one_sentence_is_unsupported(cv):
    proposal = draft()
    proposal["summary_sentences"].append({
        "id": "summary/1",
        "source_ids": ["experience/1/bullets/0"],
        "text": "I led production deployment.",
    })
    verdict = review()
    verdict["verdicts"]["summary/1"] = {
        "status": "unsupported", "reason": "Prototype, not production.",
    }
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
    {"bullets": {"experience/0/bullets/1": {
        "source_ids": ["skills/2"], "text": "Built AWS pipelines.",
    }}},
    {"bullets": {}, "summary_sentences": [{
        "id": "summary/0", "source_ids": ["nonexistent"],
        "text": "I build pipelines.",
    }]},
])
def test_invalid_draft_keeps_selected_originals_and_reports_failure(cv, proposal):
    responses = pipeline_responses()
    responses[1] = proposal
    llm, _ = fake_llm(responses[:2])
    result, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert result.summary == cv.summary
    assert result.experience[0].bullets == cv.experience[0].bullets
    assert report.rewrites
    assert all(item.status == "unclear" for item in report.rewrites)
    assert report.warnings
    assert report.invalid_rewrite_response


@pytest.mark.parametrize("verdict", [
    "invalid", {"verdicts": []},
    {"verdicts": [{"id": "wrong", "status": "supported", "reason": "OK"}]},
])
def test_invalid_review_blocks_changed_text_but_not_unchanged_originals(cv, verdict):
    responses = pipeline_responses()
    responses[2] = verdict
    llm, _ = fake_llm(responses)
    result, report = tailor_with_report(cv, "Python and Kubernetes", llm)
    assert result.experience[0].bullets == cv.experience[0].bullets
    assert result.summary == cv.summary
    assert all(item.exported == item.original for item in report.rewrites)
    assert all(item.status == "unclear" for item in report.rewrites
               if item.proposed != item.original)
    assert any(item.status == "accepted" and item.reason == "Unchanged source."
               for item in report.rewrites)
    assert report.invalid_review_response


def test_transport_errors_propagate_instead_of_success_shaped_fallback(cv):
    responses = pipeline_responses()
    responses[1] = ConnectionError("Ollama unavailable")
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
        "criteria": [{
            "id": "c", "text": "Work authorization", "quote": "Work authorization",
        }],
    }]}
    decision = {"decisions": {"c0": {
        "status": "not_evidenced", "evidence_ids": [],
        "explanation": "Not evidenced in the CV.",
    }}}
    llm, _ = fake_llm([job, decision, decision, {
        "bullets": {}, "summary_sentences": [],
    }])
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


def test_request_exposes_output_schema_in_prompt():
    llm, calls = fake_llm([{"requirements": []}])
    request(llm, ParsedJob, "Parse the job.", {"job_description": "Python"})
    prompt, kwargs = calls[0]
    system = prompt.to_messages()[0].content
    assert json.loads(system.split("Output JSON schema:\n", 1)[1]) == kwargs["format"]
    assert kwargs["format"] == ParsedJob.model_json_schema()


def test_parse_job_exposes_custom_constraints_in_prompt():
    llm, calls = fake_llm([{"requirements": []}])
    parse_job(llm, "Python")
    prompt, kwargs = calls[0]
    system = prompt.to_messages()[0].content
    assert json.loads(system.split("Output JSON schema:\n", 1)[1]) == kwargs["format"]
    requirement = kwargs["format"]["$defs"]["Requirement"]
    assert "criteria" in requirement["required"]
    assert requirement["properties"]["criteria"]["minItems"] == 1


def keyed_requirement(options=None):
    return {"requirements": {"source/0": {
        "importance": "required",
        "criteria": [{
            "text": "Python", "kind": "technology",
            "options": ["Python"] if options is None else options,
        }],
    }}}


def test_parse_job_attaches_source_text_and_stable_ids():
    llm, calls = fake_llm([keyed_requirement()])
    parsed = parse_job(llm, "Build Python pipelines.").requirements[0]
    assert parsed.text == parsed.quote == "Build Python pipelines."
    assert parsed.id == "requirement/0"
    assert parsed.criteria[0].quote in parsed.quote
    assert parsed.criteria[0].id == "requirement/0/criterion/0"
    assert len(calls) == 1


def test_parse_job_accepts_non_assessable_source_lines():
    llm, _ = fake_llm([{"requirements": {"source/0": None}}])
    assert parse_job(llm, "Employee benefits").requirements == []


def test_parser_excludes_headings_and_benefits_but_keeps_later_requirements():
    description = (
        "The Position\n"
        "Own production ML systems.\n"
        "You Will Have an Opportunity to\n"
        "Build data pipelines.\n"
        "What We Are Looking For\n"
        "Strong Python skills.\n"
        "Why You'll Love Working Here\n"
        "Innovative Environment: work with cutting-edge technology.\n"
        "Professional Growth: develop your career.\n"
        "Responsibilities\n"
        "Maintain production models.\n"
    )
    expected = [
        "Own production ML systems.",
        "Build data pipelines.",
        "Strong Python skills.",
        "Maintain production models.",
    ]
    assert assessable_job_lines(description) == expected

    def respond(prompt, **kwargs):
        sources = json.loads(prompt.to_messages()[-1].content)["source_lines"]
        return json.dumps({"requirements": {
            source_id: {
                "importance": "required",
                "criteria": [{"text": text, "kind": "general", "options": []}],
            }
            for source_id, text in sources.items()
        }})

    parsed = parse_job(RunnableLambda(respond), description)
    assert [item.quote for item in parsed.requirements] == expected
    assert parsed.requirements[-1].importance == "responsibility"


def test_parser_skips_benefits_only_job_without_model_calls():
    llm, calls = fake_llm([])
    parsed = parse_job(
        llm, "Why You'll Love Working Here\nProfessional Growth: learn new skills."
    )
    assert parsed.requirements == []
    assert not calls


def test_parser_restores_wrapped_requirements_and_excludes_repeated_benefits():
    description = (
        "Responsibilities Responsibilities Responsibilities Responsibilities\n"
        "Build global demand forecasting systems across\n"
        "prestige beauty brands, using Spark and\n"
        "Delta Lake on Databricks.\n"
        "Requirements Requirements Requirements Requirements\n"
        "Programming & Tooling Programming & Tooling Programming & Tooling "
        "Programming & Tooling: Expert in Python and\n"
        "scikit-learn for production models.\n"
        "Our hybrid work model Our hybrid work model Our hybrid work model "
        "Our hybrid work model\n"
        "Work from home up to 12 days per month.\n"
        "Our Benefits Our Benefits Our Benefits Our Benefits\n"
        "Private health insurance and training.\n"
    )
    expected = [
        "Build global demand forecasting systems across prestige beauty brands, "
        "using Spark and Delta Lake on Databricks.",
        "Programming & Tooling: Expert in Python and scikit-learn for production models.",
    ]
    assert assessable_job_lines(description) == expected
    assert "Responsibilities\n" in prepare_job_description(description)

    def respond(prompt, **kwargs):
        sources = json.loads(prompt.to_messages()[-1].content)["source_lines"]
        return json.dumps({"requirements": {
            source_id: {
                "importance": "required",
                "criteria": [{"text": text, "kind": "general", "options": []}],
            }
            for source_id, text in sources.items()
        }})

    parsed = parse_job(RunnableLambda(respond), description)
    assert [item.quote for item in parsed.requirements] == expected
    assert [item.importance for item in parsed.requirements] == [
        "responsibility", "required",
    ]


def test_parser_retries_word_by_word_criteria():
    source = "Build robust modular scalable machine learning pipelines across global prestige brands."
    verbose = {"requirements": {"source/0": {
        "importance": "responsibility",
        "criteria": [
            {"text": word, "kind": "general", "options": []}
            for word in source.rstrip(".").split()
        ],
    }}}
    concise = {"requirements": {"source/0": {
        "importance": "responsibility",
        "criteria": [{"text": source, "kind": "general", "options": []}],
    }}}
    llm, calls = fake_llm([verbose, concise])
    parsed = parse_job(llm, source)
    assert len(calls) == 2
    assert len(parsed.requirements[0].criteria) == 1


def test_parser_retries_missing_labeled_responsibility():
    description = "Responsibilities\nBuild Python models.\nMaintain data pipelines."
    response = {"requirements": {"source/0": {
        "importance": "responsibility",
        "criteria": [{"text": "Build Python models", "kind": "general", "options": []}],
    }}}
    complete = {"requirements": {"source/0": response["requirements"]["source/0"],
                                  "source/1": {
        "importance": "responsibility",
        "criteria": [{"text": "Maintain data pipelines",
                      "kind": "general", "options": []}],
    }}}
    llm, calls = fake_llm([response, complete])
    assert len(parse_job(llm, description).requirements) == 2
    assert len(calls) == 2


def test_parser_rejects_ordinary_words_as_technologies():
    bad = {"requirements": {"source/0": {
        "importance": "required",
        "criteria": [{"text": "training", "kind": "technology",
                      "options": ["training"]}],
    }}}
    good = {"requirements": {"source/0": {
        "importance": "required",
        "criteria": [{"text": "Automated training", "kind": "general",
                      "options": []}],
    }}}
    llm, calls = fake_llm([bad, good])
    assert parse_job(llm, "Automated training").requirements[0].criteria[0].kind == "general"
    assert len(calls) == 2


def test_parse_job_retries_invalid_options_with_validation_feedback():
    llm, calls = fake_llm([keyed_requirement(["Rust"]), keyed_requirement()])
    parsed = parse_job(llm, "Python")
    assert parsed.requirements[0].criteria[0].options == ["Python"]
    assert len(calls) == 2
    assert "'Rust'" in calls[1][0].to_messages()[0].content


@pytest.mark.parametrize("response", [
    "invalid JSON",
    keyed_requirement(["Rust"]),
    {"requirements": {"unknown/source": None}},
])
def test_parse_job_exhausted_retries_fail_explicitly(response):
    llm, calls = fake_llm([response] * 3)
    with pytest.raises(ModelOutputError):
        parse_job(llm, "Python")
    assert len(calls) == 3


def test_long_job_parsing_preserves_original_section_priority():
    first_line = "Build Python pipelines. " + "Reliable " * 85
    description = first_line + "\nNice-to-have\nAWS"
    assert len(description) > 800
    aws = keyed_requirement(["AWS"])
    aws["requirements"]["source/0"]["criteria"][0]["text"] = "AWS"
    response = keyed_requirement()
    response["requirements"]["source/1"] = aws["requirements"]["source/0"]
    llm, calls = fake_llm([response])
    parsed = parse_job(llm, description)
    assert len(calls) == 1
    assert [item.id for item in parsed.requirements] == ["requirement/0", "requirement/1"]
    assert parsed.requirements[1].importance == "preferred"
    assert parsed.requirements[0].quote == first_line.strip()


def test_job_chunks_preserve_source_content():
    description = "Python and SQL\n" + "Build reliable pipelines " * 100
    chunks = job_description_chunks(description)
    assert all(len(chunk) <= 800 for chunk in chunks)
    assert " ".join(description.split()) == " ".join(" ".join(chunks).split())
    assert job_description_chunks("") == [""]


def test_extraction_schema_bounds_arrays_by_source_size():
    schema = job_extraction_schema({"source/0": "Python or Amazon Web Services"})
    requirement = schema["properties"]["requirements"]["properties"]["source/0"]["anyOf"][0]
    assert requirement["properties"]["criteria"]["maxItems"] == 5
    assert "text" not in requirement["properties"]
    criteria = requirement["properties"]["criteria"]["items"]["oneOf"]
    assert criteria[0]["properties"]["options"]["maxItems"] == 0
    assert criteria[1]["properties"]["options"]["maxItems"] == 5


@pytest.mark.parametrize("option, quote, expected", [
    ("tool_integration", "Build tool integration.", "tool integration"),
    ("execution-history", "Build execution history.", "execution history"),
    ("world's", "The world\u2019s infrastructure", "world\u2019s"),
    ("Rust", "Python services", "Rust"),
    ("C++", "C services", "C++"),
    ("foo-bar", "foo bar and foo-bar", "foo-bar"),
])
def test_source_option_repair_is_limited_to_source_formatting(option, quote, expected):
    assert source_option(option, quote) == expected


def test_source_option_pattern_preserves_multiword_and_punctuated_names():
    pattern = source_option_pattern("Use .NET, C++, and Amazon Web Services.")
    for option in (".NET", "C++", "Amazon Web Services"):
        assert re.fullmatch(pattern, option)
    for option in ("Rust", "C--", "Amazon Cloud Services", ""):
        assert re.fullmatch(pattern, option) is None


@pytest.mark.parametrize("response, detail", [
    ('{"requirements": [', "EOF"),
    ('{"requirements": [{}]}', "requirements.0.text: Field required"),
])
def test_request_reports_validation_details_without_discarding_response(response, detail):
    llm, calls = fake_llm([response])
    with pytest.raises(ModelOutputError, match="invalid ParsedJob JSON") as caught:
        request(llm, ParsedJob, "Parse the job.", {})
    assert detail in str(caught.value)
    assert caught.value.response == response
    assert len(calls) == 1


def test_requirement_quotes_allow_wrapped_hyphenated_names():
    llm, _ = fake_llm([{"requirements": [{
        "id": "r", "text": "scikit-learn in production",
        "quote": "Experience with scikit-learn in production.",
        "importance": "required",
        "criteria": [{
            "id": "c", "text": "scikit-learn in production",
            "quote": "Experience with scikit-learn in production.",
            "kind": "technology", "options": ["scikit-learn"],
        }],
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


def test_parsed_requirements_without_criteria_cannot_use_old_matching(cv):
    job = {"requirements": [{
        "id": "r", "text": "Python", "quote": "Python", "importance": "required",
        "criteria": [],
    }]}
    llm, calls = fake_llm([job])
    with pytest.raises(ValueError, match="no assessable criteria"):
        tailor_with_report(cv, "Python", llm)
    assert len(calls) == 1


def test_old_cached_job_without_criteria_cannot_bypass_current_matching(cv):
    job = ParsedJob(requirements=[Requirement(
        id="r", text="Python", quote="Python", importance="required",
    )])
    llm, calls = fake_llm([])
    with pytest.raises(ValueError, match="--refresh-job-analysis"):
        match_job(llm, job, source_evidence(cv))
    assert calls == []


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
    assert [item.text for item in selected] == cv.experience[0].bullets
    assert report.contextual_evidence_ids == [
        "experience/0/bullets/0", "experience/0/bullets/2",
    ]
    source = selected[0]
    proposal = DraftSentence(
        id="b", target_id=source.id, source_ids=[source.id], text="Built a prototype."
    )
    assert "prefix" in mechanical_rejection(proposal, [source])


def test_selection_caps_skills_and_retains_role_context(cv):
    cv.skills = [f"Skill {index}" for index in range(20)]
    mapping = matches()
    mapping[0].evidence_ids = [f"skills/{index}" for index in range(20)]
    report = TailoringReport(
        evidence=source_evidence(cv), requirements=requirements(), matches=mapping
    )
    selected = select_evidence(report)
    assert len([item for item in selected if item.section == "skills"]) == 14
    assert len([item for item in selected if item.section == "experience"]) == 4
    assert not any(item.section == "ai_native" for item in selected)


def test_selection_prioritizes_matches_then_preserves_context_within_role_budget():
    cv = CV(name="Example", experience=[
        Experience(
            company="Current Co", role="Engineer", dates="2024",
            bullets=[f"Delivered project {index}." for index in range(8)],
        ),
        Experience(
            company="Previous Co", role="Engineer", dates="2020",
            bullets=[f"Supported product {index}." for index in range(4)],
        ),
    ])
    mapping = matches()[:1]
    mapping[0].evidence_ids = [
        "experience/0/bullets/6", "experience/0/bullets/7",
    ]
    report = TailoringReport(
        evidence=source_evidence(cv), requirements=requirements()[:1], matches=mapping,
    )
    score = score_matches(report.requirements, report.matches, report.rubric)

    selected = select_evidence(report)

    assert [item.bullet_index for item in selected if item.role_index == 0] == [
        0, 1, 2, 3, 6, 7,
    ]
    assert [item.bullet_index for item in selected if item.role_index == 1] == [0, 1]
    assert report.contextual_evidence_ids == [
        *(f"experience/0/bullets/{index}" for index in range(4)),
        "experience/1/bullets/0", "experience/1/bullets/1",
    ]
    assert report.selected_evidence_ids == [item.id for item in selected]
    assert select_evidence(report) == selected
    assert len(report.contextual_evidence_ids) == 6
    assert score_matches(report.requirements, report.matches, report.rubric) == score


def test_six_relevant_bullets_survive_instead_of_being_cut_to_four():
    cv = CV(name="Example", experience=[Experience(
        company="Current Co", role="Engineer", dates="2024",
        bullets=[f"Delivered Python project {index}." for index in range(8)],
    )])
    mapping = matches()[:1]
    mapping[0].evidence_ids = [
        f"experience/0/bullets/{index}" for index in range(8)
    ]
    report = TailoringReport(
        evidence=source_evidence(cv), requirements=requirements()[:1], matches=mapping,
    )
    assert [item.text for item in select_evidence(report)] == cv.experience[0].bullets[:6]
    assert report.contextual_evidence_ids == []


def test_general_polish_preserves_structure_and_rejects_invention(cv):
    selected = [item for item in source_evidence(cv) if item.section == "experience"]
    proposal = {"bullets": {
        item.id: {"source_ids": [item.id],
                  "text": "Led a production team." if index == 0 else item.text}
        for index, item in enumerate(selected)
    }, "summary_sentences": []}
    verdict = {"verdicts": {
        selected[0].id: {"status": "unsupported", "reason": "Evidence review."},
    }}
    llm, _ = fake_llm([proposal, verdict])
    result, report = polish_with_report(cv, llm)
    assert result == cv
    rejected = next(item for item in report.rewrites if item.target_id == selected[0].id)
    assert rejected.status == "rejected"
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


def framework_requirement():
    text = (
        "Experience with leading ML frameworks such as PyTorch, TensorFlow, "
        "scikit-learn, or XGBoost in production environments."
    )
    return Requirement(
        id="requirement/0", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(
            id="requirement/0/criterion/0", text="Production ML frameworks",
            quote=text, kind="technology",
            options=["PyTorch", "TensorFlow", "scikit-learn", "XGBoost"],
            operator="any", production=True,
        )],
    )


def criterion_match(requirement, status, evidence_ids):
    return RequirementMatch(
        requirement_id=requirement.id, status=status, evidence_ids=evidence_ids,
        explanation="Model assessment.",
        criteria_matches=[CriterionMatch(
            criterion_id=criterion.id, status=status, evidence_ids=evidence_ids,
            explanation="Model assessment.",
        ) for criterion in requirement.criteria],
    )


def test_parser_preserves_source_any_of_and_production_qualifier():
    requirement = framework_requirement()
    requirement.criteria[0].operator = "all"
    requirement.criteria[0].production = False
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, requirement.quote).requirements[0]
    assert parsed.criteria[0].operator == "any"
    assert parsed.criteria[0].production


def test_lowercase_framework_names_still_form_a_single_any_of_group():
    text = "Experience with pytorch, tensorflow, scikit-learn, or xgboost in production."
    requirement = Requirement(
        id="r", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(
            id="c", text=text, quote=text, kind="technology",
            options=["pytorch", "tensorflow", "scikit-learn", "xgboost"],
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, text).requirements[0]
    assert len(parsed.criteria) == 1
    assert parsed.criteria[0].operator == "any"
    decision = explicit_criterion_match(
        parsed.criteria[0],
        [EvidenceItem(id="s", section="experience", text="Used XGBoost in production.")],
    )
    assert decision.status == "direct"


def test_production_qualifier_is_scoped_to_its_technology_group():
    text = "Experience with Kafka for messaging, and Spark or Flink for processing in production."
    requirement = Requirement(
        id="r", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(
            id="c", text=text, quote=text, kind="technology", production=True,
            options=["Kafka", "Spark", "Flink"],
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, text).requirements[0]
    kafka, processing = parsed.criteria
    assert kafka.options == ["Kafka"]
    assert not kafka.production
    assert processing.options == ["Spark", "Flink"]
    assert processing.production
    assert processing.operator == "any"


def test_parser_does_not_treat_agentic_capabilities_as_tool_alternatives():
    text = (
        "Experience with Agentic AI frameworks and multi-step workflow "
        "orchestration using LangChain, LangGraph, or similar tools."
    )
    requirement = Requirement(
        id="r", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(
            id="c", text=text, quote=text, kind="technology", operator="all",
            options=["Agentic AI frameworks", "multi-step workflow orchestration",
                     "LangChain", "LangGraph", "similar tools"],
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, text).requirements[0]
    tools = [item for item in parsed.criteria if item.kind == "technology"]
    assert len(tools) == 1
    assert tools[0].options == ["LangChain", "LangGraph"]
    assert tools[0].operator == "any"
    capabilities = {item.text for item in parsed.criteria if item.kind == "general"}
    assert capabilities == {"Agentic AI frameworks", "multi-step workflow orchestration"}


def test_parser_language_options_cannot_include_communication_skills():
    text = language_requirement().quote
    requirement = Requirement(
        id="r", text=text, quote=text, importance="preferred",
        criteria=[RequirementCriterion(
            id="c", text=text, quote=text, kind="language", operator="any",
            options=["English", "working proficiency", "strong communication skills"],
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, f"Requirements\n{text}").requirements[0]
    language = next(item for item in parsed.criteria if item.kind == "language")
    assert language.options == ["English"]
    assert language.proficiency == "professional working proficiency"
    assert any(item.kind == "general" and item.text == "strong communication skills"
               for item in parsed.criteria)


def test_parser_scopes_or_to_api_alternatives_not_other_llm_capabilities():
    text = "Experience with RAG, vector databases, and integration with OpenAI, Anthropic, or Bedrock."
    requirement = Requirement(
        id="r", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(
            id="c", text=text, quote=text, kind="technology", operator="any",
            options=["RAG", "vector databases", "OpenAI", "Anthropic", "Bedrock"],
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, text).requirements[0]
    groups = [item.options for item in parsed.criteria if item.kind == "technology"]
    assert ["RAG"] in groups
    assert ["OpenAI", "Anthropic", "Bedrock"] in groups
    assert any(item.text == "vector databases" and item.kind == "general"
               for item in parsed.criteria)


def test_source_required_section_overrides_preferred_english_classification():
    text = "Professional working proficiency in English with strong communication skills"
    requirement = Requirement(
        id="r", text=text, quote=text, importance="preferred",
        criteria=[RequirementCriterion(
            id="c", text="Communication skills", quote="strong communication skills"
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(
        llm, f"What we're looking for\n• {text}\nYour responsibilities\n• Write code."
    ).requirements[0]
    assert parsed.importance == "required"
    language = next(item for item in parsed.criteria if item.kind == "language")
    assert language.options == ["english"]
    assert language.proficiency == "professional working proficiency"


def test_preferred_section_remains_preferred():
    text = "Docker experience"
    requirement = Requirement(
        id="r", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(
            id="c", text=text, quote=text, kind="technology",
            options=["Docker"],
        )],
    )
    llm, _ = fake_llm([{"requirements": [requirement.model_dump()]}])
    parsed = parse_job(llm, f"Nice to have\n• {text}").requirements[0]
    assert parsed.importance == "preferred"


def test_production_alternatives_satisfied_without_other_frameworks(cv):
    requirement = framework_requirement()
    cv.experience[0].bullets.append("Work with scikit-learn and XGBoost models in production.")
    evidence = source_evidence(cv)
    proposed = criterion_match(
        requirement, "partial", ["experience/0/bullets/2"]
    )
    corrected = enforce_match_rules([requirement], [proposed], evidence)[0]
    assert corrected.status == "direct"
    assert "experience/0/bullets/2" in corrected.evidence_ids
    assert corrected.rule_adjustments
    assert score_matches([requirement], [corrected], ScoringRubric()) == (100, 100)
    assert proposed.status == "partial"


@pytest.mark.parametrize("text,section,expected", [
    ("XGBoost", "skills", "partial"),
    ("Prototyped scikit-learn models.", "experience", "partial"),
    ("Used XGBoost in non-production experiments.", "experience", "partial"),
    ("No experience with XGBoost in production.", "experience", "not_evidenced"),
    ("Built forecasting models in production.", "experience", "not_evidenced"),
    ("Worked with XGBoost in production.", "experience", "direct"),
])
def test_framework_direct_match_requires_named_production_evidence(text, section, expected):
    requirement = framework_requirement()
    match = explicit_criterion_match(
        requirement.criteria[0],
        [EvidenceItem(id="source", section=section, text=text)],
    )
    assert match.status == expected


def test_all_of_options_do_not_receive_direct_credit_from_one_tool():
    criterion = RequirementCriterion(
        id="c", text="Docker and Kubernetes", quote="Docker and Kubernetes",
        kind="technology", options=["Docker", "Kubernetes"], operator="all",
    )
    partial = explicit_criterion_match(
        criterion, [EvidenceItem(id="s", section="skills", text="Docker")]
    )
    assert partial.status == "partial"
    complete = explicit_criterion_match(
        criterion, [EvidenceItem(id="s", section="experience",
                                 text="Deployed Docker containers on Kubernetes.")]
    )
    assert complete.status == "direct"


def language_requirement():
    text = "Professional working proficiency in English with strong communication skills"
    return Requirement(
        id="requirement/0", text=text, quote=text, importance="required",
        criteria=[
            RequirementCriterion(
                id="requirement/0/criterion/0", text="Professional English",
                quote=text, kind="language", options=["English"],
                proficiency="professional working proficiency",
            ),
            RequirementCriterion(
                id="requirement/0/criterion/1", text="Communication",
                quote="strong communication skills",
            ),
        ],
    )


def test_english_written_cv_and_stakeholder_work_do_not_prove_language(cv):
    requirement = language_requirement()
    cv.summary = "I collaborate with international stakeholders."
    cv.languages = []
    proposed = criterion_match(requirement, "direct", ["summary/0"])
    corrected = enforce_match_rules(
        [requirement], [proposed], source_evidence(cv)
    )[0]
    assert corrected.status == "not_evidenced"
    assert corrected.evidence_ids == []
    assert corrected.criteria_matches[0].status == "not_evidenced"
    assert "stakeholder" in corrected.explanation


def test_declared_languages_provide_explicit_proficiency_evidence(cv):
    evidence = source_evidence(cv)
    criterion = language_requirement().criteria[0]
    english = explicit_criterion_match(criterion, evidence)
    assert english.status == "direct"
    assert english.evidence_ids == ["languages/0"]
    german = criterion.model_copy(update={"options": ["German"]})
    assert explicit_criterion_match(german, evidence).status == "partial"


@pytest.mark.parametrize("text,status", [
    ("Professional working proficiency in English.", "direct"),
    ("Fluent in English.", "direct"),
    ("Basic proficiency in English.", "partial"),
    ("Worked with English documentation.", "partial"),
    ("English literature degree.", "partial"),
    ("Worked with English clients while fluent in German.", "partial"),
    ("English (C1).", "direct"),
])
def test_language_proficiency_requires_explicit_evidence(text, status):
    criterion = language_requirement().criteria[0]
    result = explicit_criterion_match(
        criterion, [EvidenceItem(id="s", section="summary", text=text)]
    )
    assert result.status == status


def test_explicit_framework_matches_need_no_model_calls(cv):
    requirement = framework_requirement()
    cv.experience[0].bullets.append("Work with scikit-learn and XGBoost models in production.")
    llm, calls = fake_llm([])
    reviewed = match_job(
        llm, ParsedJob(requirements=[requirement]), source_evidence(cv)
    )
    assert reviewed[0].status == "direct"
    assert calls == []


def test_matching_requires_every_criterion(cv):
    requirement = framework_requirement()
    proposed = criterion_match(requirement, "partial", ["skills/0"])
    proposed.criteria_matches = []
    with pytest.raises(ValueError, match="every criterion"):
        checked_matches(Matches(matches=[proposed]), [requirement], source_evidence(cv))


def test_unresolved_semantic_criteria_use_compact_batches(cv):
    items = [
        Requirement(id=f"requirement/{index}", text="Python", quote="Python",
                    importance="required", criteria=[RequirementCriterion(
                        id=f"criterion/{index}", text="Python programming",
                        quote="Python programming", kind="general",
                    )])
        for index in range(13)
    ]
    responses = []
    for offset in range(0, len(items), 12):
        mapping = {"decisions": [
            CriterionMatch(
                criterion_id=item.criteria[0].id,
                status="direct", evidence_ids=["skills/0"],
                explanation="Python listed in source.",
            ).model_dump() for item in items[offset:offset + 12]
        ]}
        responses.extend([mapping, mapping])
    llm, calls = fake_llm(responses)
    reviewed = match_job(llm, ParsedJob(requirements=items), source_evidence(cv))
    assert [match.requirement_id for match in reviewed] == [item.id for item in items]
    assert len(calls) == 4
    sent = json.loads(calls[0][0].messages[-1].content)
    assert len(sent["evidence"]) < len(source_evidence(cv))
    assert all(item["kind"] == "general" for req in items for item in req.model_dump()["criteria"])


def test_semantic_review_does_not_receive_explicit_technology_criteria(cv):
    requirement = framework_requirement()
    cv.experience[0].bullets.append("Work with scikit-learn and XGBoost models in production.")
    requirement.criteria.append(RequirementCriterion(
        id="communication", text="Communication skills",
        quote="Communication skills", kind="general",
    ))
    cv.skills.append("Stakeholder collaboration")
    decision = {
        "criterion_id": "communication", "status": "partial",
        "evidence_ids": ["skills/3"], "explanation": "Collaboration is related.",
    }
    llm, calls = fake_llm([
        {"decisions": {"communication": decision}},
        {"decisions": {"communication": decision}},
    ])
    reviewed = match_job(llm, ParsedJob(requirements=[requirement]), source_evidence(cv))
    assert len(calls) == 2
    assert reviewed[0].criteria_matches[0].status == "direct"
    assert reviewed[0].status == "partial"
    sent = json.loads(calls[0][0].messages[-1].content)
    assert [item["id"] for item in sent["criteria"]] == ["c0"]
    assert "requirement/0/criterion/0" not in calls[0][0].to_string()


def test_rag_does_not_establish_direct_agentic_experience(cv):
    requirement = Requirement(
        id="r", text="Agentic AI frameworks", quote="Agentic AI frameworks",
        importance="required", criteria=[RequirementCriterion(
            id="c", text="Agentic AI frameworks", quote="Agentic AI frameworks"
        )],
    )
    cv.ai_native = ["Built RAG pipelines with LangChain."]
    proposed = criterion_match(requirement, "direct", ["ai_native/0"])
    result = enforce_match_rules([requirement], [proposed], source_evidence(cv))[0]
    assert result.status == "partial"
    assert "Direct credit withheld" in result.explanation


def test_releases_do_not_establish_ci_cd_and_debugging(cv):
    text = "Software engineering fundamentals including CI/CD and debugging"
    requirement = Requirement(
        id="r", text=text, quote=text, importance="required",
        criteria=[RequirementCriterion(id="c", text=text, quote=text)],
    )
    cv.experience[0].bullets = ["Reviewed pull requests and coordinated monthly releases."]
    proposed = criterion_match(requirement, "direct", ["experience/0/bullets/0"])
    result = enforce_match_rules([requirement], [proposed], source_evidence(cv))[0]
    assert result.status == "partial"
    cv.experience[0].bullets = ["Implemented CI/CD and debugged production services."]
    assert enforce_match_rules([requirement], [proposed], source_evidence(cv))[0].status == "direct"


def test_short_transport_aliases_are_expanded_in_audit_explanations():
    response = json.dumps({"decisions": {"c0": {
        "status": "direct", "evidence_ids": ["e0"],
        "explanation": "e0 supports c0.",
    }}})
    restored = json.loads(unkey_semantic_decisions(
        response, {"c0": "requirement/0/criterion/0"}, {"e0": "skills/0"}
    ))
    decision = restored["decisions"][0]
    assert decision["criterion_id"] == "requirement/0/criterion/0"
    assert decision["evidence_ids"] == ["skills/0"]
    assert decision["explanation"] == "skills/0 supports requirement/0/criterion/0."


def test_named_skills_are_kept_when_production_proof_cites_only_work_bullets(cv):
    cv.skills.extend(["scikit-learn", "XGBoost"])
    cv.experience[0].bullets.append("Work with scikit-learn and XGBoost models in production.")
    requirement = framework_requirement()
    proposed = criterion_match(requirement, "direct", ["experience/0/bullets/2"])
    report = TailoringReport(
        requirements=[requirement], matches=[proposed], evidence=source_evidence(cv)
    )
    selected = select_evidence(report)
    assert [item.text for item in selected if item.section == "skills"] == [
        "scikit-learn", "XGBoost",
    ]
