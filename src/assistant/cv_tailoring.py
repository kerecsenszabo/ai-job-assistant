"""Evidence-first job matching and conservative, audited CV rewriting."""

from __future__ import annotations

from pathlib import Path

from langchain_core.runnables import Runnable

from assistant.cv_generator import CV, TailorDiagnostics
from assistant.performance import measure_run, measure_stage
from assistant.cv_tailoring_core import (
    CriterionMatch,
    Draft,
    DraftSentence,
    EvidenceItem,
    Matches,
    ModelOutputError,
    ParsedJob,
    Requirement,
    RequirementCriterion,
    RequirementMatch,
    Review,
    RewriteAudit,
    ScoringRubric,
    SemanticDecisions,
    SentenceVerdict,
    StrictModel,
    SummaryDraft,
    TailoringReport,
    UnquotedOptionError,
    canonical_text,
    normalized,
    request,
    source_evidence,
)
from assistant.cv_matching import (
    checked_matches,
    enforce_match_rules,
    explicit_criterion_match,
    guard_semantic_decision,
    language_proficiency_supported,
    match_job,
    mentions,
    retrieval_terms,
    retrieve_evidence,
    score_matches,
    select_evidence,
    semantic_decisions,
    unkey_semantic_decisions,
)
from assistant.job_parsing import (
    assessable_job_lines,
    collapse_repeated_heading,
    is_job_heading,
    job_description_chunks,
    job_extraction_schema,
    normalize_criteria,
    parse_job,
    prepare_job_description,
    quoted_capability_span,
    recover_unquoted_options,
    source_clauses,
    source_importance,
    source_option,
    source_option_pattern,
    source_requirement_quotes,
    split_criterion,
)
from assistant.cv_rewriting import (
    capture_rewrite_response,
    mechanical_rejection,
    rewrite,
    rewrite_review_schema,
    rewrite_schema,
    summary_voice_rejection,
)


def _tailor_with_report(
    cv: CV,
    description: str,
    llm: Runnable,
    diagnostics: TailorDiagnostics | None = None,
    rubric: ScoringRubric | None = None,
    *,
    job_cache: Path | None = None,
    refresh_job_analysis: bool = False,
    matching_cache: Path | None = None,
    refresh_matching: bool = False,
    model_identity: str | None = None,
) -> tuple[CV, TailoringReport]:
    from assistant.job_requirements import get_parsed_job
    from assistant.match_cache import get_cached_matches

    if not description.strip():
        raise ValueError("Job description cannot be empty.")
    evidence = source_evidence(cv)
    with measure_stage("job_analysis"):
        job, fingerprint, cache_hit = get_parsed_job(
            description, llm, cache_dir=job_cache, refresh=refresh_job_analysis
        )
    effective_rubric = rubric or ScoringRubric()
    with measure_stage("matching"):
        matches, matching_fingerprint, matching_hit = get_cached_matches(
            cv,
            description,
            job,
            evidence,
            llm,
            cache_dir=matching_cache,
            model_identity=model_identity,
            rubric=effective_rubric,
            refresh=refresh_matching or refresh_job_analysis,
        )
    report = TailoringReport(
        requirements=job.requirements,
        warnings=job.warnings.copy(),
        evidence=evidence,
        matches=matches,
        rubric=effective_rubric,
        job_fingerprint=fingerprint,
        job_analysis_cached=cache_hit,
        matching_fingerprint=matching_fingerprint,
        matching_analysis_cached=matching_hit,
        model_identity=model_identity,
    )
    if diagnostics is not None:
        diagnostics.report = report
    report.match_percent, report.must_have_percent = score_matches(
        job.requirements, matches, report.rubric
    )
    report.unresolved_eligibility_ids = [
        match.requirement_id
        for requirement, match in zip(job.requirements, matches)
        if requirement.importance == "eligibility" and match.status != "direct"
    ]
    report.warnings.extend(
        f"{match.requirement_id}: non-supporting references are recorded only "
        "as inspected evidence, not as a positive match."
        for match in matches
        if match.status == "not_evidenced" and match.inspected_evidence_ids
    )
    if not job.requirements:
        report.warnings.append(
            "Insufficient information: no assessable job requirements."
        )
        report.selected_evidence_ids = [
            item.id
            for item in evidence
            if item.section in ("skills", "ai_native", "experience")
        ]
        return cv.model_copy(deep=True), report
    with measure_stage("selection"):
        selected = select_evidence(report)
    tailored = cv.model_copy(
        update={
            "skills": [item.text for item in selected if item.section == "skills"],
            "ai_native": [
                item.text for item in selected if item.section == "ai_native"
            ],
            "experience": [
                role.model_copy(
                    update={
                        "bullets": [
                            item.text
                            for item in selected
                            if item.section == "experience" and item.role_index == index
                        ]
                    }
                )
                for index, role in enumerate(cv.experience)
            ],
        }
    )
    with measure_stage("rewriting"):
        result = rewrite(tailored, llm, report, selected, description, diagnostics)
    return result, report


def tailor_with_report(
    cv: CV,
    description: str,
    llm: Runnable,
    diagnostics: TailorDiagnostics | None = None,
    rubric: ScoringRubric | None = None,
    *,
    job_cache: Path | None = None,
    refresh_job_analysis: bool = False,
    matching_cache: Path | None = None,
    refresh_matching: bool = False,
    model_identity: str | None = None,
) -> tuple[CV, TailoringReport]:
    with measure_run() as performance:
        result, report = _tailor_with_report(
            cv,
            description,
            llm,
            diagnostics,
            rubric,
            job_cache=job_cache,
            refresh_job_analysis=refresh_job_analysis,
            matching_cache=matching_cache,
            refresh_matching=refresh_matching,
            model_identity=model_identity,
        )
        report.performance = performance
        return result, report


def polish_with_report(cv: CV, llm: Runnable) -> tuple[CV, TailoringReport]:
    with measure_run() as performance:
        report = TailoringReport(
            label="General-purpose CV rewrite audit",
            evidence=source_evidence(cv),
            performance=performance,
        )
        selected = [item for item in report.evidence if item.section == "experience"]
        report.selected_evidence_ids = [item.id for item in selected]
        if not selected:
            return cv.model_copy(deep=True), report
        with measure_stage("rewriting"):
            result = rewrite(cv, llm, report, selected, None, None)
        return result, report
