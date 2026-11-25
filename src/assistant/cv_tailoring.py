"""Evidence-first job matching and conservative, audited CV rewriting."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from assistant.generic_cv import GenericMLProfile


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
    model_name: str | None = None,
) -> tuple[CV, TailoringReport]:
    from assistant.job_requirements import get_parsed_job
    from assistant.match_cache import get_cached_matches

    if not description.strip():
        raise ValueError("Job description cannot be empty.")
    evidence = source_evidence(cv)
    with measure_stage("job_analysis"):
        job, fingerprint, cache_hit = get_parsed_job(
            description,
            llm,
            cache_dir=job_cache,
            refresh=refresh_job_analysis,
            model_name=model_name,
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
    model_name: str | None = None,
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
            model_name=model_name,
        )
        report.performance = performance
        return result, report


def polish_with_report(
    cv: CV,
    llm: Runnable,
    *,
    profile: GenericMLProfile | None = None,
) -> tuple[CV, TailoringReport]:
    """Build the default two-page machine learning engineer CV.

    Generic positioning selects a bounded evidence pool, then rewrites every
    exported bullet and the summary through the same audit path as job tailoring.
    The caller performs PDF page measurement after this function returns.
    """
    from assistant.generic_cv import (
        GENERIC_ML_PROFILE,
        GenericMLProfile,
        build_generic_cv,
        select_generic_evidence,
    )

    selected_profile = profile or GENERIC_ML_PROFILE
    if not isinstance(selected_profile, GenericMLProfile):
        raise ValueError("Generic CV profile has an unsupported type.")
    with measure_run() as performance:
        report = TailoringReport(
            label="Generic machine learning engineer CV",
            profile_name=selected_profile.name,
            requested_page_limit=selected_profile.max_pages,
            evidence=source_evidence(cv),
            performance=performance,
        )
        selected = select_generic_evidence(cv, report, selected_profile)
        tailored = build_generic_cv(cv, selected)
        with measure_stage("rewriting"):
            result = rewrite(
                tailored,
                llm,
                report,
                selected,
                None,
                None,
                target_context=selected_profile.prompt_context,
                rewrite_summary=True,
                merge_similar=True,
            )
        return result, report
