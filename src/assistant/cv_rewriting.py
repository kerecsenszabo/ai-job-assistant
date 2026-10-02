"""Audited CV rewriting with source-cited drafts and independent review."""

from __future__ import annotations

import json
import re

from langchain_core.runnables import Runnable
from pydantic import ValidationError

from assistant.cv_generator import (
    CLIENT_PREFIX,
    CV,
    TailorDiagnostics,
    unsupported_terms,
)
from assistant.cv_tailoring_core import (
    Draft,
    DraftSentence,
    EvidenceItem,
    ModelOutputError,
    Review,
    RewriteAudit,
    SentenceVerdict,
    SummaryDraft,
    TailoringReport,
    request,
)


NUMBERS = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?:%|\+)?")
FIRST_PERSON = re.compile(
    r"\b(?:I|[Mm]e|[Mm]y|[Mm]ine|[Ww]e|[Uu]s|[Oo]ur|[Oo]urs)\b"
)
THIRD_PERSON_CANDIDATE = re.compile(r"\bthe (?:candidate|applicant)\b", re.IGNORECASE)


def mechanical_rejection(sentence: DraftSentence, sources: list[EvidenceItem]) -> str:
    source = "\n".join(item.text + "\n" + item.context for item in sources)
    new_numbers = set(NUMBERS.findall(sentence.text)) - set(NUMBERS.findall(source))
    if new_numbers:
        return "New numerical claims absent from cited evidence: " + ", ".join(
            sorted(new_numbers)
        )
    terms = unsupported_terms(sentence.text, source)
    if terms:
        return "Named terms absent from cited evidence: " + ", ".join(terms)
    if sentence.target_id != "summary":
        prefix = CLIENT_PREFIX.match(sources[0].text)
        if prefix and not sentence.text.startswith(prefix.group()):
            return "The rewrite removed or changed an original context prefix."
    return ""


def summary_voice_rejection(original: str, proposed: str) -> str:
    if FIRST_PERSON.search(original) and (
        not FIRST_PERSON.search(proposed) or THIRD_PERSON_CANDIDATE.search(proposed)
    ):
        return "Summary changed the source's first-person voice."
    return ""


def rewrite_schema(
    targets: list[EvidenceItem], summary_ids: list[str], summary: bool
) -> dict:
    text = {"type": "string", "minLength": 1, "pattern": r"\S"}
    bullets = {}
    for target in targets:
        bullets[target.id] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["source_ids", "text"],
            "properties": {
                "source_ids": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": {"type": "string", "enum": [target.id]},
                },
                "text": text,
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["bullets", "summary_sentences"],
        "properties": {
            "bullets": {
                "type": "object",
                "additionalProperties": False,
                "required": list(bullets),
                "properties": bullets,
            },
            "summary_sentences": {
                "type": "array",
                "maxItems": 4 if summary else 0,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "source_ids", "text"],
                    "properties": {
                        "id": {
                            "type": "string",
                            "enum": [f"summary/{i}" for i in range(4)],
                        },
                        "source_ids": {
                            "type": "array",
                            "minItems": 1,
                            "uniqueItems": True,
                            "items": {"type": "string", "enum": summary_ids},
                        },
                        "text": text,
                    },
                },
            },
        },
    }


def rewrite_review_schema(ids: list[str]) -> dict:
    verdict = {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "reason"],
        "properties": {
            "status": {
                "type": "string",
                "enum": ["supported", "unsupported", "unclear"],
            },
            "reason": {"type": "string", "minLength": 1, "pattern": r"\S"},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdicts"],
        "properties": {
            "verdicts": {
                "type": "object",
                "additionalProperties": False,
                "required": ids,
                "properties": {sid: verdict for sid in ids},
            }
        },
    }


def capture_rewrite_response(raw: list[str], response: str) -> str:
    """Keep the original response and prevent JSON duplicate keys being overwritten."""
    raw.append(response)

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            result[key] = {"duplicate_json_key": key} if key in result else value
        return result

    try:
        return json.dumps(json.loads(response, object_pairs_hook=unique_pairs))
    except ValueError, TypeError:
        return response


def rewrite(
    cv: CV,
    llm: Runnable,
    report: TailoringReport,
    selected: list[EvidenceItem],
    description: str | None,
    diagnostics: TailorDiagnostics | None,
) -> CV:
    originals = {item.id: item for item in report.evidence}
    targets = [item for item in selected if item.section == "experience"]
    data = {
        "job_description": description,
        "bullet_targets": [item.model_dump() for item in targets],
        "summary_evidence": [item.model_dump() for item in selected]
        + [item.model_dump() for item in report.evidence if item.section == "summary"],
    }
    summary_sources = {item["id"] for item in data["summary_evidence"]}
    raw_draft: list[str] = []
    raw_review: list[str] = []
    problems: list[str] = []
    audits: list[RewriteAudit] = []
    candidates: list[DraftSentence] = []
    summaries: list[DraftSentence] = []
    summary_invalid = False
    summary_mechanical_rejection = False

    def record_audit(
        target: str, sources: list[str], proposed: str, status: str, reason: str
    ):
        original = cv.summary if target == "summary" else originals[target].text
        entry = RewriteAudit(
            target_id=target,
            source_ids=sources,
            original=original,
            proposed=proposed,
            exported=proposed if status == "accepted" else original,
            status=status,
            reason=reason,
        )
        audits.append(entry)
        return entry

    try:
        draft = request(
            llm,
            Draft,
            "Rewrite every selected work-experience bullet into clear, concise, "
            "professional CV language, rather than merely selecting, reordering, "
            "or copying the source bullets. Use direct action verbs, remove wordiness, "
            "and improve sentence structure while preserving every factual detail. "
            "Highlight outcomes only when the original bullet states them; never "
            "invent impact or metrics to make a bullet sound stronger. Keep the "
            "original wording only when no safe wording improvement is possible. "
            + (
                "Emphasize facts relevant to the job without adding its requirements "
                "as candidate experience. "
                if description is not None
                else "Use general-purpose wording without targeting a particular job. "
            )
            + "Return bullets as an object keyed by EXACT bullet evidence IDs, with "
            "source_ids containing only that same ID and text. Never combine "
            "different bullets/projects, borrow skills from other evidence, remove "
            "client prefixes, or change ownership, scale, tools, numbers or qualifiers. "
            "Each entry text may contain multiple sentences. "
            + (
                "Also propose up to four concise summary_sentences entries, each "
                "with a unique id from summary/0 through summary/3, text, and "
                "source_ids from summary_evidence. All "
                "claims must be fully supported by those citations; do not infer "
                "leadership, experience duration, expertise or production deployment. "
                "Preserve the original summary voice. If it already fits the role, "
                "make only a subtle, useful change or keep it; otherwise rewrite "
                "it to emphasize supported, relevant strengths. Do not insert job "
                "keywords without citing candidate evidence for each claim. "
                "Never mention the hiring company."
                if description is not None
                else "Return an empty summary_sentences list."
            ),
            data,
            json_schema=rewrite_schema(
                targets,
                sorted({item["id"] for item in data["summary_evidence"]}),
                description is not None,
            ),
            transform_response=lambda response: capture_rewrite_response(
                raw_draft, response
            ),
            stage="rewriting",
        )
    except ModelOutputError as exc:
        report.invalid_rewrite_response = raw_draft[-1] if raw_draft else exc.response
        problems.append(f"Invalid rewrite response: {exc}")
        for item in targets:
            record_audit(item.id, [item.id], "", "unclear", problems[-1])
        if description is not None:
            record_audit(
                "summary",
                ["summary/0"] if cv.summary else [],
                "",
                "unclear",
                problems[-1],
            )
        draft = None

    if draft is not None:
        entries = [
            (
                {
                    **value,
                    "id": target,
                    "target_id": target,
                    **(
                        {"invalid_keyed_fields": True}
                        if set(value) - {"source_ids", "text"}
                        else {}
                    ),
                }
                if isinstance(value, dict)
                else {"target_id": target}
            )
            for target, value in draft.bullets.items()
        ]
        entries += [
            {**value, "target_id": "summary"} if isinstance(value, dict) else {}
            for value in draft.summary_sentences
        ]
        grouped: dict[str, list[object]] = {}
        id_counts: dict[str, int] = {}
        for entry in entries:
            if isinstance(entry, dict):
                target, sid = entry.get("target_id"), entry.get("id")
                if isinstance(target, str):
                    grouped.setdefault(target, []).append(entry)
                else:
                    problems.append("Rewrite entry has no identifiable target.")
                    summary_invalid = True
                if isinstance(sid, str):
                    id_counts[sid] = id_counts.get(sid, 0) + 1
            else:
                problems.append("Malformed rewrite entry has no identifiable target.")
                summary_invalid = True
        allowed = {item.id for item in targets}
        for target in grouped.keys() - allowed - {"summary"}:
            problems.append(f"Unexpected rewrite target: {target}.")
        if len(grouped.get("summary", [])) > 4:
            summary_invalid = True
            problems.append("The proposed summary exceeds four sentences.")

        for target in [item.id for item in targets] + (
            ["summary"] if "summary" in grouped else []
        ):
            values = grouped.get(target, [])
            if target != "summary" and len(values) != 1:
                reason = (
                    "Missing bullet target."
                    if not values
                    else "Duplicate bullet target."
                )
                problems.append(f"{target}: {reason}")
                record_audit(target, [target], "", "unclear", reason)
                continue
            for value in values:
                sentence = None
                try:
                    sentence = DraftSentence.model_validate(value)
                    if id_counts[sentence.id] != 1:
                        raise ValueError("Duplicate sentence ID.")
                    if target == "summary":
                        if (
                            description is None
                            or not set(sentence.source_ids) <= summary_sources
                        ):
                            raise ValueError(
                                "Summary cites evidence outside its allowed sources."
                            )
                        if len(set(sentence.source_ids)) != len(sentence.source_ids):
                            raise ValueError("Duplicate summary source IDs.")
                    elif sentence.source_ids != [target]:
                        raise ValueError(
                            "Bullet rewrite must cite only its original bullet."
                        )
                except (ValidationError, ValueError) as exc:
                    reason = f"Invalid target entry: {exc}"
                    problems.append(f"{target}: {reason}")
                    sources = value.get("source_ids", [])
                    proposed = value.get("text", "")
                    record_audit(
                        target,
                        (
                            sentence.source_ids
                            if sentence
                            else (
                                sources
                                if isinstance(sources, list)
                                and all(isinstance(sid, str) for sid in sources)
                                else []
                            )
                        ),
                        (
                            sentence.text
                            if sentence
                            else (proposed if isinstance(proposed, str) else "")
                        ),
                        "unclear",
                        reason,
                    )
                    summary_invalid |= target == "summary"
                    continue
                if target == "summary":
                    summaries.append(sentence)
                rejection = mechanical_rejection(
                    sentence, [originals[sid] for sid in sentence.source_ids]
                )
                if rejection:
                    record_audit(
                        target,
                        sentence.source_ids,
                        sentence.text,
                        "rejected",
                        rejection,
                    )
                    report.invalid_rewrite_response = (
                        raw_draft[-1] if raw_draft else draft.model_dump_json()
                    )
                    summary_invalid |= target == "summary"
                    summary_mechanical_rejection |= target == "summary"
                elif target != "summary" and sentence.text == originals[target].text:
                    record_audit(
                        target,
                        sentence.source_ids,
                        sentence.text,
                        "accepted",
                        "Unchanged source.",
                    )
                else:
                    candidates.append(sentence)
        if problems:
            report.invalid_rewrite_response = (
                raw_draft[-1] if raw_draft else draft.model_dump_json()
            )

    verdicts: dict[str, SentenceVerdict] = {}
    if candidates:
        review_problem_count = len(problems)
        review_data = [
            {
                "sentence": sentence.model_dump(),
                "sources": [originals[sid].model_dump() for sid in sentence.source_ids],
            }
            for sentence in candidates
        ]
        try:
            review = request(
                llm,
                Review,
                "Independently fact-check EVERY proposed entry against ONLY its cited "
                "sources, including ALL claims in all its sentences. Return verdicts "
                "as an object keyed by the EXACT entry IDs, each with status and reason: "
                "supported only when every assertion is explicitly entailed; "
                "unsupported for invented/altered facts; unclear if uncertain. Scrutinize "
                "ownership (helped vs led), seniority, scope/scale, proficiency, tools, "
                "metrics, years, client identity, team vs individual work, and "
                "prototype vs production. Preserve factual qualifiers and all bullet "
                "details. Reject cross-project or cross-employer fact mixing. A shared "
                "vocabulary is NOT proof of support. Do not trust supplied citations "
                "without reading them. Explain the verdict.",
                {"entries": review_data},
                json_schema=rewrite_review_schema([item.id for item in candidates]),
                transform_response=lambda response: capture_rewrite_response(
                    raw_review, response
                ),
                stage="rewrite_review",
            )
            entries = [
                (
                    {
                        **value,
                        "id": sid,
                        **(
                            {"invalid_keyed_fields": True}
                            if set(value) - {"status", "reason"}
                            else {}
                        ),
                    }
                    if isinstance(value, dict)
                    else {"id": sid}
                )
                for sid, value in review.verdicts.items()
            ]
            grouped_verdicts: dict[str, list[object]] = {}
            for entry in entries:
                if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                    grouped_verdicts.setdefault(entry["id"], []).append(entry)
                else:
                    problems.append(
                        "Malformed review verdict without an identifiable ID."
                    )
            for sentence in candidates:
                values = grouped_verdicts.get(sentence.id, [])
                try:
                    if len(values) != 1:
                        raise ValueError(
                            "Missing review verdict."
                            if not values
                            else "Duplicate review verdict ID."
                        )
                    verdicts[sentence.id] = SentenceVerdict.model_validate(values[0])
                except (ValidationError, ValueError) as exc:
                    verdicts[sentence.id] = SentenceVerdict(
                        id=sentence.id,
                        status="unclear",
                        reason=f"Invalid review: {exc}",
                    )
                    problems.append(f"{sentence.id}: {exc}")
                    report.invalid_review_response = (
                        raw_review[-1] if raw_review else review.model_dump_json()
                    )
            if set(grouped_verdicts) - {item.id for item in candidates}:
                problems.append("Review included unexpected sentence IDs.")
                report.invalid_review_response = (
                    raw_review[-1] if raw_review else review.model_dump_json()
                )
        except ModelOutputError as exc:
            report.invalid_review_response = (
                raw_review[-1] if raw_review else exc.response
            )
            problems.append(f"Rewrite review failed: {exc}")
            verdicts = {
                item.id: SentenceVerdict(id=item.id, status="unclear", reason=str(exc))
                for item in candidates
            }
        if len(problems) > review_problem_count and raw_review:
            report.invalid_review_response = raw_review[-1]
        for sentence in candidates:
            verdict = verdicts[sentence.id]
            status = {
                "supported": "accepted",
                "unsupported": "rejected",
                "unclear": "unclear",
            }[verdict.status]
            record_audit(
                sentence.target_id,
                sentence.source_ids,
                sentence.text,
                status,
                verdict.reason,
            )

    voice_rejection = (
        summary_voice_rejection(cv.summary, " ".join(item.text for item in summaries))
        if summaries
        else ""
    )
    if voice_rejection:
        summary_invalid = True
        for audit in audits:
            if audit.target_id == "summary" and audit.status == "accepted":
                audit.status = "rejected"
                audit.reason = voice_rejection
                audit.exported = cv.summary

    accepted_summary = (
        bool(summaries)
        and not summary_invalid
        and all(
            entry.status == "accepted"
            for entry in audits
            if entry.target_id == "summary"
        )
    )
    if description is not None and summary_mechanical_rejection and summary_sources:
        retry_audits: list[RewriteAudit] = []
        retry_raw: list[str] = []
        retry_review_started = False
        retry_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["summary_sentences"],
            "properties": {
                "summary_sentences": {
                    **rewrite_schema([], sorted(summary_sources), True)["properties"][
                        "summary_sentences"
                    ],
                    "minItems": 1,
                    "maxItems": 1,
                },
            },
        }
        retry_schema["properties"]["summary_sentences"]["items"]["properties"][
            "text"
        ] = {
            "type": "string",
            "minLength": 1,
            "pattern": r"^[^0-9]*\S[^0-9]*$",
        }
        try:
            retry = request(
                llm,
                SummaryDraft,
                "The previous proposed summary was not fully supported or was missing. "
                "Propose ONE concise summary sentence with id summary/0, citing only "
                "the supplied candidate evidence. The job is context, not evidence. "
                "Do not include digits, numerical claims, years of experience, "
                "production use, ownership or uncited tools. "
                "Preserve the original summary's first-person voice. "
                "If the original summary is already strong, make a subtle supported "
                "refinement or retain its wording; otherwise highlight the most "
                "relevant supported strength. Address the prior failure without "
                "copying unsupported claims.",
                {
                    "job_description": description,
                    "original_summary": cv.summary,
                    "summary_evidence": data["summary_evidence"],
                    "previous_failures": [
                        audit.reason
                        for audit in audits
                        if audit.target_id == "summary" and audit.status != "accepted"
                    ],
                },
                json_schema=retry_schema,
                transform_response=lambda response: capture_rewrite_response(
                    retry_raw, response
                ),
                stage="rewriting",
            )
            if len(retry.summary_sentences) != 1:
                raise ModelOutputError("Summary retry must contain one sentence.")
            value = retry.summary_sentences[0]
            sentence = DraftSentence.model_validate(
                {**value, "target_id": "summary"} if isinstance(value, dict) else value
            )
            if (
                sentence.id != "summary/0"
                or not set(sentence.source_ids) <= summary_sources
                or len(sentence.source_ids) != len(set(sentence.source_ids))
            ):
                raise ModelOutputError("Summary retry has invalid evidence citations.")
            rejection = summary_voice_rejection(
                cv.summary, sentence.text
            ) or mechanical_rejection(
                sentence, [originals[sid] for sid in sentence.source_ids]
            )
            if rejection:
                retry_audits.append(
                    record_audit(
                        "summary",
                        sentence.source_ids,
                        sentence.text,
                        "rejected",
                        rejection,
                    )
                )
            elif sentence.text == cv.summary and sentence.source_ids == ["summary/0"]:
                retry_audits.append(
                    record_audit(
                        "summary",
                        sentence.source_ids,
                        sentence.text,
                        "accepted",
                        "Unchanged source.",
                    )
                )
            else:
                retry_review_started = True
                review = request(
                    llm,
                    Review,
                    "Independently check EVERY claim in this summary against ONLY "
                    "its cited candidate evidence. Supported requires every claim "
                    "to be explicitly entailed; reject altered or invented facts.",
                    {
                        "entries": [
                            {
                                "sentence": sentence.model_dump(),
                                "sources": [
                                    originals[sid].model_dump()
                                    for sid in sentence.source_ids
                                ],
                            }
                        ]
                    },
                    json_schema=rewrite_review_schema([sentence.id]),
                    transform_response=lambda response: capture_rewrite_response(
                        raw_review, response
                    ),
                    stage="rewrite_review",
                )
                if set(review.verdicts) != {sentence.id}:
                    raise ModelOutputError(
                        "Summary retry review must cover exactly one sentence."
                    )
                value = review.verdicts[sentence.id]
                if not isinstance(value, dict):
                    raise ModelOutputError(
                        "Summary retry review has an invalid verdict."
                    )
                verdict = SentenceVerdict.model_validate({**value, "id": sentence.id})
                status = {
                    "supported": "accepted",
                    "unsupported": "rejected",
                    "unclear": "unclear",
                }[verdict.status]
                retry_audits.append(
                    record_audit(
                        "summary",
                        sentence.source_ids,
                        sentence.text,
                        status,
                        verdict.reason,
                    )
                )
        except (ModelOutputError, ValidationError) as exc:
            if retry_review_started and raw_review:
                report.invalid_review_response = raw_review[-1]
            elif retry_raw:
                report.invalid_rewrite_response = retry_raw[-1]
            problems.append(f"Summary retry failed: {exc}")
        if retry_audits and retry_audits[0].status == "accepted":
            summaries = [sentence]
            accepted_summary = True
            for audit in audits:
                if (
                    audit not in retry_audits
                    and audit.target_id == "summary"
                    and audit.status == "accepted"
                ):
                    audit.status = "unclear"
                    audit.reason = "Superseded by the accepted summary retry."
                    audit.exported = cv.summary
        elif retry_audits:
            for audit in retry_audits:
                audit.exported = cv.summary
    output_bullets = {
        entry.target_id: entry.exported
        for entry in audits
        if entry.target_id != "summary"
    }
    summary = (
        " ".join(item.text for item in summaries) if accepted_summary else cv.summary
    )
    if description is not None and not accepted_summary:
        report.warnings.append(
            "Kept the source summary: proposed summary was missing or not fully supported."
        )
        for audit in audits:
            if audit.target_id == "summary":
                audit.exported = cv.summary
                if audit.status == "accepted":
                    audit.status = "unclear"
                    audit.reason = "Source summary kept because another summary sentence failed review."
        if not any(entry.target_id == "summary" for entry in audits):
            record_audit(
                "summary",
                ["summary/0"] if cv.summary else [],
                "",
                "unclear",
                "No summary sentences were proposed.",
            )
    report.rewrites.extend(audits)
    report.warnings.extend(problems)
    for audit in audits:
        if audit.status != "accepted":
            report.warnings.append(f"Kept original {audit.target_id}: {audit.reason}")
    experience = [
        role.model_copy(
            update={
                "bullets": [
                    output_bullets.get(item.id, item.text)
                    for item in targets
                    if item.role_index == index
                ]
            }
        )
        for index, role in enumerate(cv.experience)
    ]
    if diagnostics is not None:
        diagnostics.summary_fallback = description is not None and not accepted_summary
    return cv.model_copy(update={"experience": experience, "summary": summary})

