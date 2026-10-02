"""Evidence-first job matching, validation, scoring and selection."""

from __future__ import annotations

import json
import re

from langchain_core.runnables import Runnable
from assistant.cv_generator import (
    CLIENT_PREFIX,
    CONTEXT_BULLETS,
    MAX_AI_NATIVE,
    MAX_BULLETS,
    MAX_SKILLS,
    TERM,
)
from assistant.cv_tailoring_core import (
    CriterionMatch,
    EvidenceItem,
    Matches,
    ModelOutputError,
    ParsedJob,
    Requirement,
    RequirementCriterion,
    RequirementMatch,
    ScoringRubric,
    SemanticDecisions,
    TailoringReport,
    mentions,
    normalized,
    request,
)


def checked_matches(
    matches: Matches, requirements: list[Requirement], evidence: list[EvidenceItem]
) -> list[RequirementMatch]:
    expected = {item.id for item in requirements}
    ids = [item.requirement_id for item in matches.matches]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ModelOutputError("Matching must cover every requirement exactly once.")
    available = {item.id for item in evidence}
    for match in matches.matches:
        if len(match.evidence_ids) != len(set(match.evidence_ids)):
            raise ModelOutputError("A requirement match repeats evidence IDs.")
        if (
            not (set(match.evidence_ids) | set(match.inspected_evidence_ids))
            <= available
        ):
            raise ModelOutputError("A requirement match cites nonexistent CV evidence.")
        if match.status in ("direct", "partial") and not match.evidence_ids:
            raise ModelOutputError(
                "A positive requirement match has no source evidence."
            )
        requirement = next(
            item for item in requirements if item.id == match.requirement_id
        )
        criterion_ids = [item.criterion_id for item in match.criteria_matches]
        if len(criterion_ids) != len(set(criterion_ids)) or set(criterion_ids) != {
            item.id for item in requirement.criteria
        }:
            raise ModelOutputError("Matching must cover every criterion exactly once.")
        for criterion_match in match.criteria_matches:
            if (
                len(criterion_match.evidence_ids)
                != len(set(criterion_match.evidence_ids))
                or not set(criterion_match.evidence_ids) <= available
            ):
                raise ModelOutputError("A criterion match cites invalid CV evidence.")
            if (
                criterion_match.status in ("direct", "partial")
                and not criterion_match.evidence_ids
            ):
                raise ModelOutputError(
                    "A positive criterion match has no source evidence."
                )
    by_id = {}
    for match in matches.matches:
        # Models sometimes cite inspected items when explaining a gap. These
        # references must never be treated as positive matching evidence.
        if match.status == "not_evidenced" and match.evidence_ids:
            match = match.model_copy(
                update={
                    "inspected_evidence_ids": list(
                        dict.fromkeys(
                            [*match.inspected_evidence_ids, *match.evidence_ids]
                        )
                    ),
                    "evidence_ids": [],
                }
            )
        by_id[match.requirement_id] = match
    return [by_id[item.id] for item in requirements]


NEGATIVE_EVIDENCE = re.compile(
    r"\b(?:no experience|not (?:used|using|worked)|never (?:used|worked)|"
    r"without experience|lack(?:ing)? (?:experience|knowledge)|unfamiliar)\b"
)
NON_PRODUCTION = re.compile(
    r"\b(?:non-production|not (?:in )?production|prototype only)\b"
)
LANGUAGE_PROFICIENCY = re.compile(
    r"\b(?:proficien(?:t|cy)|fluen(?:t|cy)|native|c[12]|professional working|"
    r"full professional)\b"
)
LIMITED_PROFICIENCY = re.compile(
    r"\b(?:basic|beginner|elementary|limited|conversational|intermediate)\b"
)


def language_proficiency_supported(clause: str, language: str) -> bool:
    text = normalized(clause)
    if LIMITED_PROFICIENCY.search(text):
        return False
    level = LANGUAGE_PROFICIENCY.pattern.removeprefix(r"\b").removesuffix(r"\b")
    name = re.escape(normalized(language))
    return bool(
        re.search(r"\b" + level + r"\s+(?:in\s+)?" + name + r"\b", text)
        or re.search(
            r"\b"
            + name
            + r"\s*(?:[:(\-]\s*|is\s+|(?:language )?proficiency (?:at )?)?"
            + level
            + r"\b",
            text,
        )
    )


def explicit_criterion_match(
    criterion: RequirementCriterion,
    evidence: list[EvidenceItem],
) -> CriterionMatch:
    """Assess named options from actual CV text, never from model explanations."""
    option_matches = []
    for option in criterion.options:
        direct_ids = []
        partial_ids = []
        for item in evidence:
            if item.section == "role":
                continue
            for clause in re.split(r"[.;\n]", item.text):
                if not mentions(clause, option) or NEGATIVE_EVIDENCE.search(
                    normalized(clause)
                ):
                    continue
                if criterion.kind == "language":
                    direct = (
                        not criterion.proficiency
                        or language_proficiency_supported(clause, option)
                    )
                elif criterion.production:
                    direct = (
                        item.section == "experience"
                        and mentions(clause, "production")
                        and not NON_PRODUCTION.search(normalized(clause))
                    )
                else:
                    direct = item.section in ("experience", "ai_native") or not (
                        criterion.experience_required
                        or re.search(r"\bexperience\b", normalized(criterion.quote))
                    )
                (direct_ids if direct else partial_ids).append(item.id)
        option_matches.append(
            (
                option,
                (
                    "direct"
                    if direct_ids
                    else "partial" if partial_ids else "not_evidenced"
                ),
                list(dict.fromkeys(direct_ids or partial_ids)),
            )
        )
    direct = [entry for entry in option_matches if entry[1] == "direct"]
    supported = [entry for entry in option_matches if entry[1] != "not_evidenced"]
    if criterion.operator == "any":
        chosen = direct or supported
        status = "direct" if direct else "partial" if supported else "not_evidenced"
    else:
        chosen = supported
        status = (
            "direct"
            if len(direct) == len(option_matches)
            else "partial" if supported else "not_evidenced"
        )
    source_ids = list(
        dict.fromkeys(source_id for _, _, ids in chosen for source_id in ids)
    )
    if not supported:
        explanation = (
            "No explicit CV evidence for: " + ", ".join(criterion.options) + "."
        )
    else:
        explanation = (
            f"{criterion.operator}-of options: "
            + "; ".join(f"{option}: {state}" for option, state, _ in option_matches)
            + "."
        )
    if criterion.production:
        explanation += (
            " Direct credit requires explicit production use in work experience."
        )
    if criterion.kind == "language":
        explanation += (
            " Communication or stakeholder work does not prove language proficiency."
        )
    return CriterionMatch(
        criterion_id=criterion.id,
        status=status,
        evidence_ids=source_ids,
        explanation=explanation,
    )


def enforce_match_rules(
    requirements: list[Requirement],
    matches: list[RequirementMatch],
    evidence: list[EvidenceItem],
) -> list[RequirementMatch]:
    result = []
    for requirement, match in zip(requirements, matches):
        if not requirement.criteria:
            result.append(match)
            continue
        by_id = {item.criterion_id: item for item in match.criteria_matches}
        corrected = []
        adjustments = []
        missing_language = False
        for criterion in requirement.criteria:
            proposed = by_id[criterion.id]
            decision = (
                explicit_criterion_match(criterion, evidence)
                if criterion.kind in ("technology", "language")
                else guard_semantic_decision(criterion, proposed, evidence)
            )
            if decision.status != proposed.status:
                adjustments.append(
                    f"{criterion.id}: {proposed.status} -> {decision.status}; "
                    + decision.explanation
                )
            if criterion.kind == "language" and decision.status == "not_evidenced":
                missing_language = True
            corrected.append(decision)
        statuses = [item.status for item in corrected]
        if missing_language:
            status = "not_evidenced"
        elif all(status == "direct" for status in statuses):
            status = "direct"
        elif any(status in ("direct", "partial") for status in statuses):
            status = "partial"
        elif "unclear" in statuses:
            status = "unclear"
        else:
            status = "not_evidenced"
        source_ids = list(
            dict.fromkeys(
                source_id
                for decision in corrected
                if decision.status in ("direct", "partial")
                for source_id in decision.evidence_ids
            )
        )
        if status != match.status:
            adjustments.append(
                f"Overall match: {match.status} -> {status} from explicit criterion coverage."
            )
        result.append(
            match.model_copy(
                update={
                    "status": status,
                    "evidence_ids": (
                        source_ids if status in ("direct", "partial") else []
                    ),
                    "criteria_matches": corrected,
                    "rule_adjustments": adjustments,
                    "explanation": " ".join(
                        f"{criterion.text}: {decision.status}. {decision.explanation}"
                        for criterion, decision in zip(requirement.criteria, corrected)
                    ),
                }
            )
        )
    return result


RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "or",
    "the",
    "with",
    "in",
    "of",
    "for",
    "to",
    "such",
    "as",
    "including",
    "experience",
    "strong",
    "skill",
    "knowledge",
}
RETRIEVAL_RELATED = {
    "communication": {"stakeholder", "collaboration", "collaborate", "partner"},
    "testing": {"test", "tested", "benchmark", "benchmarked"},
    "ci": {"release", "github", "pipeline", "deployment"},
    "cd": {"release", "github", "pipeline", "deployment"},
    "agentic": {"langchain", "langgraph", "rag", "llm", "orchestration"},
    "orchestration": {"workflow", "pipeline", "langchain", "langgraph"},
}


def retrieval_terms(text: str) -> set[str]:
    return {
        match.group().casefold().removesuffix("s") for match in TERM.finditer(text)
    } - RETRIEVAL_STOPWORDS


def retrieve_evidence(
    criterion: RequirementCriterion,
    evidence: list[EvidenceItem],
    related_ids: set[str],
) -> list[EvidenceItem]:
    terms = retrieval_terms(criterion.text)
    for term in tuple(terms):
        terms.update(RETRIEVAL_RELATED.get(term, set()))
    ranked = sorted(
        evidence,
        key=lambda item: -(
            len(terms & retrieval_terms(item.text))
            + (0.25 if item.id in related_ids else 0)
        ),
    )
    selected = [
        item
        for item in ranked
        if terms & retrieval_terms(item.text) or item.id in related_ids
    ][:6]
    if not selected:
        selected = [item for item in evidence if item.section == "summary"][:1]
    if re.search(r"\byears?\b", normalized(criterion.text)):
        selected.extend(item for item in evidence if item.section == "role")
    return list({item.id: item for item in selected}.values())


def unkey_semantic_decisions(
    response: str,
    criterion_aliases: dict[str, str] | None = None,
    evidence_aliases: dict[str, str] | None = None,
) -> str:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(
            "Semantic matching returned invalid JSON.", response
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), dict):
        return response
    for key, decision in payload["decisions"].items():
        criterion_id = (criterion_aliases or {}).get(key, key)
        if not isinstance(decision, dict) or decision.get(
            "criterion_id", criterion_id
        ) not in (key, criterion_id):
            raise ModelOutputError(
                "A semantic decision has an inconsistent ID.", response
            )
        decision["criterion_id"] = criterion_id
        if isinstance(decision.get("evidence_ids"), list):
            if not all(
                isinstance(source_id, str) for source_id in decision["evidence_ids"]
            ):
                raise ModelOutputError(
                    "Semantic evidence IDs must be strings.", response
                )
            decision["evidence_ids"] = [
                (evidence_aliases or {}).get(source_id, source_id)
                for source_id in decision["evidence_ids"]
            ]
        if isinstance(decision.get("explanation"), str):
            aliases = {**(criterion_aliases or {}), **(evidence_aliases or {})}
            decision["explanation"] = re.sub(
                r"\b(?:c|e)\d+\b",
                lambda match: aliases.get(match.group(), match.group()),
                decision["explanation"],
            )
    payload["decisions"] = list(payload["decisions"].values())
    return json.dumps(payload)


def semantic_decisions(
    llm: Runnable,
    tasks: list[dict],
    candidates: dict[str, list[EvidenceItem]],
) -> list[CriterionMatch]:
    criterion_aliases = {f"c{index}": task["id"] for index, task in enumerate(tasks)}
    originals = {item.id: item for items in candidates.values() for item in items}
    evidence_aliases = {
        f"e{index}": source_id for index, source_id in enumerate(originals)
    }
    source_aliases = {source_id: alias for alias, source_id in evidence_aliases.items()}
    keyed = {}
    compact_tasks = []
    for alias, task in zip(criterion_aliases, tasks):
        criterion_id = task["id"]
        schema = CriterionMatch.model_json_schema()
        schema["properties"].pop("criterion_id")
        schema["required"].remove("criterion_id")
        ids = [source_aliases[item.id] for item in candidates[criterion_id]]
        if ids:
            schema["properties"]["evidence_ids"]["items"]["enum"] = ids
            schema["properties"]["evidence_ids"]["minItems"] = 1
        else:
            schema["properties"]["evidence_ids"]["maxItems"] = 0
            schema["properties"]["status"]["enum"] = ["not_evidenced", "unclear"]
        schema["properties"]["evidence_ids"]["maxItems"] = min(4, len(ids))
        schema["properties"]["explanation"]["maxLength"] = 160
        keyed[alias] = schema
        compact = {"id": alias, "text": task["text"], "candidate_evidence_ids": ids}
        if normalized(task["job_quote"]) != normalized(task["text"]):
            compact["job_quote"] = task["job_quote"]
        compact_tasks.append(compact)
    schema = {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "object",
                "properties": keyed,
                "required": list(keyed),
                "additionalProperties": False,
            }
        },
        "required": ["decisions"],
        "additionalProperties": False,
    }
    evidence = {
        alias: {
            "text": originals[source_id].text,
            **(
                {"context": originals[source_id].context}
                if originals[source_id].context
                else {}
            ),
        }
        for alias, source_id in evidence_aliases.items()
    }
    data = {"criteria": compact_tasks, "evidence": evidence}
    instruction = (
        "Assess ONLY these unresolved semantic criteria against their listed "
        "candidate evidence IDs. Explicit technologies/languages are assessed "
        "locally; do not reassess them. Direct means every clause is explicitly "
        "supported; partial means related/incomplete evidence; not_evidenced "
        "means no support in retrieved evidence; unclear means ambiguous. "
        "Positive decisions require supporting IDs. When candidates exist, "
        "negative decisions cite at least one inspected candidate; these "
        "references will not count as supporting evidence. "
        "Never infer ownership, production features or tools from general work. "
        "RAG alone does not establish agentic or autonomous workflows. Releases "
        "alone do not establish CI/CD. Role dates support role tenure, not years "
        "using a specific tool. Never treat the job quote as candidate evidence. "
        "Return decisions keyed by the short criterion ID (c0, c1, ...). "
        "Cite only listed short evidence IDs (e0, e1, ...). No duplicate IDs. "
        "Use concise explanations of at most 160 characters."
    )

    def assess(stage: str, review: bool) -> list[CriterionMatch]:
        prompt = (
            "Independently verify and correct proposed decisions. " if review else ""
        ) + instruction
        for attempt in range(2):
            try:
                response = request(
                    llm,
                    SemanticDecisions,
                    prompt,
                    data,
                    json_schema=schema,
                    transform_response=lambda reply: unkey_semantic_decisions(
                        reply, criterion_aliases, evidence_aliases
                    ),
                    stage=stage,
                )
                ids = [item.criterion_id for item in response.decisions]
                if len(set(ids)) != len(ids) or set(ids) != set(
                    criterion_aliases.values()
                ):
                    raise ModelOutputError(
                        "Semantic decisions must cover every criterion once."
                    )
                corrected = []
                for decision in response.decisions:
                    allowed = {item.id for item in candidates[decision.criterion_id]}
                    if not set(decision.evidence_ids) <= allowed or (
                        decision.status in ("direct", "partial")
                        and not decision.evidence_ids
                    ):
                        raise ModelOutputError(
                            f"Semantic decision {decision.criterion_id} ({decision.status}) "
                            f"has invalid source citations: {decision.evidence_ids}."
                        )
                    ids = list(dict.fromkeys(decision.evidence_ids))
                    explanation = decision.explanation
                    if len(ids) != len(decision.evidence_ids):
                        explanation += " Repeated citations were deduplicated."
                    if decision.status not in ("direct", "partial"):
                        ids = []
                    corrected.append(
                        decision.model_copy(
                            update={
                                "evidence_ids": ids,
                                "explanation": explanation,
                            }
                        )
                    )
                return corrected
            except ModelOutputError as exc:
                if attempt == 1:
                    raise
                prompt += f"\nPrevious reply was invalid: {exc} Correct it."
        raise AssertionError("Unreachable semantic matching attempt.")

    proposed = assess("matching", False)
    by_id = {item.criterion_id: item for item in proposed}
    data["proposed_decisions"] = {
        alias: {
            "status": by_id[criterion_id].status,
            "evidence_ids": [
                source_aliases[source_id]
                for source_id in by_id[criterion_id].evidence_ids
            ],
            "explanation": by_id[criterion_id].explanation,
        }
        for alias, criterion_id in criterion_aliases.items()
    }
    return assess("matching_review", True)


SEMANTIC_ANCHORS = (
    (r"\bagentic\b", "agentic capabilities", r"\b(?:agentic|agents?|autonomous)\b"),
    (
        r"\bci/cd\b",
        "CI/CD",
        r"\b(?:ci/cd|continuous integration|continuous (?:delivery|deployment))\b",
    ),
    (r"\bdebugging\b", "debugging", r"\b(?:debug(?:ged|ging)?|troubleshoot(?:ing)?)\b"),
)


def guard_semantic_decision(
    criterion: RequirementCriterion,
    decision: CriterionMatch,
    evidence: list[EvidenceItem],
) -> CriterionMatch:
    if decision.status != "direct":
        return decision
    quoted = normalized(criterion.text + " " + criterion.quote)
    cited = normalized(
        " ".join(item.text for item in evidence if item.id in decision.evidence_ids)
    )
    missing = [
        label
        for required, label, supported in SEMANTIC_ANCHORS
        if re.search(required, quoted) and not re.search(supported, cited)
    ]
    if not missing:
        return decision
    return decision.model_copy(
        update={
            "status": "partial",
            "explanation": decision.explanation
            + " Direct credit withheld: explicit "
            + ", ".join(missing)
            + " evidence is absent from the cited CV statements.",
        }
    )


def match_job(
    llm: Runnable, job: ParsedJob, evidence: list[EvidenceItem]
) -> list[RequirementMatch]:
    known: dict[str, CriterionMatch] = {}
    tasks = []
    candidates = {}
    for requirement in job.requirements:
        if not requirement.criteria:
            raise ModelOutputError(
                f"{requirement.id} has no criteria; reparse the job with --refresh-job-analysis."
            )
        explicit = [
            explicit_criterion_match(criterion, evidence)
            for criterion in requirement.criteria
            if criterion.kind != "general"
        ]
        known.update((decision.criterion_id, decision) for decision in explicit)
        related_ids = {
            source_id for decision in explicit for source_id in decision.evidence_ids
        }
        for criterion in requirement.criteria:
            if criterion.kind == "general":
                selected = retrieve_evidence(criterion, evidence, related_ids)
                candidates[criterion.id] = selected
                tasks.append(
                    {
                        "id": criterion.id,
                        "text": criterion.text,
                        "job_quote": criterion.quote,
                        "candidate_evidence_ids": [item.id for item in selected],
                    }
                )
    if tasks:
        # Bound the compact semantic workload, rather than repeatedly assessing
        # already-resolved tools/languages against the complete CV.
        for offset in range(0, len(tasks), 12):
            batch = tasks[offset : offset + 12]
            ids = {item["id"] for item in batch}
            known.update(
                (decision.criterion_id, decision)
                for decision in semantic_decisions(
                    llm,
                    batch,
                    {key: value for key, value in candidates.items() if key in ids},
                )
            )
    combined = [
        RequirementMatch(
            requirement_id=requirement.id,
            status="unclear",
            evidence_ids=[],
            explanation="Aggregating explicit and reviewed semantic criteria.",
            criteria_matches=[known[item.id] for item in requirement.criteria],
        )
        for requirement in job.requirements
    ]
    return enforce_match_rules(job.requirements, combined, evidence)


def score_matches(
    requirements: list[Requirement],
    matches: list[RequirementMatch],
    rubric: ScoringRubric,
) -> tuple[float | None, float | None]:
    by_id = {item.requirement_id: item for item in matches}

    def score(items: list[Requirement]) -> float | None:
        if not items:
            return None
        credits = {
            "direct": 1,
            "partial": rubric.partial_credit,
            "not_evidenced": 0,
            "unclear": 0,
        }
        numerator = sum(
            rubric.weight(item) * credits[by_id[item.id].status] for item in items
        )
        return round(100 * numerator / sum(rubric.weight(item) for item in items), 1)

    return score(requirements), score(
        [
            item
            for item in requirements
            if item.importance in ("required", "eligibility")
        ]
    )


def select_evidence(report: TailoringReport) -> list[EvidenceItem]:
    report.contextual_evidence_ids = []
    relevance: dict[str, float] = {}
    named_skills: set[str] = set()
    requirements = {item.id: item for item in report.requirements}
    for match in report.matches:
        if match.status not in ("direct", "partial"):
            continue
        credit = 1 if match.status == "direct" else report.rubric.partial_credit
        for source_id in match.evidence_ids:
            relevance[source_id] = relevance.get(source_id, 0) + (
                report.rubric.weight(requirements[match.requirement_id]) * credit
            )
        requirement = requirements[match.requirement_id]
        for item in report.evidence:
            if item.section == "skills" and mentions(
                requirement.quote.replace("-", " "),
                item.text.replace("-", " "),
            ):
                named_skills.add(item.id)
                relevance[item.id] = relevance.get(item.id, 0) + (
                    report.rubric.weight(requirement) * credit
                )
    selected = []
    limits = {"skills": MAX_SKILLS, "ai_native": MAX_AI_NATIVE}
    for section, limit in limits.items():
        items = [
            item
            for item in report.evidence
            if item.section == section and relevance.get(item.id, 0) > 0
        ]
        selected.extend(
            sorted(
                items,
                key=lambda item: (item.id not in named_skills, -relevance[item.id]),
            )[:limit]
        )
    roles = sorted(
        {item.role_index for item in report.evidence if item.section == "experience"}
    )
    for role_index in roles:
        items = [
            item
            for item in report.evidence
            if item.section == "experience" and item.role_index == role_index
        ]
        relevant = [item for item in items if relevance.get(item.id, 0) > 0]
        if relevant:
            anchors = {}
            anchor = None
            for item in items:
                if CLIENT_PREFIX.match(item.text):
                    anchor = item
                elif anchor is not None:
                    anchors[item.id] = anchor
            chosen: dict[str, EvidenceItem] = {}
            ranked = sorted(relevant, key=lambda item: -relevance[item.id])
            ranked.extend(item for item in items if relevance.get(item.id, 0) == 0)
            for item in ranked:
                required = [item]
                if item.id in anchors:
                    required.append(anchors[item.id])
                additions = [entry for entry in required if entry.id not in chosen]
                if len(chosen) + len(additions) <= MAX_BULLETS:
                    chosen.update((entry.id, entry) for entry in additions)
            ordered = [item for item in items if item.id in chosen]
            selected.extend(ordered)
            report.contextual_evidence_ids.extend(
                item.id for item in ordered if relevance.get(item.id, 0) == 0
            )
        else:
            # Keep chronology visible without manufacturing relevance.
            context = items[:CONTEXT_BULLETS]
            selected.extend(context)
            report.contextual_evidence_ids.extend(item.id for item in context)
    report.selected_evidence_ids = [item.id for item in selected]
    return selected
