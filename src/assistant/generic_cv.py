"""Generic ML evidence selection, bullet grouping, and page packing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from assistant.cv_generator import CLIENT_PREFIX, CV, PageSettings, to_latex, write_pdf
from assistant.cv_tailoring_core import EvidenceItem, TailoringReport


@dataclass(frozen=True)
class GenericMLProfile:
    """Broad ML positioning used to rank a candidate's existing evidence."""

    name: str = "Generic Machine Learning Engineer"
    prompt_context: str = (
        "a generic machine learning engineer role, emphasizing model development, "
        "evaluation, data and feature pipelines, production inference, MLOps, "
        "Python, SQL, cloud deployment, software quality, reliability, and "
        "evidence-backed LLM or RAG experience"
    )
    max_pages: int = 2
    source_bullet_pool: int = 60
    skill_limit: int = 20
    ai_native_limit: int = 6
    preferred_font_size: int = 11
    preferred_margin_cm: float = 1.6
    minimum_font_size: int = 10
    minimum_margin_cm: float = 1.15


GENERIC_ML_PROFILE = GenericMLProfile()

# A domain vocabulary, not a job specification. It only ranks existing evidence.
PROFILE_TERM_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("machine learning", 6.0),
    ("ml", 4.0),
    ("model", 3.0),
    ("models", 3.0),
    ("forecasting", 3.5),
    ("recommender", 4.0),
    ("inference", 4.0),
    ("experiment", 2.0),
    ("experimentation", 2.0),
    ("evaluation", 2.5),
    ("feature engineering", 3.0),
    ("scikit-learn", 4.0),
    ("xgboost", 4.0),
    ("mlflow", 4.0),
    ("python", 4.0),
    ("sql", 4.0),
    ("pyspark", 4.0),
    ("spark", 3.0),
    ("databricks", 3.0),
    ("dbt", 3.0),
    ("airflow", 3.0),
    ("kafka", 3.0),
    ("data", 1.5),
    ("pipeline", 2.0),
    ("pipelines", 2.0),
    ("api", 2.0),
    ("services", 1.5),
    ("docker", 3.0),
    ("kubernetes", 3.0),
    ("cloud", 3.0),
    ("azure", 2.0),
    ("bentoml", 3.0),
    ("production", 3.0),
    ("deploy", 3.0),
    ("deployment", 3.0),
    ("monitoring", 3.0),
    ("logging", 2.0),
    ("incident", 2.0),
    ("support", 1.5),
    ("performance", 2.0),
    ("reliability", 2.0),
    ("scalable", 2.0),
    ("test", 2.0),
    ("pytest", 3.0),
    ("test-driven", 3.0),
    ("clean-code", 2.0),
    ("code review", 2.0),
    ("architecture", 2.0),
    ("documentation", 1.5),
    ("rag", 4.0),
    ("llm", 4.0),
    ("ai", 2.0),
    ("chatbot", 3.0),
    ("copilot", 2.0),
    ("prompt engineering", 2.0),
    ("embedded", 2.0),
    ("microprocessor", 2.0),
    ("linux", 2.0),
    ("git", 1.0),
)

MERGE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "ai_rag": ("rag", "llm", "chatbot", "hugging face", "chromadb", "prompt"),
    "performance": ("benchmark", "performance", "tune", "tuning", "optimization", "profiling"),
    "quality": ("test", "pytest", "test-driven", "clean-code"),
    "delivery": ("delivered", "presented", "client", "project", "mvp", "agile team"),
    "release": ("model upgrade", "deployment risk", "a/b"),
    "model_lifecycle": (
        "experiment tracking", "mlflow", "lineage", "a/b", "model upgrade",
    ),
    "deployment": (
        "deploy", "deployment", "docker", "kubernetes", "bentoml", "github actions",
        "infrastructure-as-code", "ci/cd",
    ),
    "operations": (
        "production service", "incident", "logging", "monitoring", "retry", "timeout",
        "error handling", "health check", "slo", "reliability", "support",
    ),
    "software_build": (
        "pc applications", "server-based", "data-collection", "data-storage",
        "data-visualization", "c++",
    ),
    "data": (
        "data", "pipeline", "sql", "dbt", "spark", "pyspark", "airflow", "kafka",
        "database", "migration", "transformation", "databricks", "feature engineering",
    ),
    "leadership": (
        "design", "architecture", "mentor", "code owner", "code review", "debt", "release",
    ),
    "stakeholder": (
        "market", "stakeholder", "requirement", "business", "planning", "supply-chain",
        "founder",
    ),
    "model_development": (
        "model", "recommender", "scikit", "xgboost", "inference", "evaluation",
    ),
    "startup": (
        "startups", "founders", "early-stage", "pcb",
        "complete hardware and software solutions",
    ),
    "embedded": ("embedded", "microprocessor", "actuator", "pcb", "hardware"),
    "software": ("api", "service", "backend", "server", "microservice", "python", "scala", "c++"),
}

GROUP_CAPACITY = {
    "ai_rag": 3,
    "data": 3,
    "delivery": 3,
    "performance": 3,
    "software_build": 3,
    "software": 3,
}


def _term_score(text: str, weights: tuple[tuple[str, float], ...]) -> float:
    lowered = " ".join(text.casefold().split())
    score = 0.0
    for term, weight in weights:
        pattern = rf"(?<![\w+#]){re.escape(term)}(?![\w+#])"
        score += weight * min(2, len(re.findall(pattern, lowered)))
    return score


def _evidence_score(item: EvidenceItem) -> float:
    score = _term_score(item.text, PROFILE_TERM_WEIGHTS)
    if item.section == "experience":
        score *= 2.0
        score += _term_score(item.context, PROFILE_TERM_WEIGHTS) * 0.5
    if item.role_index is not None:
        score += max(0.0, 2.5 - item.role_index * 0.35)
    return round(score, 3)


def _targets_for_roles(
    role_count: int, preferred: tuple[int, ...], total: int
) -> dict[int, int]:
    if role_count <= 0:
        return {}
    targets = {
        role: max(1, preferred[role] if role < len(preferred) else 1)
        for role in range(role_count)
    }
    while sum(targets.values()) > total:
        adjustable = [role for role in reversed(sorted(targets)) if targets[role] > 1]
        if not adjustable:
            break
        targets[adjustable[0]] -= 1
    unused = max(0, total - sum(targets.values()))
    for role in sorted(targets):
        targets[role] += unused // role_count
        if role < unused % role_count:
            targets[role] += 1
    return targets


def _choose_role_items(
    items_by_role: dict[int, list[EvidenceItem]],
    scores: dict[str, float],
    *,
    preferred: tuple[int, ...],
    total: int,
) -> list[EvidenceItem]:
    if not items_by_role:
        return []
    available = sum(len(items) for items in items_by_role.values())
    if total >= available:
        all_role_items = [
            item for role in sorted(items_by_role) for item in items_by_role[role]
        ]
        return sorted(
            all_role_items,
            key=lambda item: (item.role_index or 0, item.bullet_index or 0),
        )
    total = min(max(total, len(items_by_role)), available)
    targets = _targets_for_roles(len(items_by_role), preferred, total)
    chosen: set[str] = set()
    for role, role_items in items_by_role.items():
        ranked = sorted(
            role_items,
            key=lambda item: (-scores.get(item.id, 0.0), item.bullet_index or 0),
        )
        chosen.update(item.id for item in ranked[: targets.get(role, 1)])
    remaining = sorted(
        (
            item
            for role_items in items_by_role.values()
            for item in role_items
            if item.id not in chosen
        ),
        key=lambda item: (
            -scores.get(item.id, 0.0),
            item.role_index or 0,
            item.bullet_index or 0,
        ),
    )
    chosen.update(item.id for item in remaining[: max(0, total - len(chosen))])
    all_role_items = [
        item for role in sorted(items_by_role) for item in items_by_role[role]
    ]
    return sorted(
        (item for item in all_role_items if item.id in chosen),
        key=lambda item: (item.role_index or 0, item.bullet_index or 0),
    )


def _merge_category(item: EvidenceItem) -> str:
    lowered = item.text.casefold()
    scores = {
        category: sum(lowered.count(term) for term in terms)
        for category, terms in MERGE_CATEGORIES.items()
    }
    category, score = max(scores.items(), key=lambda pair: pair[1])
    return category if score else "other"


def _can_merge(first: EvidenceItem, second: EvidenceItem) -> bool:
    if first.role_index != second.role_index or _merge_category(first) != _merge_category(second):
        return False
    first_prefix = CLIENT_PREFIX.match(first.text)
    second_prefix = CLIENT_PREFIX.match(second.text)
    if second_prefix and not first_prefix:
        return False
    if first_prefix and second_prefix and first_prefix.group(1) != second_prefix.group(1):
        return False
    return len((first.text + " " + second.text).split()) <= 90


def _group_role_bullets(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Merge tightly related atomic bullets without crossing roles or client contexts."""
    used: set[str] = set()
    groups: list[EvidenceItem] = []
    ordered = sorted(items, key=lambda item: item.bullet_index or 0)
    for index, item in enumerate(ordered):
        if item.id in used:
            continue
        group = [item]
        used.add(item.id)
        category = _merge_category(item)
        capacity = GROUP_CAPACITY.get(category, 2)
        if category != "other":
            for candidate in ordered[index + 1 :]:
                if len(group) >= capacity:
                    break
                if candidate.id in used:
                    continue
                combined_words = len(
                    " ".join(source.text for source in [*group, candidate]).split()
                )
                if combined_words <= 105 and _can_merge(item, candidate):
                    group.append(candidate)
                    used.add(candidate.id)
        context = group[0].context
        groups.append(
            EvidenceItem(
                id=f"experience/{item.role_index}/groups/{len(groups)}",
                section="experience",
                text="\n".join(source.text for source in group),
                role_index=item.role_index,
                bullet_index=len(groups),
                context=context,
                source_ids=[source.id for source in group],
            )
        )
    return groups


def select_generic_evidence(
    cv: CV, report: TailoringReport, profile: GenericMLProfile = GENERIC_ML_PROFILE
) -> list[EvidenceItem]:
    """Select ML-relevant source facts and group related bullets within each role."""
    report.selection_scores = {item.id: _evidence_score(item) for item in report.evidence}
    experience_items = [item for item in report.evidence if item.section == "experience"]
    roles: dict[int, list[EvidenceItem]] = {}
    for item in experience_items:
        if item.role_index is not None:
            roles.setdefault(item.role_index, []).append(item)
    selected_sources = _choose_role_items(
        roles,
        report.selection_scores,
        preferred=(20, 28, 8),
        total=profile.source_bullet_pool,
    )

    selected_text = "\n".join(item.text for item in selected_sources).casefold()
    skill_items = [item for item in report.evidence if item.section == "skills"]
    skill_scores = {
        item.id: _evidence_score(item)
        + (2.0 if item.text.casefold() in selected_text else 0.0)
        for item in skill_items
    }
    selected_skills = sorted(
        skill_items,
        key=lambda item: (-skill_scores[item.id], item.id),
    )[: profile.skill_limit]
    ai_items = [item for item in report.evidence if item.section == "ai_native"]
    selected_ai = sorted(
        ai_items,
        key=lambda item: (-report.selection_scores[item.id], item.id),
    )[: profile.ai_native_limit]

    grouped_experience: list[EvidenceItem] = []
    for role_index in sorted(roles):
        role_selected = [item for item in selected_sources if item.role_index == role_index]
        grouped_experience.extend(_group_role_bullets(role_selected))
    for group in grouped_experience:
        report.selection_scores[group.id] = round(
            sum(report.selection_scores[item_id] for item_id in group.source_ids), 3
        )

    report.evidence.extend(grouped_experience)
    selected = [*selected_skills, *selected_ai, *grouped_experience]
    report.selected_evidence_ids = [item.id for item in selected]
    report.exported_evidence_ids = report.selected_evidence_ids.copy()
    covered = {
        item_id
        for item in selected
        for item_id in (item.source_ids or [item.id])
    }
    report.omitted_evidence_ids = [
        item.id
        for item in experience_items + skill_items + ai_items
        if item.id not in covered
    ]
    return selected


def build_generic_cv(cv: CV, selected: list[EvidenceItem]) -> CV:
    """Apply selected and grouped facts without changing their source wording."""
    return cv.model_copy(
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
                            if item.section == "experience"
                            and item.role_index == index
                        ]
                    }
                )
                for index, role in enumerate(cv.experience)
            ],
        }
    )


def _targets_for_group_pool(
    report: TailoringReport, preferred: tuple[int, ...], total: int
) -> dict[int, list[str]]:
    grouped: dict[int, list[EvidenceItem]] = {}
    selected = set(report.selected_evidence_ids)
    for item in report.evidence:
        if (
            item.section == "experience"
            and item.id in selected
            and item.role_index is not None
        ):
            grouped.setdefault(item.role_index, []).append(item)
    chosen = _choose_role_items(
        grouped,
        report.selection_scores,
        preferred=preferred,
        total=total,
    )
    result: dict[int, list[str]] = {}
    for item in chosen:
        result.setdefault(item.role_index or 0, []).append(item.id)
    return result


def _packed_cv(
    cv: CV,
    report: TailoringReport,
    *,
    experience_targets: tuple[int, ...],
    experience_total: int,
    skills_limit: int,
    ai_native_limit: int,
) -> CV:
    selected = set(report.selected_evidence_ids)
    selected_groups = _targets_for_group_pool(
        report,
        preferred=experience_targets,
        total=experience_total,
    )
    evidence_by_id = {item.id: item for item in report.evidence}
    grouped_by_role = {
        role: [
            item_id
            for item_id in report.selected_evidence_ids
            if evidence_by_id[item_id].section == "experience"
            and evidence_by_id[item_id].role_index == role
        ]
        for role in range(len(cv.experience))
    }

    experience = []
    for role_index, role in enumerate(cv.experience):
        group_ids = grouped_by_role.get(role_index, [])
        kept_ids = selected_groups.get(role_index, [])
        if len(group_ids) != len(role.bullets):
            raise ValueError("Generic CV export no longer matches grouped evidence.")
        experience.append(
            role.model_copy(
                update={
                    "bullets": [
                        text for group_id, text in zip(group_ids, role.bullets)
                        if group_id in kept_ids
                    ]
                }
            )
        )

    skill_ids = [
        item_id
        for item_id in report.selected_evidence_ids
        if evidence_by_id[item_id].section == "skills"
    ]
    ai_ids = [
        item_id
        for item_id in report.selected_evidence_ids
        if evidence_by_id[item_id].section == "ai_native"
    ]
    if len(skill_ids) != len(cv.skills) or len(ai_ids) != len(cv.ai_native):
        raise ValueError("Generic CV export no longer matches selected skills.")

    packed = cv.model_copy(
        update={
            "skills": cv.skills[:skills_limit],
            "ai_native": cv.ai_native[:ai_native_limit],
            "experience": experience,
        }
    )
    kept_group_ids = {item_id for ids in selected_groups.values() for item_id in ids}
    exported = skill_ids[:skills_limit] + ai_ids[:ai_native_limit] + list(kept_group_ids)
    exported_set = set(exported)
    report.exported_evidence_ids = [
        item_id for item_id in report.selected_evidence_ids if item_id in exported_set
    ]
    covered = {
        source_id
        for item_id in exported_set
        for source_id in (evidence_by_id[item_id].source_ids or [item_id])
    }
    selectable = {"skills", "ai_native", "experience"}
    report.omitted_evidence_ids = [
        item.id
        for item in report.evidence
        if item.section in selectable and item.id not in covered and item.id not in exported_set
    ]
    return packed


def pdf_page_count(path: Path) -> int:
    with pdfplumber.open(path) as pdf:
        return len(pdf.pages)


def fit_generic_cv_to_pages(
    cv: CV,
    report: TailoringReport,
    output: Path,
    page_settings: PageSettings,
    *,
    profile: GenericMLProfile = GENERIC_ML_PROFILE,
    allow_layout_fallback: bool = True,
) -> tuple[CV, PageSettings, int | None]:
    """Render and trim already-reviewed grouped evidence until it fits."""
    report.requested_page_limit = profile.max_pages
    stages = (
        ((14, 14, 8), 36, 20, 6, "full grouped evidence budget"),
        ((12, 12, 8), 32, 20, 6, "expanded grouped evidence budget"),
        ((11, 11, 8), 30, 20, 6, "upper grouped evidence budget"),
        ((10, 10, 8), 28, 20, 6, "reduced grouped evidence budget"),
        ((9, 9, 8), 26, 20, 6, "focused grouped evidence budget"),
        ((8, 8, 8), 24, 18, 6, "compact grouped evidence budget"),
        ((8, 8, 6), 22, 20, 6, "tight grouped evidence budget"),
        ((8, 8, 6), 22, 16, 5, "core grouped evidence budget"),
        ((6, 6, 4), 16, 14, 4, "minimum grouped evidence budget"),
    )

    settings = page_settings
    for stage_index, (
        experience_targets,
        experience_total,
        skills_limit,
        ai_native_limit,
        label,
    ) in enumerate(stages):
        candidate = _packed_cv(
            cv,
            report,
            experience_targets=experience_targets,
            experience_total=experience_total,
            skills_limit=skills_limit,
            ai_native_limit=ai_native_limit,
        )
        if stage_index > 0:
            report.layout_fallbacks.append(label)
        write_pdf(to_latex(candidate, page_settings=settings), output)
        try:
            pages = pdf_page_count(output)
        except Exception as exc:
            report.warnings.append(
                f"Could not measure the generic CV page count: {exc}. "
                "The two-page limit was not verified."
            )
            report.final_page_count = None
            return candidate, settings, None
        report.final_page_count = pages
        if pages <= profile.max_pages:
            return candidate, settings, pages

    if allow_layout_fallback:
        candidate = _packed_cv(
            cv,
            report,
            experience_targets=(6, 6, 4),
            experience_total=16,
            skills_limit=12,
            ai_native_limit=3,
        )
        fallback_settings = (
            PageSettings(
                settings.paper_size,
                11,
                min(settings.margin_cm, profile.preferred_margin_cm),
            ),
            PageSettings(
                settings.paper_size,
                profile.minimum_font_size,
                min(settings.margin_cm, profile.minimum_margin_cm),
            ),
        )
        for settings in fallback_settings:
            if settings.font_size > page_settings.font_size:
                continue
            write_pdf(to_latex(candidate, page_settings=settings), output)
            try:
                pages = pdf_page_count(output)
            except Exception as exc:
                report.warnings.append(
                    f"Could not measure the generic CV page count: {exc}. "
                    "The two-page limit was not verified."
                )
                report.final_page_count = None
                return candidate, settings, None
            report.final_page_count = pages
            report.layout_fallbacks.append(
                f"layout fallback: {settings.font_size}pt, "
                f"{settings.margin_cm:g}cm margins"
            )
            if pages <= profile.max_pages:
                return candidate, settings, pages

    raise ValueError(
        "The generic ML CV still exceeds two pages after removing optional evidence. "
        "Review the source CV or use an explicit smaller font and margin."
    )
