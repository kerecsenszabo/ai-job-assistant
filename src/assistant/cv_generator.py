"""Render a structured CV as LaTeX/PDF, optionally tailored to a job."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from assistant.cv_tailoring import TailoringReport
    from assistant.performance import RunPerformance

MODEL = "granite4.2:3b"
# Source CV + job description + full JSON reply exceeds Ollama's default window.
CONTEXT_TOKENS = 16384
# Models that answer with an empty string when reasoning is switched off.
REASONING_REQUIRED = ("gpt-oss",)
MAX_SKILLS = 10
MAX_AI_NATIVE = 3
MAX_BULLETS = 4


@dataclass
class TailorDiagnostics:
    """Operational measurements from one tailoring run."""

    selection_seconds: float = 0.0
    summary_seconds: float = 0.0
    summary_attempts: int = 0
    summary_fallback: bool = False
    report: TailoringReport | None = None
    performance: RunPerformance | None = None


def local_llm(model: str) -> ChatOllama:
    """Local LLM with room for a full CV round-trip.

    Uses the chat endpoint, and leaves reasoning at the model default for
    harmony-style models: ``gpt-oss`` returns an empty reply to both
    completion requests and requests that disable thinking.
    """
    return ChatOllama(
        model=model,
        reasoning=None if model.startswith(REASONING_REQUIRED) else False,
        temperature=0,
        num_ctx=CONTEXT_TOKENS,
    )


def structured_llm(llm: Runnable, schema: type[BaseModel]) -> Runnable:
    """Constrain an Ollama request to the JSON schema for *schema*."""
    return llm.bind(format=schema.model_json_schema())


class _InstalledModel(BaseModel):
    name: str
    digest: str


class _InstalledModels(BaseModel):
    models: list[_InstalledModel]


def local_model_identity(llm: ChatOllama) -> str:
    """Include installed weights and inference settings in matching-cache keys."""
    base_url = llm.base_url or "http://localhost:11434"
    response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
    response.raise_for_status()
    installed = _InstalledModels.model_validate_json(response.text)
    name = llm.model if ":" in llm.model else f"{llm.model}:latest"
    matched = next((item for item in installed.models if item.name == name), None)
    if matched is None:
        raise ValueError(f"Cannot identify installed model {name} for matching cache.")
    return json.dumps({
        "model": name, "digest": matched.digest, "temperature": llm.temperature,
        "num_ctx": llm.num_ctx, "reasoning": llm.reasoning,
    }, sort_keys=True)


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    role: str
    dates: str
    bullets: list[str] = Field(min_length=1)


class Education(BaseModel):
    model_config = ConfigDict(extra="forbid")

    institution: str
    degree: str
    dates: str = ""


class Publication(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    authors: str = ""
    date: str = ""
    publisher: str = ""
    doi: str = ""
    url: str = ""


class Certification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    issuer: str = ""
    date: str = ""
    url: str = ""


class Language(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, pattern=r"\S")
    proficiency: str = Field(min_length=1, pattern=r"\S")


class CV(BaseModel):
    """The intentionally small, portable JSON representation of a CV."""

    model_config = ConfigDict(extra="forbid")

    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    links: list[str] = Field(default_factory=list)
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    ai_native: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)
    languages: list[Language] = Field(default_factory=list)


CLIENT_PREFIX = re.compile(r"^([^:()]+?)(?: \([^)]*\))?: ")


def load_cv(path: Path) -> CV:
    """Load and validate a CV JSON file."""
    try:
        return CV.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CV JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"CV JSON is invalid: {path}") from exc


TERM = re.compile(r"[A-Za-z][A-Za-z0-9+#]*")


def unsupported_terms(text: str, source: str) -> list[str]:
    """Proper nouns and acronyms in *text* that the source CV never mentions."""
    known = {match.group().casefold() for match in TERM.finditer(source)}
    terms = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        for position, match in enumerate(TERM.finditer(sentence)):
            word = match.group()
            notable = any(c.isupper() for c in word[1:]) or (
                position > 0 and word[0].isupper()
            )
            if notable and word.casefold() not in known:
                terms.append(word)
    return list(dict.fromkeys(terms))


def tailor_cv(
    cv: CV,
    job_description: str,
    *,
    model: str = MODEL,
    llm: Runnable | None = None,
    diagnostics: TailorDiagnostics | None = None,
    job_cache: Path | None = None,
    refresh_job_analysis: bool = False,
    matching_cache: Path | None = None,
    refresh_matching: bool = False,
    model_identity: str | None = None,
) -> CV:
    """Tailor from source evidence; expose the full audit through diagnostics."""
    from assistant.cv_tailoring import tailor_with_report

    model_llm = llm or local_llm(model)
    if matching_cache is not None and model_identity is None:
        if not isinstance(model_llm, ChatOllama):
            raise ValueError("A custom model needs model_identity to cache matching.")
        model_identity = local_model_identity(model_llm)
    result, report = tailor_with_report(
        cv, job_description, model_llm, diagnostics,
        job_cache=job_cache, refresh_job_analysis=refresh_job_analysis,
        matching_cache=matching_cache, refresh_matching=refresh_matching,
        model_identity=model_identity,
    )
    if diagnostics is None:
        for warning in report.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
    return result


def polish_cv(cv: CV, *, model: str = MODEL, llm: Runnable | None = None) -> CV:
    """Polish prose only when an independent evidence review accepts it."""
    from assistant.cv_tailoring import polish_with_report

    result, report = polish_with_report(cv, llm or local_llm(model))
    for warning in report.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    return result


def escape_latex(value: str) -> str:
    """Escape text inserted into a LaTeX document."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def to_latex(cv: CV) -> str:
    """Render a CV to a compact, ATS-friendly LaTeX document."""
    contact = " | ".join(
        escape_latex(value)
        for value in [cv.email, cv.phone, cv.location, *cv.links]
        if value
    )
    lines = [
        r"\documentclass[12pt,a4paper]{article}",
        r"\usepackage[margin=1.6cm]{geometry}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{enumitem}",
        r"\setlist[itemize]{leftmargin=*,nosep}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        rf"\begin{{center}}\LARGE\textbf{{{escape_latex(cv.name)}}}\end{{center}}",
        rf"\begin{{center}}{contact}\end{{center}}" if contact else "",
    ]
    if cv.summary:
        lines += [r"\section*{Profile}", escape_latex(cv.summary)]
    if cv.skills:
        lines += [r"\section*{Skills}", escape_latex(", ".join(cv.skills))]
    if cv.ai_native:
        lines += [
            r"\section*{AI-Native Practice}",
            r"\begin{itemize}",
            *[rf"\item {escape_latex(bullet)}" for bullet in cv.ai_native],
            r"\end{itemize}",
        ]
    if cv.experience:
        lines.append(r"\section*{Experience}")
        for item in cv.experience:
            lines += [
                rf"\noindent\textbf{{{escape_latex(item.role)}}} -- "
                rf"{escape_latex(item.company)} \hfill {escape_latex(item.dates)}",
                r"\begin{itemize}",
                *[rf"\item {escape_latex(bullet)}" for bullet in item.bullets],
                r"\end{itemize}",
            ]
    if cv.languages:
        lines += [
            r"\section*{Languages}",
            escape_latex(", ".join(
                f"{language.name}: {language.proficiency}" for language in cv.languages
            )),
        ]
    if cv.education:
        lines.append(r"\section*{Education}")
        for item in cv.education:
            lines.append(
                rf"\noindent\textbf{{{escape_latex(item.degree)}}} -- "
                rf"{escape_latex(item.institution)} \hfill {escape_latex(item.dates)}"
            )
    if cv.publications:
        lines.append(r"\section*{Publications}")
        for item in cv.publications:
            details_parts = [
                escape_latex(value)
                for value in [item.authors, item.date, item.publisher]
                if value
            ]
            if item.doi:
                doi = escape_latex(item.doi)
                details_parts.append(rf"\href{{https://doi.org/{doi}}}{{DOI: {doi}}}")
            elif item.url:
                details_parts.append(escape_latex(item.url))
            details = " -- ".join(details_parts)
            lines.append(
                rf"\noindent\textbf{{{escape_latex(item.title)}}}"
                + (rf" ({details})" if details else "")
            )
    if cv.certifications:
        lines.append(r"\section*{Certifications}")
        for item in cv.certifications:
            details = " -- ".join(
                escape_latex(value)
                for value in [item.issuer, item.date, item.url]
                if value
            )
            lines.append(
                rf"\noindent\textbf{{{escape_latex(item.name)}}}"
                + (rf" ({details})" if details else "")
            )
    lines += [r"\end{document}"]
    return "\n".join(line for line in lines if line)


def write_pdf(latex: str, output: Path) -> Path:
    """Compile LaTeX with pdflatex, or Tectonic when pdflatex is unavailable."""
    executable = shutil.which("pdflatex") or shutil.which("tectonic")
    if executable is None:
        raise RuntimeError(
            "A LaTeX compiler is required to create a PDF. "
            "Install MacTeX (pdflatex) or Tectonic."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    tex_path = output.with_suffix(".tex")
    tex_path.write_text(latex, encoding="utf-8")
    if Path(executable).name == "tectonic":
        command = [executable, "--outdir", str(output.parent), str(tex_path)]
    else:
        command = [
            executable,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-output-directory",
            str(output.parent),
            str(tex_path),
        ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    generated = output.parent / f"{tex_path.stem}.pdf"
    if not generated.exists():
        raise RuntimeError(f"pdflatex did not create {generated}")
    return generated


def main() -> None:
    started = time.perf_counter()
    from assistant.cv_tailoring import (
        ScoringRubric,
        polish_with_report,
        tailor_with_report,
    )

    parser = argparse.ArgumentParser(
        description="Export a complete JSON CV as PDF, optionally tailored to a job."
    )
    parser.add_argument("--cv", type=Path, required=True, help="Source CV JSON file")
    parser.add_argument(
        "--job", type=Path, help="Job description text file (omit for the full CV)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    parser.add_argument(
        "--rubric", type=Path, help="Optional scoring rubric JSON with requirement weights"
    )
    parser.add_argument(
        "--job-cache", type=Path, default=Path("output/job-requirements"),
        help="Directory for reusable parsed job analyses (default: output/job-requirements)",
    )
    parser.add_argument(
        "--refresh-job-analysis", action="store_true",
        help="Reparse the job and replace its cached analysis",
    )
    parser.add_argument(
        "--matching-cache", type=Path, default=Path("output/cv-matches"),
        help="Private cache of completed CV/job matching analyses",
    )
    parser.add_argument(
        "--refresh-matching", action="store_true",
        help="Recompute matching instead of reusing a completed analysis",
    )
    parser.add_argument(
        "--no-matching-cache", action="store_true",
        help="Disable completed matching cache for this run",
    )
    args = parser.parse_args()
    if args.refresh_job_analysis and args.job is None:
        parser.error("--refresh-job-analysis requires --job")
    if args.refresh_matching and args.job is None:
        parser.error("--refresh-matching requires --job")

    cv = load_cv(args.cv)
    llm = local_llm(args.model)
    if args.job is not None:
        matching_cache = None if args.no_matching_cache else args.matching_cache
        identity = local_model_identity(llm) if matching_cache is not None else None
        rubric = (
            ScoringRubric.model_validate_json(args.rubric.read_text(encoding="utf-8"))
            if args.rubric is not None else None
        )
        cv, report = tailor_with_report(
            cv, args.job.read_text(encoding="utf-8"), llm, rubric=rubric,
            job_cache=args.job_cache, refresh_job_analysis=args.refresh_job_analysis,
            matching_cache=matching_cache, refresh_matching=args.refresh_matching,
            model_identity=identity,
        )
    else:
        if args.rubric is not None:
            parser.error("--rubric requires --job")
        cv, report = polish_with_report(cv, llm)
    export_started = time.perf_counter()
    write_pdf(to_latex(cv), args.output)
    args.output.with_suffix(".json").write_text(
        cv.model_dump_json(indent=2), encoding="utf-8"
    )
    report_path = args.output.with_suffix(".report.json")
    report.performance.stage_seconds["export"] = time.perf_counter() - export_started
    report.performance.total_seconds = time.perf_counter() - started
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    for warning in report.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    if args.job is not None:
        print(
            "Job analysis: reused cached requirements."
            if report.job_analysis_cached
            else "Job analysis: parsed and cached requirements."
        )
        print(
            "Matching analysis: "
            + ("reused cached results." if report.matching_analysis_cached else "computed.")
        )
        match = (
            f"{report.match_percent:.1f}%"
            if report.match_percent is not None else "insufficient information"
        )
        print(f"CV-evidenced job match: {match} (not a hiring probability)")
        if report.must_have_percent is not None:
            print(f"Must-have coverage: {report.must_have_percent:.1f}%")
        if report.unresolved_eligibility_ids:
            print("Warning: unresolved eligibility constraints; see the report.", file=sys.stderr)
    print(f"Created {args.output}")
    print(f"Evidence and rewrite report: {report_path}")
    print(
        f"Performance: {report.performance.total_seconds:.1f}s, "
        f"{report.performance.total_model_calls} model call(s)"
    )
    for stage, calls in report.performance.stage_model_calls.items():
        print(
            f"  {stage}: {calls} call(s), "
            f"{report.performance.stage_model_seconds[stage]:.1f}s"
        )


if __name__ == "__main__":
    main()
