"""Tailor a structured CV to a job description and render it as LaTeX/PDF."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, ConfigDict, Field

MODEL = "llama3.1:8b"


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
    url: str = ""


class Certification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    issuer: str = ""
    date: str = ""
    url: str = ""


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
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)
    certifications: list[Certification] = Field(default_factory=list)


TAILOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You tailor CVs for a job application. Return only valid JSON matching
the supplied CV schema. Use only facts present in the source CV; never invent
employers, dates, metrics, technologies, or qualifications. Reorder skills and
experience for relevance, and rewrite bullets only to clarify existing evidence.
Keep every claim truthful and concise.""",
        ),
        (
            "human",
            "Job description:\n{job_description}\n\nSource CV JSON:\n{cv_json}",
        ),
    ]
)


def load_cv(path: Path) -> CV:
    """Load and validate a CV JSON file."""
    try:
        return CV.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CV JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"CV JSON is invalid: {path}") from exc


def tailor_cv(
    cv: CV,
    job_description: str,
    *,
    model: str = MODEL,
    llm: Any | None = None,
) -> CV:
    """Use the local LLM to tailor *cv* to *job_description*."""
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")
    chain = TAILOR_PROMPT | (llm or OllamaLLM(model=model)) | StrOutputParser()
    response = chain.invoke(
        {
            "job_description": job_description.strip(),
            "cv_json": cv.model_dump_json(indent=2),
        }
    )
    try:
        return CV.model_validate_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("The language model returned invalid CV JSON.") from exc


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
        r"\documentclass[10pt,a4paper]{article}",
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
            details = " -- ".join(
                escape_latex(value)
                for value in [item.authors, item.date, item.url]
                if value
            )
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
    parser = argparse.ArgumentParser(
        description="Tailor a JSON CV and export it as PDF."
    )
    parser.add_argument("--cv", type=Path, required=True, help="Source CV JSON file")
    parser.add_argument(
        "--job", type=Path, required=True, help="Job description text file"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    args = parser.parse_args()

    tailored = tailor_cv(
        load_cv(args.cv),
        args.job.read_text(encoding="utf-8"),
        model=args.model,
    )
    args.output.with_suffix(".json").write_text(
        tailored.model_dump_json(indent=2), encoding="utf-8"
    )
    write_pdf(to_latex(tailored), args.output)
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
