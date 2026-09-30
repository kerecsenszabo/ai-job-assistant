"""Render a structured CV as LaTeX/PDF, optionally tailored to a job."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import OllamaLLM
from pydantic import BaseModel, ConfigDict, Field

MODEL = "llama3.1:8b"
# Source CV + job description + full JSON reply exceeds Ollama's default window.
CONTEXT_TOKENS = 16384


def local_llm(model: str) -> OllamaLLM:
    """Local LLM with room for a full CV round-trip."""
    return OllamaLLM(model=model, temperature=0, num_ctx=CONTEXT_TOKENS)


def structured_llm(llm: Runnable, schema: type[BaseModel]) -> Runnable:
    """Constrain an Ollama request to the JSON schema for *schema*."""
    return llm.bind(format=schema.model_json_schema())


class Experience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: str
    role: str
    dates: str
    bullets: list[str] = Field(min_length=1)


class PolishedExperience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experience: list[Experience]


class TailorSelection(BaseModel):
    """Ids of existing CV items, ranked by relevance to a job."""

    model_config = ConfigDict(extra="forbid")

    skills: list[int] = Field(default_factory=list)
    ai_native: list[int] = Field(default_factory=list)
    experience: list[list[int]] = Field(default_factory=list)


class TailoredSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=1)


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


SELECT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You select CV content for a job application. You never write CV text;
you only rank existing items by their numeric ids. Return only a JSON object:
{{"skills": [ids], "ai_native": [ids], "experience": [[ids], ...]}}
- skills: ids of the skills relevant to the job, most relevant first.
- ai_native: ids of all ai_native items, most relevant first.
- experience: one list per role, in the given role order. Each list holds the
  ids of that role's bullets most relevant to the job, most relevant first.
  Prefer bullets showing seniority and ownership when they are relevant.""",
        ),
        (
            "human",
            "Job description:\n{job_description}\n\nNumbered CV items:\n{items_json}",
        ),
    ]
)


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Write the profile summary of a CV tailored to a job. Return only a
JSON object: {{"summary": "..."}}.
The candidate evidence is the only source of facts; the job description only
tells you what to emphasise. Never claim a technology, framework, cloud,
domain or experience the evidence does not mention, even if the job asks for
it. Do not mention or describe the hiring company. Start from the original
summary and adapt it: bring forward the evidence, including ai_native items,
that matches the job's main requirements. Write 3-4 concise sentences in the
same first-person voice as the original summary. Avoid cliches such as
"seasoned", "proven track record", "results-driven" or "I am excited".""",
        ),
        (
            "human",
            "Job description:\n{job_description}\n\n"
            "Candidate evidence JSON:\n{evidence_json}{feedback}",
        ),
    ]
)


POLISH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Polish the wording of CV experience bullets for a general-purpose CV.
Return only valid JSON with an "experience" array in the same structure as the
input. Keep every company, role, date, position and bullet in the same order;
output exactly one bullet for each input bullet. Preserve ALL facts, tools,
clients, time spans, responsibilities and qualifications from each bullet.
Do not shorten by dropping details or invent anything. Use consistent tense:
present for current work, past for previous roles. For consecutive bullets about
the same client, the input already names the client in the first bullet only;
preserve that grouping and make subsequent bullets read naturally in context.
Keep other sections untouched.""",
        ),
        ("human", "Source experience JSON:\n{experience_json}"),
    ]
)


CLIENT_PREFIX = re.compile(r"^([^:()]+?)(?: \([^)]*\))?: ")


def group_client_bullets(experience: Experience) -> Experience:
    """Avoid repeating a client's name in consecutive bullets about its project."""
    bullets = []
    previous_client = None
    for bullet in experience.bullets:
        match = CLIENT_PREFIX.match(bullet)
        client = match.group(1) if match else None
        bullets.append(
            bullet[match.end() :]
            if match is not None and client == previous_client
            else bullet
        )
        previous_client = client
    return experience.model_copy(update={"bullets": bullets})


def load_cv(path: Path) -> CV:
    """Load and validate a CV JSON file."""
    try:
        return CV.model_validate_json(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"CV JSON not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"CV JSON is invalid: {path}") from exc


def same_roles(generated: list[Any], source: list[Experience]) -> bool:
    """Whether the LLM kept every role with its company, title and dates."""
    return len(generated) == len(source) and all(
        (item.company, item.role, item.dates)
        == (original.company, original.role, original.dates)
        for item, original in zip(generated, source)
    )


MIN_SKILLS = 8
MIN_BULLETS = 3
SUMMARY_ATTEMPTS = 2
TERM = re.compile(r"[A-Za-z][A-Za-z0-9+#]*")


def ranked(ids: list[int], count: int, minimum: int) -> list[int]:
    """Valid, unique ids in the model's order, topped up to *minimum* items."""
    chosen = list(dict.fromkeys(i for i in ids if 0 <= i < count))
    chosen += [i for i in range(count) if i not in chosen][
        : max(0, minimum - len(chosen))
    ]
    return chosen


def cluster_by_client(bullets: list[str]) -> list[str]:
    """Keep bullets about the same client together, in order of first mention."""
    groups: dict[object, list[str]] = {}
    for bullet in bullets:
        match = CLIENT_PREFIX.match(bullet)
        groups.setdefault(match.group(1) if match else object(), []).append(bullet)
    return [bullet for group in groups.values() for bullet in group]


def unsupported_terms(text: str, source: str) -> list[str]:
    """Proper nouns and acronyms in *text* that the source CV never mentions."""
    known = source.casefold()
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
    llm: Any | None = None,
) -> CV:
    """Use the local LLM to tailor *cv* to *job_description*.

    The model only ranks existing skills, AI-native items and experience
    bullets, so it cannot invent responsibilities. It then writes a summary
    from the selected evidence; summaries naming anything absent from the
    source CV are retried and, failing that, replaced by the source summary.
    """
    if not job_description.strip():
        raise ValueError("Job description cannot be empty.")
    job_description = job_description.strip()
    llm = llm or local_llm(model)

    items = {
        "skills": dict(enumerate(cv.skills)),
        "ai_native": dict(enumerate(cv.ai_native)),
        "experience": [
            {
                "company": item.company,
                "role": item.role,
                "bullets": dict(enumerate(item.bullets)),
            }
            for item in cv.experience
        ],
    }
    response = (
        SELECT_PROMPT | structured_llm(llm, TailorSelection) | StrOutputParser()
    ).invoke(
        {"job_description": job_description, "items_json": json.dumps(items, indent=2)}
    )
    try:
        selection = TailorSelection.model_validate_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("The language model returned an invalid selection.") from exc

    skill_ids = ranked(selection.skills, len(cv.skills), MIN_SKILLS)
    ai_ids = ranked(selection.ai_native, len(cv.ai_native), len(cv.ai_native))
    experience = []
    for index, item in enumerate(cv.experience):
        ids = selection.experience[index] if index < len(selection.experience) else []
        bullets = [item.bullets[i] for i in ranked(ids, len(item.bullets), MIN_BULLETS)]
        experience.append(
            group_client_bullets(
                item.model_copy(update={"bullets": cluster_by_client(bullets)})
            )
        )
    tailored = cv.model_copy(
        update={
            "skills": [cv.skills[i] for i in skill_ids],
            "ai_native": [cv.ai_native[i] for i in ai_ids],
            "experience": experience,
        }
    )

    evidence = tailored.model_dump(
        include={"summary", "skills", "ai_native", "experience"}
    )
    source_text = cv.model_dump_json()
    feedback = ""
    for _ in range(SUMMARY_ATTEMPTS):
        response = (
            SUMMARY_PROMPT | structured_llm(llm, TailoredSummary) | StrOutputParser()
        ).invoke(
            {
                "job_description": job_description,
                "evidence_json": json.dumps(evidence, indent=2),
                "feedback": feedback,
            }
        )
        try:
            summary = TailoredSummary.model_validate_json(response).summary
        except (json.JSONDecodeError, ValueError):
            feedback = "\n\nYour previous reply was not valid JSON. Try again."
            continue
        unsupported = unsupported_terms(summary, source_text)
        if not unsupported:
            return tailored.model_copy(update={"summary": summary})
        feedback = (
            "\n\nYour previous summary mentioned terms the evidence does not "
            f"support: {', '.join(unsupported)}. Rewrite it without them."
        )
    print(
        "Warning: kept the source summary; the model's tailored summary was not "
        "supported by the CV.",
        file=sys.stderr,
    )
    return tailored


def polish_cv(cv: CV, *, model: str = MODEL, llm: Any | None = None) -> CV:
    """Polish experience prose without selecting or filtering CV content."""
    chain = (
        POLISH_PROMPT
        | structured_llm(llm or local_llm(model), PolishedExperience)
        | StrOutputParser()
    )
    response = chain.invoke(
        {
            "experience_json": json.dumps(
                [group_client_bullets(item).model_dump() for item in cv.experience],
                indent=2,
            ),
        }
    )
    try:
        polished = PolishedExperience.model_validate_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "The language model returned invalid experience JSON."
        ) from exc
    if not same_roles(polished.experience, cv.experience) or any(
        len(item.bullets) != len(source.bullets)
        for item, source in zip(polished.experience, cv.experience)
    ):
        raise ValueError("The language model changed the experience structure.")
    return cv.model_copy(update={"experience": polished.experience})


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
    parser = argparse.ArgumentParser(
        description="Export a complete JSON CV as PDF, optionally tailored to a job."
    )
    parser.add_argument("--cv", type=Path, required=True, help="Source CV JSON file")
    parser.add_argument(
        "--job", type=Path, help="Job description text file (omit for the full CV)"
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path")
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    args = parser.parse_args()

    cv = load_cv(args.cv)
    if args.job is not None:
        cv = tailor_cv(cv, args.job.read_text(encoding="utf-8"), model=args.model)
    else:
        cv = polish_cv(cv, model=args.model)
    write_pdf(to_latex(cv), args.output)
    args.output.with_suffix(".json").write_text(
        cv.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"Created {args.output}")


if __name__ == "__main__":
    main()
