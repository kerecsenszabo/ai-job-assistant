import json
from pathlib import Path

import pytest
from langchain_core.runnables import RunnableLambda

from assistant import cli, cv_generator
from assistant.cv_generator import CV, escape_latex, to_latex
from assistant.cv_tailoring import TailoringReport


@pytest.fixture(autouse=True)
def fake_model_identity(monkeypatch):
    monkeypatch.setattr(cv_generator, "local_model_identity", lambda llm: "fake-v1")


def test_latex_preserves_sections_and_escapes_text():
    cv = CV(
        name="Example & Co",
        skills=["C++", "Python"],
        experience=[{
            "company": "Company", "role": "Engineer", "dates": "2024",
            "bullets": ["Improved performance by 10%."],
        }],
        education=[{"institution": "University", "degree": "BSc"}],
        publications=[{"title": "Paper"}],
        certifications=[{"name": "Certificate"}],
        languages=[
            {"name": "English", "proficiency": "Native"},
            {"name": "Hungarian", "proficiency": "Native"},
            {"name": "German", "proficiency": "Basic"},
        ],
    )
    latex = to_latex(cv)
    assert r"\documentclass[11pt,a4paper]{article}" in latex
    assert r"Example \& Co" in latex
    assert r"10\%" in latex
    for title in ("Skills", "Experience", "Languages", "Education", "Publications", "Certifications"):
        assert rf"\section*{{{title}}}" in latex
    assert "English: Native, Hungarian: Native, German: Basic" in latex
    assert escape_latex("_#$") == r"\_\#\$"


def test_languages_are_optional_for_existing_source_cvs():
    cv = CV(name="Example")
    assert cv.languages == []
    assert r"\section*{Languages}" not in to_latex(cv)


def test_latex_renders_anonymous_project_bullets_under_employer_heading():
    cv = CV(name="Example", experience=[{
        "company": "EPAM", "role": "Engineer", "dates": "2023 - 2025",
        "bullets": [
            "Delivered solutions for multiple clients.",
            "Developed recommender models over two years.",
            "Deployed services with Docker.",
            "Supported production services.",
            "Built data pipelines during a six-month project.",
            "Tuned Spark workloads.",
        ],
    }])
    original = cv.model_dump_json()

    latex = to_latex(cv)

    assert r"\textbf{Engineer} -- EPAM" in latex
    assert r"\textit{" not in latex
    assert r"\item Developed recommender models over two years." in latex
    assert r"\item Deployed services with Docker." in latex
    assert r"\item Supported production services." in latex
    assert r"\item Delivered solutions for multiple clients." in latex
    assert latex.count(r"\begin{itemize}") == latex.count(r"\end{itemize}") == 1
    assert cv.model_dump_json() == original


def test_cli_writes_separate_report_not_match_score_in_cv(tmp_path, monkeypatch, capsys):
    from assistant import cv_tailoring

    source = tmp_path / "source.json"
    source.write_text(CV(name="Example").model_dump_json())
    job = tmp_path / "job.txt"
    job.write_text("Python required")
    output = tmp_path / "cv.pdf"
    report = TailoringReport(
        match_percent=50, must_have_percent=50,
        warnings=["A rewrite was rejected."],
        unresolved_eligibility_ids=["requirement/0"],
    )
    monkeypatch.setattr(cv_generator, "local_llm", lambda model: object())
    monkeypatch.setattr(
        cv_tailoring, "tailor_with_report",
        lambda cv, description, llm, **kwargs: (cv, report),
    )
    monkeypatch.setattr(
        cv_generator, "write_pdf",
        lambda latex, path: path.write_text(latex),
    )
    cli.main([
        "generate", "--cv", str(source), "--job", str(job),
        "--output", str(output),
    ])
    assert output.exists()
    exported = json.loads(output.with_suffix(".json").read_text())
    audit = json.loads(output.with_suffix(".report.json").read_text())
    assert "match_percent" not in exported
    assert audit["match_percent"] == 50
    captured = capsys.readouterr()
    assert "50.0%" in captured.out
    assert "not a hiring probability" in captured.out
    assert "parsed and cached" in captured.out
    assert "rejected" in captured.err
    assert "eligibility" in captured.err


def test_cli_general_cv_also_writes_rewrite_audit(tmp_path, monkeypatch):
    from assistant import cv_tailoring

    source = tmp_path / "source.json"
    source.write_text(CV(name="Example").model_dump_json())
    output = tmp_path / "cv.pdf"
    cache_dir = tmp_path / "unused-job-cache"
    monkeypatch.setattr(cv_generator, "local_llm", lambda model: object())
    monkeypatch.setattr(
        cv_tailoring, "polish_with_report",
        lambda cv, llm: (cv, TailoringReport(label="General-purpose CV rewrite audit")),
    )
    monkeypatch.setattr(cv_generator, "write_pdf", lambda latex, path: path.write_text(latex))
    cli.main([
        "generate", "--cv", str(source), "--output", str(output),
        "--job-cache", str(cache_dir),
    ])
    assert output.with_suffix(".report.json").exists()
    assert not cache_dir.exists()


def test_cli_exports_rewritten_experience_for_every_role(tmp_path, monkeypatch):
    cv = CV(
        name="Example", summary="I build pipelines.", skills=["Python", "SQL"],
        experience=[
            {
                "company": "Current Co", "role": "Engineer", "dates": "2024",
                "bullets": ["Built Python pipelines.", "Maintained SQL reports."],
            },
            {
                "company": "Previous Co", "role": "Developer", "dates": "2023",
                "bullets": ["Client A: Supported a prototype."],
            },
        ],
    )
    rewritten = {
        "experience/0/bullets/0": "Developed Python pipelines.",
        "experience/0/bullets/1": "Updated SQL reports.",
        "experience/1/bullets/0": "Client A: provided support for a prototype.",
    }
    responses = iter([
        {
            "bullets": {
                sid: {"source_ids": [sid], "text": text}
                for sid, text in rewritten.items()
            },
            "summary_sentences": [],
        },
        {"verdicts": {
            sid: {"status": "supported", "reason": "Preserves the original facts."}
            for sid in rewritten
        }},
    ])
    llm = RunnableLambda(lambda prompt, **kwargs: json.dumps(next(responses)))
    source = tmp_path / "source.json"
    source.write_text(cv.model_dump_json())
    output = tmp_path / "cv.pdf"
    monkeypatch.setattr(cv_generator, "local_llm", lambda model: llm)
    monkeypatch.setattr(cv_generator, "write_pdf", lambda latex, path: path.write_text(latex))
    cli.main([
        "generate", "--cv", str(source), "--output", str(output),
    ])

    exported = CV.model_validate_json(output.with_suffix(".json").read_text())
    assert [role.bullets for role in exported.experience] == [
        list(rewritten.values())[:2], list(rewritten.values())[2:],
    ]
    for text in rewritten.values():
        assert text in output.read_text()
    assert exported.model_dump(exclude={"experience"}) == cv.model_dump(exclude={"experience"})
    assert [role.model_dump(exclude={"bullets"}) for role in exported.experience] == [
        role.model_dump(exclude={"bullets"}) for role in cv.experience
    ]
    assert source.read_text() == cv.model_dump_json()
    report = TailoringReport.model_validate_json(output.with_suffix(".report.json").read_text())
    assert len(report.rewrites) == 3
    assert all(item.status == "accepted" and item.exported != item.original
               for item in report.rewrites)


@pytest.mark.parametrize("refresh", [False, True])
def test_cli_passes_cache_options_and_displays_reuse(
    tmp_path, monkeypatch, capsys, refresh
):
    from assistant import cv_tailoring

    source = tmp_path / "source.json"
    source.write_text(CV(name="Example").model_dump_json())
    job = tmp_path / "job.txt"
    job.write_text("Python required")
    output = tmp_path / "cv.pdf"
    cache_dir = tmp_path / "requirements"
    calls = []

    def tailor(cv, description, llm, *, rubric, job_cache, refresh_job_analysis, **kwargs):
        calls.append((description, rubric, job_cache, refresh_job_analysis))
        return cv, TailoringReport(job_analysis_cached=not refresh)

    monkeypatch.setattr(cv_generator, "local_llm", lambda model: object())
    monkeypatch.setattr(cv_tailoring, "tailor_with_report", tailor)
    monkeypatch.setattr(cv_generator, "write_pdf", lambda latex, path: path.write_text(latex))
    args = [
        "generate", "--cv", str(source), "--job", str(job),
        "--output", str(output), "--job-cache", str(cache_dir),
    ]
    if refresh:
        args.append("--refresh-job-analysis")
    cli.main(args)
    assert calls == [("Python required", None, cache_dir, refresh)]
    expected = "parsed and cached" if refresh else "reused cached"
    assert expected in capsys.readouterr().out


def test_cli_default_cache_directory(tmp_path, monkeypatch):
    from assistant import cv_tailoring

    source = tmp_path / "source.json"
    source.write_text(CV(name="Example").model_dump_json())
    job = tmp_path / "job.txt"
    job.write_text("Python required")
    output = tmp_path / "cv.pdf"
    calls = []

    def tailor(cv, description, llm, **kwargs):
        calls.append(kwargs)
        return cv, TailoringReport()

    monkeypatch.setattr(cv_generator, "local_llm", lambda model: object())
    monkeypatch.setattr(cv_tailoring, "tailor_with_report", tailor)
    monkeypatch.setattr(cv_generator, "write_pdf", lambda latex, path: path.write_text(latex))
    cli.main([
        "generate", "--cv", str(source), "--job", str(job),
        "--output", str(output),
    ])
    assert calls == [{
        "rubric": None, "job_cache": Path("output/job-requirements"),
        "refresh_job_analysis": False,
        "matching_cache": Path("output/cv-matches"),
        "refresh_matching": False, "model_identity": "fake-v1",
    }]


def test_cli_refresh_requires_job_before_loading_cv(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--cv", "missing.json", "--output", "cv.pdf",
            "--refresh-job-analysis",
        ])
    assert exc.value.code == 2
    assert "--refresh-job-analysis requires --job" in capsys.readouterr().err


@pytest.mark.parametrize("opt_in", [False, True])
def test_tailor_cv_passes_optional_cache_settings(tmp_path, monkeypatch, opt_in):
    from assistant import cv_tailoring

    cv = CV(name="Example")
    llm = object()
    diagnostics = cv_generator.TailorDiagnostics()
    cache = tmp_path / "requirements" if opt_in else None
    calls = []

    def tailor(source, description, runnable, measurements, **kwargs):
        calls.append((source, description, runnable, measurements, kwargs))
        return source, TailoringReport()

    monkeypatch.setattr(cv_tailoring, "tailor_with_report", tailor)
    options = (
        {"job_cache": cache, "refresh_job_analysis": True} if opt_in else {}
    )
    result = cv_generator.tailor_cv(
        cv, "Python required", llm=llm, diagnostics=diagnostics, **options
    )
    assert result == cv
    assert calls == [(cv, "Python required", llm, diagnostics, {
        "job_cache": cache, "refresh_job_analysis": opt_in,
        "matching_cache": None, "refresh_matching": False, "model_identity": None,
    })]


def test_default_model_is_granite_3b():
    assert cv_generator.MODEL == "granite4.2:3b"


def test_cli_can_disable_matching_cache(tmp_path, monkeypatch):
    from assistant import cv_tailoring

    source = tmp_path / "source.json"
    source.write_text(CV(name="Example").model_dump_json())
    job = tmp_path / "job.txt"
    job.write_text("Python")
    calls = []

    def tailor(cv, description, llm, **kwargs):
        calls.append(kwargs)
        return cv, TailoringReport()

    monkeypatch.setattr(cv_generator, "local_llm", lambda model: object())
    monkeypatch.setattr(cv_tailoring, "tailor_with_report", tailor)
    monkeypatch.setattr(cv_generator, "write_pdf", lambda latex, path: path.write_text(latex))
    monkeypatch.setattr(
        cv_generator, "local_model_identity",
        lambda llm: pytest.fail("Disabled cache must not look up model metadata"),
    )
    cli.main([
        "generate", "--cv", str(source), "--job", str(job),
        "--output", str(tmp_path / "cv.pdf"), "--no-matching-cache",
    ])
    assert calls[0]["matching_cache"] is None
    assert calls[0]["model_identity"] is None


def test_cli_matching_refresh_requires_job(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([
            "generate", "--cv", "missing.json", "--output", "cv.pdf",
            "--refresh-matching",
        ])
    assert exc.value.code == 2
    assert "--refresh-matching requires --job" in capsys.readouterr().err
