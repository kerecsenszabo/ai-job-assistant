import json
import sys
from pathlib import Path

import pytest

from assistant import cv_generator
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
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", str(source), "--job", str(job),
        "--output", str(output),
    ])
    cv_generator.main()
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
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", str(source), "--output", str(output),
        "--job-cache", str(cache_dir),
    ])
    cv_generator.main()
    assert output.with_suffix(".report.json").exists()
    assert not cache_dir.exists()


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
        "cv_generator", "--cv", str(source), "--job", str(job),
        "--output", str(output), "--job-cache", str(cache_dir),
    ]
    if refresh:
        args.append("--refresh-job-analysis")
    monkeypatch.setattr(sys, "argv", args)
    cv_generator.main()
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
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", str(source), "--job", str(job),
        "--output", str(output),
    ])
    cv_generator.main()
    assert calls == [{
        "rubric": None, "job_cache": Path("output/job-requirements"),
        "refresh_job_analysis": False,
        "matching_cache": Path("output/cv-matches"),
        "refresh_matching": False, "model_identity": "fake-v1",
    }]


def test_cli_refresh_requires_job_before_loading_cv(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", "missing.json", "--output", "cv.pdf",
        "--refresh-job-analysis",
    ])
    with pytest.raises(SystemExit) as exc:
        cv_generator.main()
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
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", str(source), "--job", str(job),
        "--output", str(tmp_path / "cv.pdf"), "--no-matching-cache",
    ])
    cv_generator.main()
    assert calls[0]["matching_cache"] is None
    assert calls[0]["model_identity"] is None


def test_cli_matching_refresh_requires_job(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", "missing.json", "--output", "cv.pdf",
        "--refresh-matching",
    ])
    with pytest.raises(SystemExit) as exc:
        cv_generator.main()
    assert exc.value.code == 2
    assert "--refresh-matching requires --job" in capsys.readouterr().err
