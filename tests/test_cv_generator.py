import json
import sys

from assistant import cv_generator
from assistant.cv_generator import CV, escape_latex, to_latex
from assistant.cv_tailoring import TailoringReport


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
    )
    latex = to_latex(cv)
    assert r"Example \& Co" in latex
    assert r"10\%" in latex
    for title in ("Skills", "Experience", "Education", "Publications", "Certifications"):
        assert rf"\section*{{{title}}}" in latex
    assert escape_latex("_#$") == r"\_\#\$"


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
        lambda cv, description, llm, rubric: (cv, report),
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
    assert "rejected" in captured.err
    assert "eligibility" in captured.err


def test_cli_general_cv_also_writes_rewrite_audit(tmp_path, monkeypatch):
    from assistant import cv_tailoring

    source = tmp_path / "source.json"
    source.write_text(CV(name="Example").model_dump_json())
    output = tmp_path / "cv.pdf"
    monkeypatch.setattr(cv_generator, "local_llm", lambda model: object())
    monkeypatch.setattr(
        cv_tailoring, "polish_with_report",
        lambda cv, llm: (cv, TailoringReport(label="General-purpose CV rewrite audit")),
    )
    monkeypatch.setattr(cv_generator, "write_pdf", lambda latex, path: path.write_text(latex))
    monkeypatch.setattr(sys, "argv", [
        "cv_generator", "--cv", str(source), "--output", str(output),
    ])
    cv_generator.main()
    assert output.with_suffix(".report.json").exists()
