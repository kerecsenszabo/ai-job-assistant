import json
import shutil
import sqlite3
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from assistant import model_benchmark as benchmark
from assistant.cv_generator import CV, TailorDiagnostics
from assistant.cv_tailoring import EvidenceItem, RequirementMatch, RewriteAudit, TailoringReport


REPORT_COLUMNS = (
    "match_percent", "must_have_percent", "accepted_rewrites",
    "rejected_rewrites", "unclear_rewrites", "report_json",
)


@pytest.fixture
def workspace():
    # Keep database and job fixtures in the project, not system temporary paths.
    path = Path.cwd() / f".benchmark-test-{uuid.uuid4().hex}"
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


@pytest.fixture
def cv():
    return CV(name="Candidate", summary="Python engineer", skills=["Python"])


@pytest.fixture
def report():
    return TailoringReport(
        match_percent=62.5,
        must_have_percent=50,
        selected_evidence_ids=["skills/0"],
        evidence=[EvidenceItem(id="skills/0", text="Python", section="skills")],
        matches=[
            RequirementMatch(
                requirement_id="r1", status="direct",
                evidence_ids=["skills/0"], explanation="Source lists Python",
            ),
        ],
        rewrites=[
            RewriteAudit(
                target_id=f"skills/{index}", source_ids=["skills/0"],
                original="Python", proposed="Python development",
                exported="Python development" if status == "accepted" else "Python",
                status=status, reason=f"Automated verdict: {status}",
            )
            for index, status in enumerate(("accepted", "accepted", "rejected", "unclear"))
        ],
    )


def result(cv, **kwargs):
    values = dict(
        run_id="run", model="model", job="job.txt", repetition=1, status="ok",
        total_seconds=3.5, diagnostics=TailorDiagnostics(
            selection_seconds=1, summary_seconds=2, summary_attempts=2,
            summary_fallback=True,
        ),
        keyword_recall=0.75, selection_recall=0.5, summary_words=2,
        selected_items=1, output_json=cv.model_dump_json(),
    )
    values.update(kwargs)
    return benchmark.BenchmarkResult(**values)


def create_old_database(path, cv, *, intermediate=False):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE benchmark_runs (
                run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
                ollama_version TEXT NOT NULL, cv_path TEXT NOT NULL,
                context_tokens INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE benchmark_results (
                run_id TEXT NOT NULL, model TEXT NOT NULL, job TEXT NOT NULL,
                repetition INTEGER NOT NULL, status TEXT NOT NULL,
                total_seconds REAL NOT NULL, selection_seconds REAL NOT NULL,
                summary_seconds REAL NOT NULL, summary_attempts INTEGER NOT NULL,
                summary_fallback INTEGER NOT NULL, keyword_recall REAL,
                selected_items INTEGER, output_json TEXT, error TEXT,
                PRIMARY KEY (run_id, model, job, repetition)
            )
            """
        )
        connection.execute(
            "INSERT INTO benchmark_runs VALUES (?, ?, ?, ?, ?)",
            ("run", "2026-09-30", "old", "cv.json", 16384),
        )
        connection.execute(
            "INSERT INTO benchmark_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run", "legacy", "job.txt", 1, "ok", 3, 1, 2, 1, 0, 0.4, 1,
             cv.model_dump_json(), None),
        )
        if intermediate:
            connection.execute("ALTER TABLE benchmark_results ADD COLUMN selection_recall REAL")
            connection.execute("ALTER TABLE benchmark_results ADD COLUMN summary_words INTEGER")
            connection.execute(
                "UPDATE benchmark_results SET selection_recall = 0.25, summary_words = 2"
            )


@pytest.mark.parametrize("intermediate", [False, True])
def test_connect_migrates_old_schema_without_losing_records(workspace, cv, intermediate):
    database = workspace / "old.sqlite"
    create_old_database(database, cv, intermediate=intermediate)
    for _ in range(2):
        with benchmark.connect(database) as connection:
            columns = {row[1]: row for row in connection.execute(
                "PRAGMA table_info(benchmark_results)"
            )}
            assert set(REPORT_COLUMNS) <= columns.keys()
            assert all(columns[name][3] == 0 for name in REPORT_COLUMNS)
            row = connection.execute(
                "SELECT keyword_recall, selection_recall, summary_words, "
                + ", ".join(REPORT_COLUMNS)
                + " FROM benchmark_results"
            ).fetchone()
            assert row == (0.4, 0.25 if intermediate else None,
                           2 if intermediate else None, *([None] * 6))
            assert connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone() == (1,)


def test_save_result_persists_all_metrics_and_full_report(workspace, cv, report):
    stored = result(cv, **benchmark.report_metrics(TailorDiagnostics(report=report)))
    database = workspace / "results.sqlite"
    with benchmark.connect(database) as connection:
        benchmark.save_result(connection, stored)
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT selection_seconds, summary_seconds, summary_attempts, summary_fallback, "
            "keyword_recall, selection_recall, summary_words, selected_items, output_json, "
            + ", ".join(REPORT_COLUMNS) + " FROM benchmark_results"
        ).fetchone()
    assert row[:-1] == (1, 2, 2, 1, 0.75, 0.5, 2, 1, cv.model_dump_json(),
                       62.5, 50, 2, 1, 1)
    assert json.loads(row[-1]) == report.model_dump(mode="json")
    assert json.loads(row[-1])["matches"][0]["evidence_ids"] == ["skills/0"]
    assert json.loads(row[-1])["rewrites"][2]["exported"] == "Python"


def test_save_result_without_report_keeps_metrics_null(workspace, cv):
    with benchmark.connect(workspace / "results.sqlite") as connection:
        benchmark.save_result(connection, result(cv))
        benchmark.save_result(connection, result(
            cv, repetition=2, status="error", error="RuntimeError: failed",
        ))
        rows = connection.execute(
            "SELECT " + ", ".join(REPORT_COLUMNS)
            + " FROM benchmark_results ORDER BY repetition"
        ).fetchall()
    assert rows == [tuple([None] * 6), tuple([None] * 6)]


@pytest.mark.parametrize("with_report", [False, True])
def test_benchmark_one_collects_report_and_keyword_proxies(
    workspace, monkeypatch, cv, report, with_report,
):
    job = workspace / "job.txt"
    job.write_text("Python Kubernetes", encoding="utf-8")
    calls = []

    def tailor(source, description, *, model, diagnostics):
        calls.append((source, description, model))
        diagnostics.report = report if with_report else None
        diagnostics.selection_seconds = 1.25
        return cv

    monkeypatch.setattr(benchmark, "tailor_cv", tailor)
    measured = benchmark.benchmark_one("run", "local-model", job, 2, cv)
    assert calls == [(cv, "Python Kubernetes", "local-model")]
    assert measured.status == "ok"
    assert measured.keyword_recall == measured.selection_recall == 0.5
    assert measured.summary_words == 2
    assert measured.selected_items == 1
    assert measured.output_json == cv.model_dump_json()
    assert measured.diagnostics.selection_seconds == 1.25
    assert measured.total_seconds >= 0
    assert measured.match_percent == (62.5 if with_report else None)
    assert measured.must_have_percent == (50 if with_report else None)
    assert measured.accepted_rewrites == (2 if with_report else None)
    assert measured.rejected_rewrites == (1 if with_report else None)
    assert measured.unclear_rewrites == (1 if with_report else None)
    assert measured.report_json == (report.model_dump_json() if with_report else None)


def test_empty_report_distinguishes_zero_verdicts_from_missing_report():
    metrics = benchmark.report_metrics(TailorDiagnostics(report=TailoringReport(
        match_percent=0, must_have_percent=None,
    )))
    assert metrics["match_percent"] == 0
    assert metrics["must_have_percent"] is None
    assert metrics["accepted_rewrites"] == 0
    assert metrics["rejected_rewrites"] == 0
    assert metrics["unclear_rewrites"] == 0
    assert benchmark.report_metrics(TailorDiagnostics()) == {}


def test_benchmark_failure_preserves_available_report(workspace, monkeypatch, cv, report):
    job = workspace / "job.txt"
    job.write_text("Python", encoding="utf-8")

    def fail(source, description, *, model, diagnostics):
        diagnostics.report = report
        raise RuntimeError("generation failed")

    monkeypatch.setattr(benchmark, "tailor_cv", fail)
    measured = benchmark.benchmark_one("run", "model", job, 1, cv)
    assert measured.status == "error"
    assert measured.error == "RuntimeError: generation failed"
    assert measured.report_json == report.model_dump_json()
    assert measured.keyword_recall is None
    assert measured.output_json is None


def test_report_handles_old_records_and_retains_relative_keyword_comparison(
    workspace, cv, report, capsys,
):
    database = workspace / "old.sqlite"
    create_old_database(database, cv, intermediate=True)
    with benchmark.connect(database) as connection:
        benchmark.print_report(connection)
        old_output = capsys.readouterr().out
        legacy_line = next(line for line in old_output.splitlines() if line.startswith("legacy"))
        assert legacy_line.split()[-4:] == ["-", "-", "-", "0/1"]
        assert "25.0%" in legacy_line
        measured = result(cv, **benchmark.report_metrics(TailorDiagnostics(report=report)))
        benchmark.save_result(connection, measured)
        benchmark.save_result(connection, replace(
            measured, repetition=2, match_percent=None, must_have_percent=None,
            accepted_rewrites=None, rejected_rewrites=None, unclear_rewrites=None,
            report_json=None,
        ))
        relative = benchmark.relative_selection_recall(connection, "run")
        assert relative["legacy"] == pytest.approx(-16.6666667)
        assert relative["model"] == pytest.approx(8.3333333)
        benchmark.print_report(connection, "run")
    output = capsys.readouterr().out
    model_line = next(line for line in output.splitlines()
                      if line.startswith("model ") and "avg s" not in line)
    assert model_line.split()[-4:] == ["62.5%", "50.0%", "2/1/1", "1/2"]
    assert "source-evidence coverage" in output
    assert "not match metrics or proof of faithfulness" in output
    assert "approved rewrites are not proof" in output
    assert "report_json" in output
