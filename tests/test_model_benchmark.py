import sqlite3
from dataclasses import replace

import pytest

from assistant import model_benchmark as benchmark
from assistant.cv_generator import CV, TailorDiagnostics
from assistant.cv_tailoring import (
    EvidenceItem,
    RequirementMatch,
    RewriteAudit,
    TailoringReport,
)
from assistant.performance import RunPerformance


@pytest.fixture
def cv():
    return CV(name="Candidate", summary="Python engineer", skills=["Python"])


@pytest.fixture
def report():
    return TailoringReport(
        match_percent=62.5,
        must_have_percent=50,
        performance=RunPerformance(
            total_model_calls=4,
            stage_seconds={"matching": 21.0, "rewriting": 12.5},
        ),
        selected_evidence_ids=["skills/0"],
        evidence=[EvidenceItem(id="skills/0", text="Python", section="skills")],
        matches=[
            RequirementMatch(
                requirement_id="r1",
                status="direct",
                evidence_ids=["skills/0"],
                explanation="Source lists Python",
            )
        ],
        rewrites=[
            RewriteAudit(
                target_id=f"skills/{index}",
                source_ids=["skills/0"],
                original="Python",
                proposed="Python development",
                exported="Python development" if status == "accepted" else "Python",
                status=status,
                reason=f"Automated verdict: {status}",
            )
            for index, status in enumerate(
                ("accepted", "accepted", "rejected", "unclear")
            )
        ],
    )


def result(cv, **kwargs):
    return benchmark.BenchmarkResult(
        run_id="run",
        model="model",
        job="job.txt",
        repetition=1,
        status="ok",
        total_seconds=3.5,
        output_json=cv.model_dump_json(),
        **kwargs,
    )


def test_list_models_shows_current_suite(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark,
        "installed_model_info",
        lambda: {"granite4.2:3b": ("digest", "2.2 GB")},
    )
    benchmark.list_models()
    output = capsys.readouterr().out
    for model, size, tier in benchmark.MODEL_SUITE:
        line = next(
            line for line in output.splitlines() if line.startswith(model + " ")
        )
        assert line.split() == [
            model,
            *size.split(),
            *tier.split(),
            "yes" if model == "granite4.2:3b" else "no",
        ]
    assert "Download sizes are not runtime RAM" in output


def test_connect_creates_only_current_result_fields(tmp_path):
    path = tmp_path / "nested" / "benchmark.sqlite"
    for _ in range(2):
        with benchmark.connect(path) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(benchmark_results)")
            }
            assert columns == set(benchmark.BenchmarkResult.__dataclass_fields__)


@pytest.mark.parametrize("command", ["run", "report"])
def test_benchmark_defaults_use_fresh_product_database(command):
    arguments = ["--cv", "cv.pdf", "--jobs", "job.txt"] if command == "run" else []
    args = benchmark.parse_args([command, *arguments])
    assert str(args.database) == "output/benchmarks.sqlite"
    if command == "run":
        assert args.repeat == 1
        assert str(args.job_cache) == "output/job-requirements"
        assert not args.refresh_job_analysis


def test_unsupported_schema_is_rejected_without_modifying_archive(tmp_path):
    path = tmp_path / "archive.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE benchmark_results (keyword_recall REAL)")
        connection.execute("INSERT INTO benchmark_results VALUES (0.75)")
    before = path.read_bytes()
    with pytest.raises(ValueError, match="use --database with a new file"):
        benchmark.connect(path)
    assert path.read_bytes() == before


def test_save_result_persists_evidence_metrics_and_full_report(tmp_path, cv, report):
    stored = result(cv, **benchmark.report_metrics(TailorDiagnostics(report=report)))
    with benchmark.connect(tmp_path / "benchmark.sqlite") as connection:
        benchmark.save_result(connection, stored)
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM benchmark_results").fetchone()
    assert row["output_json"] == cv.model_dump_json()
    assert row["report_json"] == report.model_dump_json()
    assert row["match_percent"] == 62.5
    assert row["must_have_percent"] == 50
    assert (
        row["accepted_rewrites"],
        row["rejected_rewrites"],
        row["unclear_rewrites"],
    ) == (2, 1, 1)
    assert (row["model_calls"], row["matching_seconds"], row["rewriting_seconds"]) == (
        4,
        21,
        12.5,
    )
    assert row["matching_cached"] == 0


def test_failed_case_without_report_keeps_metrics_null(tmp_path, cv):
    failed = replace(
        result(cv), status="error", error="Generation failed", output_json=None
    )
    with benchmark.connect(tmp_path / "benchmark.sqlite") as connection:
        benchmark.save_result(connection, failed)
        row = connection.execute(
            "SELECT status, error, output_json, report_json, model_calls, match_percent "
            "FROM benchmark_results"
        ).fetchone()
    assert row == ("error", "Generation failed", None, None, None, None)


def test_benchmark_one_collects_output_audit_and_uses_job_cache(
    tmp_path, monkeypatch, cv, report
):
    job = tmp_path / "job.txt"
    job.write_text("Python required")
    cache = tmp_path / "jobs"
    calls = []

    def tailor(
        source,
        description,
        *,
        model,
        llm,
        diagnostics,
        job_cache,
        refresh_job_analysis,
    ):
        calls.append(
            (
                source,
                description,
                model,
                llm.num_predict,
                job_cache,
                refresh_job_analysis,
            )
        )
        diagnostics.report = report
        return source

    monkeypatch.setattr(benchmark, "tailor_cv", tailor)
    measured = benchmark.benchmark_one(
        "run", "local-model", job, 2, cv, job_cache=cache
    )
    assert calls == [
        (
            cv,
            "Python required",
            "local-model",
            benchmark.MAX_OUTPUT_TOKENS,
            cache,
            False,
        )
    ]
    assert measured.status == "ok"
    assert measured.output_json == cv.model_dump_json()
    assert measured.total_seconds >= 0
    assert measured.match_percent == 62.5
    assert measured.report_json == report.model_dump_json()
    assert measured.model_calls == 4
    assert measured.matching_seconds == 21
    assert measured.rewriting_seconds == 12.5


def test_empty_report_distinguishes_zero_verdicts_from_missing_report():
    metrics = benchmark.report_metrics(
        TailorDiagnostics(
            report=TailoringReport(
                match_percent=0,
                must_have_percent=None,
            )
        )
    )
    assert metrics["match_percent"] == 0
    assert metrics["must_have_percent"] is None
    for name in (
        "accepted_rewrites",
        "rejected_rewrites",
        "unclear_rewrites",
        "model_calls",
    ):
        assert metrics[name] == 0
    assert benchmark.report_metrics(TailorDiagnostics()) == {}


@pytest.mark.parametrize("with_report", [False, True])
def test_failed_benchmark_preserves_error_and_available_audit(
    tmp_path,
    monkeypatch,
    cv,
    report,
    with_report,
):
    job = tmp_path / "job.txt"
    job.write_text("Python")

    def fail(
        source,
        description,
        *,
        model,
        llm,
        diagnostics,
        job_cache,
        refresh_job_analysis,
    ):
        diagnostics.report = report if with_report else None
        raise RuntimeError("generation failed")

    monkeypatch.setattr(benchmark, "tailor_cv", fail)
    measured = benchmark.benchmark_one("run", "local-model", job, 1, cv)
    assert measured.status == "error"
    assert measured.error == "RuntimeError: generation failed"
    assert measured.output_json is None
    assert measured.report_json == (report.model_dump_json() if with_report else None)


def test_benchmark_matrix_persists_successes_and_failures(
    tmp_path,
    monkeypatch,
    cv,
    report,
):
    jobs = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for job in jobs:
        job.write_text("Python required")
    database = tmp_path / "benchmark.sqlite"
    args = benchmark.parse_args(
        [
            "run",
            "--cv",
            str(tmp_path / "cv.json"),
            "--jobs",
            *(str(job) for job in jobs),
            "--models",
            "first",
            "second",
            "--repeat",
            "2",
            "--database",
            str(database),
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "installed_model_info",
        lambda: {"first": ("digest-1", "1 GB"), "second": ("digest-2", "2 GB")},
    )
    monkeypatch.setattr(benchmark, "command_output", lambda *args: "test-version")
    monkeypatch.setattr(benchmark, "load_cv", lambda path, model: cv)
    unloaded = []
    monkeypatch.setattr(benchmark, "unload_model", unloaded.append)
    cases = []

    def run_case(
        run_id,
        model,
        job,
        repetition,
        source,
        *,
        job_cache,
        refresh_job_analysis,
    ):
        cases.append(
            (model, job.name, repetition, source, job_cache, refresh_job_analysis)
        )
        measured = replace(
            result(cv, **benchmark.report_metrics(TailorDiagnostics(report=report))),
            run_id=run_id,
            model=model,
            job=job.name,
            repetition=repetition,
        )
        if model == "first" and job == jobs[0] and repetition == 1:
            return replace(
                measured,
                status="error",
                output_json=None,
                error="Model unavailable",
            )
        return measured

    monkeypatch.setattr(benchmark, "benchmark_one", run_case)
    benchmark.run_benchmark(args)
    assert [(model, job, repetition) for model, job, repetition, _, _, _ in cases] == [
        (model, job.name, repetition)
        for model in ["first", "second"]
        for job in jobs
        for repetition in [1, 2]
    ]
    assert all(
        source == cv and cache == args.job_cache and not refresh
        for _, _, _, source, cache, refresh in cases
    )
    assert unloaded == ["first", "second"]
    with sqlite3.connect(database) as connection:
        assert connection.execute(
            "SELECT context_tokens FROM benchmark_runs"
        ).fetchone() == (benchmark.CONTEXT_TOKENS,)
        assert connection.execute(
            "SELECT status, COUNT(*) FROM benchmark_results GROUP BY status ORDER BY status"
        ).fetchall() == [("error", 1), ("ok", 7)]
        assert connection.execute(
            "SELECT model, model_id FROM benchmark_models ORDER BY model"
        ).fetchall() == [("first", "digest-1"), ("second", "digest-2")]


def test_refreshes_each_job_until_success_then_reuses_analysis(
    tmp_path,
    monkeypatch,
    cv,
):
    jobs = [tmp_path / "first.txt", tmp_path / "second.txt"]
    for job in jobs:
        job.write_text("Python required")
    args = benchmark.parse_args(
        [
            "run",
            "--cv",
            str(tmp_path / "cv.json"),
            "--jobs",
            *(str(job) for job in jobs),
            "--models",
            "first",
            "second",
            "--repeat",
            "2",
            "--database",
            str(tmp_path / "benchmark.sqlite"),
            "--refresh-job-analysis",
        ]
    )
    monkeypatch.setattr(
        benchmark,
        "installed_model_info",
        lambda: {"first": ("digest-1", "1 GB"), "second": ("digest-2", "2 GB")},
    )
    monkeypatch.setattr(benchmark, "command_output", lambda *args: "test-version")
    monkeypatch.setattr(benchmark, "load_cv", lambda path, model: cv)
    monkeypatch.setattr(benchmark, "unload_model", lambda model: None)
    calls = []

    def run_case(
        run_id,
        model,
        job,
        repetition,
        source,
        *,
        job_cache,
        refresh_job_analysis,
    ):
        calls.append((model, job.name, repetition, refresh_job_analysis))
        if job == jobs[0] and repetition == 1 and model == "first":
            return benchmark.BenchmarkResult(
                run_id,
                model,
                job.name,
                repetition,
                "error",
                0.1,
                error="Job parse failed",
            )
        return benchmark.BenchmarkResult(
            run_id,
            model,
            job.name,
            repetition,
            "ok",
            0.1,
            model_calls=1,
            matching_seconds=0.05,
            rewriting_seconds=0.05,
        )

    monkeypatch.setattr(benchmark, "benchmark_one", run_case)
    benchmark.run_benchmark(args)
    assert calls == [
        ("first", "first.txt", 1, True),
        ("first", "first.txt", 2, True),
        ("first", "second.txt", 1, True),
        ("first", "second.txt", 2, False),
        ("second", "first.txt", 1, False),
        ("second", "first.txt", 2, False),
        ("second", "second.txt", 1, False),
        ("second", "second.txt", 2, False),
    ]


def test_report_shows_current_evidence_metrics(tmp_path, cv, report, capsys):
    measured = result(cv, **benchmark.report_metrics(TailorDiagnostics(report=report)))
    with benchmark.connect(tmp_path / "benchmark.sqlite") as connection:
        connection.execute(
            "INSERT INTO benchmark_runs VALUES (?, ?, ?, ?, ?)",
            ("run", "2026-10-01", "test", "cv.pdf", 16384),
        )
        benchmark.save_result(connection, measured)
        benchmark.print_report(connection)
    output = capsys.readouterr().out
    line = next(
        line
        for line in output.splitlines()
        if line.startswith("model ") and "avg s" not in line
    )
    assert line.split()[-8:] == [
        "62.5%",
        "50.0%",
        "2/1/1",
        "1/1",
        "4.0",
        "21.0",
        "12.5",
        "0",
    ]
    assert "source-evidence coverage" in output
    assert "approved rewrites are not proof" in output
    assert "report_json" in output
    assert "kw proxy" not in output


def test_report_does_not_treat_failed_cases_as_zero_coverage(tmp_path, cv, capsys):
    failed = replace(result(cv), status="error", output_json=None, error="Failed")
    with benchmark.connect(tmp_path / "benchmark.sqlite") as connection:
        benchmark.save_result(connection, failed)
        benchmark.print_report(connection, "run")
    output = capsys.readouterr().out
    line = next(
        line
        for line in output.splitlines()
        if line.startswith("model ") and "avg s" not in line
    )
    assert "0/1" in line and "0.0%" not in line
