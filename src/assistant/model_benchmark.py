"""Benchmark local Ollama models on the CV-tailoring workload."""

from __future__ import annotations

import argparse
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from assistant.cv_generator import CV, TailorDiagnostics, load_cv, tailor_cv
from assistant.cv_parser import read_document

DEFAULT_DATABASE = Path("output/benchmarks.sqlite")
DEFAULT_JOB_CACHE = Path("output/job-requirements")
MODEL_SUITE = (
    ("qwen3.5:0.8b", "1.0 GB", "basic"),
    ("lfm2.5-thinking:1.2b", "731 MB", "basic challenger"),
    ("granite4.2:3b", "2.2 GB", "baseline"),
    ("qwen3.5:2b-q4_K_M", "1.9 GB", "challenger"),
    ("openbmb/minicpm5-2b:2b", "1.6 GB", "priority challenger"),
    ("nemotron-3-nano:4b", "2.8 GB", "challenger"),
    ("gemma4:e2b-it-qat", "4.3 GB", "current winner"),
)


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    model: str
    job: str
    repetition: int
    status: str
    total_seconds: float
    output_json: str | None = None
    error: str | None = None
    match_percent: float | None = None
    must_have_percent: float | None = None
    accepted_rewrites: int | None = None
    rejected_rewrites: int | None = None
    unclear_rewrites: int | None = None
    report_json: str | None = None
    model_calls: int | None = None
    matching_seconds: float | None = None
    rewriting_seconds: float | None = None
    matching_cached: bool | None = None


def connect(path: Path) -> sqlite3.Connection:
    """Open the benchmark database and ensure its schema exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    existing = {
        row[1] for row in connection.execute("PRAGMA table_info(benchmark_results)")
    }
    if existing and existing != set(BenchmarkResult.__dataclass_fields__):
        connection.close()
        raise ValueError(
            f"Unsupported benchmark database schema: {path}. "
            "Keep this archive and use --database with a new file."
        )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            ollama_version TEXT NOT NULL,
            cv_path TEXT NOT NULL,
            context_tokens INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_results (
            run_id TEXT NOT NULL,
            model TEXT NOT NULL,
            job TEXT NOT NULL,
            repetition INTEGER NOT NULL,
            status TEXT NOT NULL,
            total_seconds REAL NOT NULL,
            output_json TEXT,
            error TEXT,
            match_percent REAL,
            must_have_percent REAL,
            accepted_rewrites INTEGER,
            rejected_rewrites INTEGER,
            unclear_rewrites INTEGER,
            report_json TEXT,
            model_calls INTEGER,
            matching_seconds REAL,
            rewriting_seconds REAL,
            matching_cached INTEGER,
            PRIMARY KEY (run_id, model, job, repetition),
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS benchmark_models (
            run_id TEXT NOT NULL,
            model TEXT NOT NULL,
            model_id TEXT NOT NULL,
            size TEXT NOT NULL,
            PRIMARY KEY (run_id, model),
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
        )
        """
    )
    connection.commit()
    return connection


def command_output(*command: str) -> str:
    """Run a small local command and return its output."""
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip() or completed.stderr.strip()


def installed_model_info() -> dict[str, tuple[str, str]]:
    """Return installed Ollama model names with their digest and size."""
    lines = command_output("ollama", "list").splitlines()[1:]
    info = {}
    for line in lines:
        columns = line.split()
        if len(columns) >= 4:
            info[columns[0]] = (columns[1], f"{columns[2]} {columns[3]}")
    return info


def pull_model(model: str) -> None:
    """Install one model through Ollama."""
    subprocess.run(["ollama", "pull", model], check=True)


def unload_model(model: str) -> None:
    """Evict a model from memory so the next one starts from a clean state.

    Ollama otherwise keeps each model resident for five minutes, which leaves
    several ``llama-server`` processes holding unified memory at once.
    """
    subprocess.run(
        ["ollama", "stop", model],
        check=False,
        capture_output=True,
    )


def report_metrics(diagnostics: TailorDiagnostics) -> dict:
    """Collect source-coverage metrics and preserve the complete audit report."""
    report = diagnostics.report
    if report is None:
        return {}
    return {
        "match_percent": report.match_percent,
        "must_have_percent": report.must_have_percent,
        "accepted_rewrites": sum(item.status == "accepted" for item in report.rewrites),
        "rejected_rewrites": sum(item.status == "rejected" for item in report.rewrites),
        "unclear_rewrites": sum(item.status == "unclear" for item in report.rewrites),
        "report_json": report.model_dump_json(),
        "model_calls": report.performance.total_model_calls,
        "matching_seconds": report.performance.stage_seconds.get("matching", 0.0),
        "rewriting_seconds": report.performance.stage_seconds.get("rewriting", 0.0),
        "matching_cached": report.matching_analysis_cached,
    }


def save_result(connection: sqlite3.Connection, result: BenchmarkResult) -> None:
    """Persist one completed or failed model/job evaluation."""
    connection.execute(
        """
        INSERT INTO benchmark_results (
            run_id, model, job, repetition, status, total_seconds,
            output_json, error, match_percent, must_have_percent,
            accepted_rewrites, rejected_rewrites, unclear_rewrites, report_json,
            model_calls, matching_seconds, rewriting_seconds, matching_cached
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.run_id,
            result.model,
            result.job,
            result.repetition,
            result.status,
            result.total_seconds,
            result.output_json,
            result.error,
            result.match_percent,
            result.must_have_percent,
            result.accepted_rewrites,
            result.rejected_rewrites,
            result.unclear_rewrites,
            result.report_json,
            result.model_calls,
            result.matching_seconds,
            result.rewriting_seconds,
            result.matching_cached,
        ),
    )
    connection.commit()


def benchmark_one(
    run_id: str,
    model: str,
    job_path: Path,
    repetition: int,
    cv: CV,
    *,
    job_cache: Path | None = None,
) -> BenchmarkResult:
    """Run one model against one job and collect workload-specific metrics."""
    job_description = read_document(job_path)
    diagnostics = TailorDiagnostics()
    started = time.perf_counter()
    try:
        tailored = tailor_cv(
            cv,
            job_description,
            model=model,
            diagnostics=diagnostics,
            job_cache=job_cache,
        )
    except Exception as exc:
        return BenchmarkResult(
            run_id=run_id,
            model=model,
            job=job_path.name,
            repetition=repetition,
            status="error",
            total_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
            **report_metrics(diagnostics),
        )
    return BenchmarkResult(
        run_id=run_id,
        model=model,
        job=job_path.name,
        repetition=repetition,
        status="ok",
        total_seconds=time.perf_counter() - started,
        output_json=tailored.model_dump_json(),
        **report_metrics(diagnostics),
    )


def run_benchmark(args: argparse.Namespace) -> None:
    """Execute and persist a benchmark matrix."""
    missing_jobs = [job for job in args.jobs if not job.is_file()]
    if missing_jobs:
        names = ", ".join(str(job) for job in missing_jobs)
        raise SystemExit(f"Job descriptions not found: {names}")
    names_seen = [job.name for job in args.jobs]
    duplicates = sorted({name for name in names_seen if names_seen.count(name) > 1})
    if duplicates:
        raise SystemExit(
            f"Job descriptions given more than once: {', '.join(duplicates)}"
        )
    models = args.models or [model for model, _, _ in MODEL_SUITE]
    available = installed_model_info()
    missing = [model for model in models if model not in available]
    if missing and not args.pull:
        names = ", ".join(missing)
        raise SystemExit(f"Models not installed: {names}. Re-run with --pull.")
    for model in missing:
        pull_model(model)
    available = installed_model_info()

    run_id = (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    connection = connect(args.database)
    connection.execute(
        "INSERT INTO benchmark_runs VALUES (?, ?, ?, ?, ?)",
        (
            run_id,
            datetime.now(UTC).isoformat(),
            command_output("ollama", "--version"),
            str(args.cv),
            16384,
        ),
    )
    connection.executemany(
        "INSERT INTO benchmark_models VALUES (?, ?, ?, ?)",
        [
            (run_id, model, available[model][0], available[model][1])
            for model in models
        ],
    )
    connection.commit()
    cv = load_cv(args.cv, model=models[0])
    if args.cv.suffix.lower() == ".pdf":
        source_path = args.database.parent / f"{run_id}.source.json"
        source_path.write_text(cv.model_dump_json(indent=2), encoding="utf-8")
        print(f"PDF imported once for all models; review extracted facts in {source_path}.")
        unload_model(models[0])
    total = len(models) * len(args.jobs) * args.repeat
    position = 0
    for model in models:
        for job in args.jobs:
            for repetition in range(1, args.repeat + 1):
                position += 1
                print(f"[{position}/{total}] {model} / {job.name} / run {repetition}")
                result = benchmark_one(
                    run_id, model, job, repetition, cv, job_cache=args.job_cache
                )
                save_result(connection, result)
                if result.status == "ok":
                    print(
                        f"  {result.total_seconds:.1f}s, "
                        f"{result.model_calls} model call(s), "
                        f"{result.matching_seconds:.1f}s matching, "
                        f"{result.rewriting_seconds:.1f}s rewriting"
                    )
                else:
                    print(f"  ERROR: {result.error}")
        unload_model(model)
    print(f"Run {run_id} saved to {args.database}")
    print_report(connection, run_id)


def print_report(connection: sqlite3.Connection, run_id: str | None = None) -> None:
    """Print aggregate results for a run, defaulting to the latest."""
    if run_id is None:
        row = connection.execute(
            "SELECT run_id FROM benchmark_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise SystemExit("No benchmark runs found.")
        run_id = row[0]
    rows = connection.execute(
        """
        SELECT model,
               COUNT(*) AS cases,
               SUM(status = 'ok') AS successes,
               AVG(CASE WHEN status = 'ok' THEN total_seconds END) AS seconds,
               AVG(CASE WHEN status = 'ok' THEN match_percent END) AS match_percent,
               AVG(CASE WHEN status = 'ok' THEN must_have_percent END) AS must_have_percent,
               SUM(CASE WHEN status = 'ok' THEN accepted_rewrites END) AS accepted,
               SUM(CASE WHEN status = 'ok' THEN rejected_rewrites END) AS rejected,
               SUM(CASE WHEN status = 'ok' THEN unclear_rewrites END) AS unclear,
               SUM(status = 'ok' AND report_json IS NOT NULL) AS reports,
               AVG(CASE WHEN status = 'ok' THEN model_calls END) AS model_calls,
               AVG(CASE WHEN status = 'ok' THEN matching_seconds END) AS matching_seconds,
               AVG(CASE WHEN status = 'ok' THEN rewriting_seconds END) AS rewriting_seconds,
               SUM(CASE WHEN status = 'ok' THEN matching_cached END) AS cached_matches
        FROM benchmark_results
        WHERE run_id = ?
        GROUP BY model
        """,
        (run_id,),
    ).fetchall()
    rows = sorted(
        rows,
        key=lambda row: (-row[2] / row[1], row[3] if row[3] is not None else float("inf")),
    )
    print(f"\nBenchmark run: {run_id}")
    print(
        "model                         pass      avg s   match  must-have  rewrites A/R/U  reports  "
        "calls  match s  rewrite s  cached"
    )
    print("-" * 140)
    for (
        model, cases, successes, seconds,
        match, must_have, accepted, rejected, unclear, reports,
        calls, matching, rewriting, cached,
    ) in rows:
        duration = f"{seconds:.1f}" if seconds is not None else "-"
        match_text = f"{match:.1f}%" if match is not None else "-"
        must_have_text = f"{must_have:.1f}%" if must_have is not None else "-"
        rewrite_text = (
            f"{accepted}/{rejected}/{unclear}" if accepted is not None else "-"
        )
        calls_text = f"{calls:.1f}" if calls is not None else "-"
        matching_text = f"{matching:.1f}" if matching is not None else "-"
        rewriting_text = f"{rewriting:.1f}" if rewriting is not None else "-"
        cached_text = str(cached) if cached is not None else "-"
        print(
            f"{model:<29} {successes:>2}/{cases:<2} {duration:>10} "
            f"{match_text:>7} {must_have_text:>10} "
            f"{rewrite_text:>15} {reports:>3}/{successes} {calls_text:>6} "
            f"{matching_text:>8} {rewriting_text:>10} {cached_text:>7}"
        )
    print(
        "\nmatch / must-have = source-evidence coverage of job requirements, not\n"
        "model quality; averages exclude missing values, shown as '-'.\n"
        "rewrites A/R/U = accepted/rejected/unclear automated verdict totals;\n"
        "approved rewrites are not proof of faithfulness. reports = audited/pass.\n"
        "calls/match s/rewrite s = average local model calls and stage time;\n"
        "cached = results reusing completed matching (a cold benchmark has 0).\n"
        "Full evidence, requirement matches and rewrite audits are in report_json."
    )


def list_models() -> None:
    """Display the curated model ladder and local installation status."""
    available = installed_model_info()
    print(f"{'model':<44} {'download':<9} {'tier':<10} installed")
    print("-" * 74)
    for model, size, tier in MODEL_SUITE:
        print(f"{model:<44} {size:<9} {tier:<10} {'yes' if model in available else 'no'}")
    print(
        "\nDownload sizes are not runtime RAM. This shortlist targets 8 GB total RAM;\n"
        "16K context, runtime buffers and the OS need additional memory.\n"
        "Candidate peak memory and swap use must be measured before deployment."
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="job-assistant benchmark", description=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="show the curated model suite")
    models_parser.set_defaults(handler=lambda args: list_models())

    run_parser = subparsers.add_parser("run", help="run and persist benchmarks")
    run_parser.add_argument("--cv", type=Path, required=True, help="Source CV PDF or JSON")
    run_parser.add_argument("--jobs", type=Path, nargs="+", required=True, help="Job TXT/PDF files")
    run_parser.add_argument("--models", nargs="+")
    run_parser.add_argument("--repeat", type=int, default=1)
    run_parser.add_argument("--pull", action="store_true")
    run_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    run_parser.add_argument("--job-cache", type=Path, default=DEFAULT_JOB_CACHE)
    run_parser.set_defaults(handler=run_benchmark)

    report_parser = subparsers.add_parser("report", help="report a stored benchmark")
    report_parser.add_argument("--run-id")
    report_parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    report_parser.set_defaults(
        handler=lambda args: print_report(connect(args.database), args.run_id)
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if getattr(args, "repeat", 1) < 1:
        raise SystemExit("--repeat must be at least 1.")
    args.handler(args)
