"""Benchmark local Ollama models on the CV-tailoring workload."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from assistant.cv_generator import CV, TailorDiagnostics, load_cv, tailor_cv

DEFAULT_DATABASE = Path("output/model-benchmarks.sqlite")
DEFAULT_JOB_CACHE = Path("output/job-requirements")
MODEL_SUITE = (
    ("qwen3.5:0.8b", "1.0 GB", "basic"),
    ("granite4.2:3b", "2.2 GB", "basic"),
    ("granite4.2:8b", "5.3 GB", "balanced"),
    ("lfm2.5:8b", "5.2 GB", "balanced"),
    ("gemma4:12b", "8.0 GB", "advanced"),
    ("gemma4:26b-a4b", "18 GB", "advanced"),
)
WORD = re.compile(r"[a-z][a-z0-9+#.-]{2,}")
STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "have",
    "into",
    "our",
    "that",
    "the",
    "their",
    "this",
    "using",
    "with",
    "your",
}


@dataclass(frozen=True)
class BenchmarkResult:
    run_id: str
    model: str
    job: str
    repetition: int
    status: str
    total_seconds: float
    diagnostics: TailorDiagnostics
    keyword_recall: float | None = None
    selection_recall: float | None = None
    summary_words: int | None = None
    selected_items: int | None = None
    output_json: str | None = None
    error: str | None = None
    match_percent: float | None = None
    must_have_percent: float | None = None
    accepted_rewrites: int | None = None
    rejected_rewrites: int | None = None
    unclear_rewrites: int | None = None
    report_json: str | None = None


def connect(path: Path) -> sqlite3.Connection:
    """Open the benchmark database and ensure its schema exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
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
            selection_seconds REAL NOT NULL,
            summary_seconds REAL NOT NULL,
            summary_attempts INTEGER NOT NULL,
            summary_fallback INTEGER NOT NULL,
            keyword_recall REAL,
            selection_recall REAL,
            summary_words INTEGER,
            selected_items INTEGER,
            output_json TEXT,
            error TEXT,
            match_percent REAL,
            must_have_percent REAL,
            accepted_rewrites INTEGER,
            rejected_rewrites INTEGER,
            unclear_rewrites INTEGER,
            report_json TEXT,
            PRIMARY KEY (run_id, model, job, repetition),
            FOREIGN KEY (run_id) REFERENCES benchmark_runs(run_id)
        )
        """
    )
    existing = {
        row[1] for row in connection.execute("PRAGMA table_info(benchmark_results)")
    }
    for column, kind in (
        ("selection_recall", "REAL"),
        ("summary_words", "INTEGER"),
        ("match_percent", "REAL"),
        ("must_have_percent", "REAL"),
        ("accepted_rewrites", "INTEGER"),
        ("rejected_rewrites", "INTEGER"),
        ("unclear_rewrites", "INTEGER"),
        ("report_json", "TEXT"),
    ):
        if column not in existing:
            connection.execute(
                f"ALTER TABLE benchmark_results ADD COLUMN {column} {kind}"
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


def keywords(text: str) -> set[str]:
    """Extract coarse technical and domain terms for an automatic recall proxy."""
    return {word for word in WORD.findall(text.casefold()) if word not in STOPWORDS}


def selected_text(cv: CV) -> str:
    """Text whose relevance was controlled by the tailoring model."""
    parts = [cv.summary, *cv.skills, *cv.ai_native]
    parts.extend(bullet for role in cv.experience for bullet in role.bullets)
    return " ".join(parts)


def selection_text(cv: CV) -> str:
    """Exported list items, excluding the summary.

    The summary is free text, so a model that writes at length scores higher
    on plain keyword overlap without choosing better evidence. Items may be
    rewritten after evidence selection, and pool size varies with the source CV.
    This text is a keyword proxy, not a check of source support or faithfulness.
    """
    parts = [*cv.skills, *cv.ai_native]
    parts.extend(bullet for role in cv.experience for bullet in role.bullets)
    return " ".join(parts)


def keyword_recall(job_description: str, cv: CV) -> float:
    """Keyword-overlap proxy across exported items and the summary."""
    required = keywords(job_description)
    return len(required & keywords(selected_text(cv))) / len(required) if required else 0.0


def selection_recall(job_description: str, cv: CV) -> float:
    """Keyword-overlap proxy across exported list items alone."""
    required = keywords(job_description)
    if not required:
        return 0.0
    return len(required & keywords(selection_text(cv))) / len(required)


def summary_word_count(cv: CV) -> int:
    """Length of the generated summary, to expose length-driven recall."""
    return len(cv.summary.split())


def selected_item_count(cv: CV) -> int:
    """Count model-selected list items in a tailored CV."""
    return (
        len(cv.skills)
        + len(cv.ai_native)
        + sum(len(role.bullets) for role in cv.experience)
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
    }


def save_result(connection: sqlite3.Connection, result: BenchmarkResult) -> None:
    """Persist one completed or failed model/job evaluation."""
    connection.execute(
        """
        INSERT INTO benchmark_results (
            run_id, model, job, repetition, status, total_seconds,
            selection_seconds, summary_seconds, summary_attempts,
            summary_fallback, keyword_recall, selection_recall, summary_words,
            selected_items, output_json, error, match_percent, must_have_percent,
            accepted_rewrites, rejected_rewrites, unclear_rewrites, report_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.run_id,
            result.model,
            result.job,
            result.repetition,
            result.status,
            result.total_seconds,
            result.diagnostics.selection_seconds,
            result.diagnostics.summary_seconds,
            result.diagnostics.summary_attempts,
            result.diagnostics.summary_fallback,
            result.keyword_recall,
            result.selection_recall,
            result.summary_words,
            result.selected_items,
            result.output_json,
            result.error,
            result.match_percent,
            result.must_have_percent,
            result.accepted_rewrites,
            result.rejected_rewrites,
            result.unclear_rewrites,
            result.report_json,
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
    job_description = job_path.read_text(encoding="utf-8")
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
            diagnostics=diagnostics,
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
        diagnostics=diagnostics,
        keyword_recall=keyword_recall(job_description, tailored),
        selection_recall=selection_recall(job_description, tailored),
        summary_words=summary_word_count(tailored),
        selected_items=selected_item_count(tailored),
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
    cv = load_cv(args.cv)
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
                        f"keyword proxy {result.keyword_recall:.1%}, "
                        f"{result.diagnostics.summary_attempts} summary attempt(s)"
                        + (", source summary kept" if result.diagnostics.summary_fallback else "")
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
               AVG(CASE WHEN status = 'ok' THEN selection_recall END) AS recall,
               AVG(CASE WHEN status = 'ok' THEN summary_words END) AS words,
               SUM(summary_fallback) AS fallbacks,
               AVG(CASE WHEN status = 'ok' THEN summary_attempts END) AS attempts,
               AVG(CASE WHEN status = 'ok' THEN match_percent END) AS match_percent,
               AVG(CASE WHEN status = 'ok' THEN must_have_percent END) AS must_have_percent,
               SUM(CASE WHEN status = 'ok' THEN accepted_rewrites END) AS accepted,
               SUM(CASE WHEN status = 'ok' THEN rejected_rewrites END) AS rejected,
               SUM(CASE WHEN status = 'ok' THEN unclear_rewrites END) AS unclear,
               SUM(status = 'ok' AND report_json IS NOT NULL) AS reports
        FROM benchmark_results
        WHERE run_id = ?
        GROUP BY model
        """,
        (run_id,),
    ).fetchall()
    diversity = model_diversity(connection, run_id)
    relative = relative_selection_recall(connection, run_id)
    rows = sorted(
        rows,
        key=lambda row: (-row[2], -(relative.get(row[0]) or -1.0), row[3] or 0.0),
    )
    print(f"\nBenchmark run: {run_id}")
    print(
        "model                         pass      avg s  kw proxy    vs job  words  "
        "diversity  fallback  attempts   match  must-have  rewrites A/R/U  reports"
    )
    print("-" * 159)
    for (
        model, cases, successes, seconds, recall, words, fallbacks, attempts,
        match, must_have, accepted, rejected, unclear, reports,
    ) in rows:
        duration = f"{seconds:.1f}" if seconds is not None else "-"
        recall_text = f"{recall:.1%}" if recall is not None else "-"
        edge = relative.get(model)
        edge_text = f"{edge:+.1f}pp" if edge is not None else "-"
        words_text = f"{words:.0f}" if words is not None else "-"
        diversity_text = (
            f"{diversity[model]:.1%}" if diversity.get(model) is not None else "-"
        )
        attempts_text = f"{attempts:.2f}" if attempts is not None else "-"
        match_text = f"{match:.1f}%" if match is not None else "-"
        must_have_text = f"{must_have:.1f}%" if must_have is not None else "-"
        rewrite_text = (
            f"{accepted}/{rejected}/{unclear}" if accepted is not None else "-"
        )
        print(
            f"{model:<29} {successes:>2}/{cases:<2} {duration:>10} "
            f"{recall_text:>8} {edge_text:>9} {words_text:>6} {diversity_text:>10} "
            f"{fallbacks:>9} {attempts_text:>9} {match_text:>7} {must_have_text:>10} "
            f"{rewrite_text:>15} {reports:>3}/{successes}"
        )
    print(
        "\nkw proxy = job-term overlap in exported list items (summary excluded).\n"
        "vs job = mean keyword-proxy gap to the per-job average, retaining the\n"
        "         relative comparison across jobs of unequal keyword difficulty.\n"
        "Keyword proxies are not match metrics or proof of faithfulness.\n"
        "match / must-have = source-evidence coverage of job requirements, not\n"
        "model quality; averages exclude missing values, shown as '-'.\n"
        "rewrites A/R/U = accepted/rejected/unclear automated verdict totals;\n"
        "approved rewrites are not proof of faithfulness. reports = audited/pass.\n"
        "Full evidence, requirement matches and rewrite audits are in report_json."
    )


def relative_selection_recall(
    connection: sqlite3.Connection, run_id: str
) -> dict[str, float | None]:
    """Each model's mean keyword-proxy gap to the per-job average, in points.

    Comparing against the same job reduces keyword-difficulty effects; it does
    not establish model quality or source faithfulness.
    """
    rows = connection.execute(
        """
        SELECT model, job, selection_recall
        FROM benchmark_results
        WHERE run_id = ? AND status = 'ok' AND selection_recall IS NOT NULL
        """,
        (run_id,),
    ).fetchall()
    per_job: dict[str, list[float]] = {}
    for _model, job, recall in rows:
        per_job.setdefault(job, []).append(recall)
    averages = {job: sum(v) / len(v) for job, v in per_job.items()}
    gaps: dict[str, list[float]] = {}
    for model, job, recall in rows:
        gaps.setdefault(model, []).append((recall - averages[job]) * 100)
    models = {model for model, *_ in rows}
    return {
        model: sum(gaps[model]) / len(gaps[model]) if gaps.get(model) else None
        for model in models
    }


def model_diversity(
    connection: sqlite3.Connection, run_id: str
) -> dict[str, float | None]:
    """Average dissimilarity between a model's outputs for different jobs."""
    rows = connection.execute(
        """
        SELECT model, job, repetition, output_json
        FROM benchmark_results
        WHERE run_id = ? AND status = 'ok'
        ORDER BY model, repetition, job
        """,
        (run_id,),
    ).fetchall()
    grouped: dict[tuple[str, int], list[set[str]]] = {}
    for model, _job, repetition, output_json in rows:
        cv = CV.model_validate_json(output_json)
        signature = set(cv.skills) | set(cv.ai_native)
        signature.update(bullet for role in cv.experience for bullet in role.bullets)
        grouped.setdefault((model, repetition), []).append(signature)
    values: dict[str, list[float]] = {}
    for (model, _repetition), signatures in grouped.items():
        for index, left in enumerate(signatures):
            for right in signatures[index + 1 :]:
                union = left | right
                distance = 1 - len(left & right) / len(union) if union else 0.0
                values.setdefault(model, []).append(distance)
    models = {model for model, *_ in rows}
    return {
        model: sum(values[model]) / len(values[model]) if values.get(model) else None
        for model in models
    }


def list_models() -> None:
    """Display the curated model ladder and local installation status."""
    available = installed_model_info()
    print("model                         size      tier       installed")
    print("-" * 67)
    for model, size, tier in MODEL_SUITE:
        print(f"{model:<29} {size:<9} {tier:<10} {'yes' if model in available else 'no'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    models_parser = subparsers.add_parser("models", help="show the curated model suite")
    models_parser.set_defaults(handler=lambda args: list_models())

    run_parser = subparsers.add_parser("run", help="run and persist benchmarks")
    run_parser.add_argument("--cv", type=Path, required=True)
    run_parser.add_argument("--jobs", type=Path, nargs="+", required=True)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if getattr(args, "repeat", 1) < 1:
        raise SystemExit("--repeat must be at least 1.")
    args.handler(args)


if __name__ == "__main__":
    main()
