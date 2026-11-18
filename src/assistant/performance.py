"""Per-run measurements without coupling model helpers to the CLI."""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
import uuid

from pydantic import BaseModel, Field


class RunPerformance(BaseModel):
    total_seconds: float = 0.0
    total_model_calls: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    stage_model_calls: dict[str, int] = Field(default_factory=dict)
    stage_model_seconds: dict[str, float] = Field(default_factory=dict)


_current: ContextVar[RunPerformance | None] = ContextVar("cv_performance", default=None)
_log_path: ContextVar[Path | None] = ContextVar("cv_call_log_path", default=None)
_run_id: ContextVar[str | None] = ContextVar("cv_call_log_run_id", default=None)


@contextmanager
def measure_run(log_path: Path | None = Path("output/model-calls.jsonl")):
    performance = RunPerformance()
    token = _current.set(performance)
    log_token = _log_path.set(log_path)
    run_token = _run_id.set(uuid.uuid4().hex)
    started = perf_counter()
    try:
        yield performance
    finally:
        performance.total_seconds = perf_counter() - started
        _current.reset(token)
        _log_path.reset(log_token)
        _run_id.reset(run_token)


@contextmanager
def measure_stage(stage: str):
    started = perf_counter()
    try:
        yield
    finally:
        performance = _current.get()
        if performance is not None:
            performance.stage_seconds[stage] = (
                performance.stage_seconds.get(stage, 0.0) + perf_counter() - started
            )


@contextmanager
def measure_model_call(stage: str, attempt: int | None = None):
    started = perf_counter()
    performance = _current.get()
    if performance is not None:
        performance.total_model_calls += 1
        performance.stage_model_calls[stage] = (
            performance.stage_model_calls.get(stage, 0) + 1
        )
    outcome = "ok"
    error_type = None
    try:
        yield
    except Exception as exc:
        outcome = "error"
        error_type = type(exc).__name__
        raise
    finally:
        elapsed = perf_counter() - started
        if performance is not None:
            performance.stage_model_seconds[stage] = (
                performance.stage_model_seconds.get(stage, 0.0)
                + elapsed
            )
        path = _log_path.get()
        if path is not None:
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": _run_id.get(),
                "stage": stage,
                "elapsed_seconds": round(elapsed, 3),
                "outcome": outcome,
            }
            if attempt is not None:
                record["attempt"] = attempt
            if error_type is not None:
                record["error_type"] = error_type
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=True) + "\n")
