"""Per-run measurements without coupling model helpers to the CLI."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter

from pydantic import BaseModel, Field


class RunPerformance(BaseModel):
    total_seconds: float = 0.0
    total_model_calls: int = 0
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    stage_model_calls: dict[str, int] = Field(default_factory=dict)
    stage_model_seconds: dict[str, float] = Field(default_factory=dict)


_current: ContextVar[RunPerformance | None] = ContextVar("cv_performance", default=None)


@contextmanager
def measure_run():
    performance = RunPerformance()
    token = _current.set(performance)
    started = perf_counter()
    try:
        yield performance
    finally:
        performance.total_seconds = perf_counter() - started
        _current.reset(token)


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
def measure_model_call(stage: str):
    started = perf_counter()
    performance = _current.get()
    if performance is not None:
        performance.total_model_calls += 1
        performance.stage_model_calls[stage] = (
            performance.stage_model_calls.get(stage, 0) + 1
        )
    try:
        yield
    finally:
        if performance is not None:
            performance.stage_model_seconds[stage] = (
                performance.stage_model_seconds.get(stage, 0.0) + perf_counter() - started
            )
