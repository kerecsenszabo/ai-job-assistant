"""Per-run measurements without coupling model helpers to the CLI."""

from contextlib import contextmanager
from collections.abc import Mapping
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
_benchmark_context: ContextVar[dict[str, object]] = ContextVar(
    "cv_benchmark_log_context", default={}
)
_active_usage: ContextVar[dict[str, object] | None] = ContextVar(
    "cv_active_model_usage", default=None
)

PRICING_SOURCE = "Ollama pricing page, checked 2026-11-18"
_BASE_RATES_PER_MILLION = {
    "gemma4": {"input": 0.14, "cached_input": 0.05, "output": 0.40},
    "gpt-oss:120b": {"input": 0.15, "cached_input": 0.014, "output": 0.60},
}
_DEEPSEEK_V41_FLASH_RATES = {
    "standard": {"input": 0.30, "cached_input": 0.006, "output": 1.20},
    "off_peak": {"input": 0.15, "cached_input": 0.003, "output": 0.60},
}


@contextmanager
def benchmark_call_context(
    *, benchmark_run_id: str, model: str, job: str, repetition: int, cloud: bool
):
    """Attach benchmark identifiers to every model-call log event in this case."""
    token = _benchmark_context.set(
        {
            "benchmark_run_id": benchmark_run_id,
            "model": model,
            "job": job,
            "repetition": repetition,
            "cloud": cloud,
        }
    )
    try:
        yield
    finally:
        _benchmark_context.reset(token)


def record_model_usage(usage: object, response_metadata: object = None) -> None:
    """Capture Ollama/LangChain token metadata for the active request."""
    active = _active_usage.get()
    if active is None:
        return
    usage_data = usage if isinstance(usage, Mapping) else {}
    response_data = response_metadata if isinstance(response_metadata, Mapping) else {}
    input_details = usage_data.get("input_token_details", {})
    output_details = usage_data.get("output_token_details", {})
    if not isinstance(input_details, Mapping):
        input_details = {}
    if not isinstance(output_details, Mapping):
        output_details = {}

    input_tokens = usage_data.get("input_tokens")
    output_tokens = usage_data.get("output_tokens")
    token_usage_source = "langchain_usage_metadata"
    if input_tokens is None:
        input_tokens = response_data.get("prompt_eval_count")
        token_usage_source = "ollama_response_metadata"
    if output_tokens is None:
        output_tokens = response_data.get("eval_count")
        if token_usage_source == "langchain_usage_metadata":
            token_usage_source = "ollama_response_metadata"
    if input_tokens is None and output_tokens is None:
        return
    active["token_usage_source"] = token_usage_source
    active["input_tokens"] = input_tokens
    active["output_tokens"] = output_tokens
    active["total_tokens"] = (
        input_tokens + output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int)
        else usage_data.get("total_tokens")
    )
    cached = input_details.get("cache_read", input_details.get("cached_tokens", 0))
    if isinstance(cached, int) and cached >= 0:
        active["cached_input_tokens"] = cached
    reasoning = output_details.get("reasoning", output_details.get("reasoning_tokens"))
    if isinstance(reasoning, int):
        active["reasoning_tokens"] = reasoning


def _pricing_for(
    model: str, timestamp: datetime
) -> tuple[dict[str, float] | None, str | None]:
    key = model.removesuffix("-cloud")
    if key.startswith("gemma4"):
        return _BASE_RATES_PER_MILLION["gemma4"], "standard"
    if key.startswith("gpt-oss:120b"):
        return _BASE_RATES_PER_MILLION["gpt-oss:120b"], "standard"
    if key.startswith("deepseek-v4.1-flash"):
        off_peak = timestamp.weekday() >= 5 or timestamp.hour < 12 or timestamp.hour >= 18
        tier = "off_peak" if off_peak else "standard"
        return _DEEPSEEK_V41_FLASH_RATES[tier], tier
    return None, None


def _estimated_cost(
    model: str,
    input_tokens: object,
    cached_input_tokens: object,
    output_tokens: object,
    timestamp: datetime,
) -> tuple[float | None, float | None, float | None, str | None]:
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None, None, None, None
    rates, tier = _pricing_for(model, timestamp)
    if rates is None:
        return None, None, None, None
    cached = (
        min(input_tokens, cached_input_tokens)
        if isinstance(cached_input_tokens, int)
        else 0
    )
    uncached = max(0, input_tokens - cached)
    input_amount = (
        uncached * rates["input"] + cached * rates["cached_input"]
    ) / 1_000_000
    output_amount = output_tokens * rates["output"] / 1_000_000
    return (
        round(input_amount + output_amount, 10),
        round(input_amount, 10),
        round(output_amount, 10),
        tier,
    )


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
    usage: dict[str, object] = {}
    usage_token = _active_usage.set(usage)
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
            timestamp = datetime.now(UTC)
            context = _benchmark_context.get()
            model = str(context["model"]) if context.get("model") else None
            record = {
                "timestamp": timestamp.isoformat(),
                "run_id": _run_id.get(),
                "stage": stage,
                "elapsed_seconds": round(elapsed, 3),
                "outcome": outcome,
            }
            for field in ("benchmark_run_id", "model", "job", "repetition"):
                if field in context:
                    record[field] = context[field]
            if "cloud" in context:
                record["provider"] = (
                    "ollama_cloud" if context["cloud"] else "ollama_local"
                )
            record.update(usage)
            if model and context.get("cloud") is True:
                rates, tier = _pricing_for(model, timestamp)
                estimated, input_cost, output_cost, tier = _estimated_cost(
                    model,
                    usage.get("input_tokens"),
                    usage.get("cached_input_tokens"),
                    usage.get("output_tokens"),
                    timestamp,
                )
                record["pricing_source"] = PRICING_SOURCE
                if rates is None:
                    record["estimated_cost_usd"] = None
                    record["pricing_status"] = "rate_unavailable"
                elif estimated is None:
                    record["estimated_cost_usd"] = None
                    record["pricing_tier"] = tier
                    record["pricing_status"] = "token_usage_unavailable"
                else:
                    record["estimated_cost_usd"] = estimated
                    record["estimated_input_cost_usd"] = input_cost
                    record["estimated_output_cost_usd"] = output_cost
                    record["pricing_tier"] = tier
                    record["pricing_status"] = "estimated"
            if attempt is not None:
                record["attempt"] = attempt
            if error_type is not None:
                record["error_type"] = error_type
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(record, ensure_ascii=True) + "\n")
        _active_usage.reset(usage_token)
