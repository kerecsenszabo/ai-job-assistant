import pytest
from langchain_core.runnables import RunnableLambda

from assistant.cv_tailoring import ParsedJob, request
from assistant.performance import measure_run, measure_stage


def test_model_call_count_and_stage_timings_are_recorded():
    llm = RunnableLambda(lambda prompt, **kwargs: '{"requirements":[]}')
    with measure_run() as performance:
        with measure_stage("job_analysis"):
            request(llm, ParsedJob, "Parse.", {}, stage="job_parsing")
        request(llm, ParsedJob, "Parse.", {}, stage="matching")
    assert performance.total_model_calls == 2
    assert performance.stage_model_calls == {"job_parsing": 1, "matching": 1}
    assert performance.total_seconds > 0
    assert performance.stage_seconds["job_analysis"] > 0
    assert all(value > 0 for value in performance.stage_model_seconds.values())


def test_failed_model_requests_are_counted_and_scopes_do_not_leak():
    def unavailable(prompt, **kwargs):
        raise ConnectionError("Ollama unavailable")

    with measure_run() as failed:
        with pytest.raises(ConnectionError):
            request(
                RunnableLambda(unavailable), ParsedJob, "Parse.", {}, stage="matching"
            )
    assert failed.total_model_calls == 1
    assert failed.stage_model_seconds["matching"] > 0
    with measure_run() as next_run:
        pass
    assert next_run.total_model_calls == 0
    assert next_run.stage_model_calls == {}
