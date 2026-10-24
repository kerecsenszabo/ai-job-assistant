# Benchmarking Small Models for CV Tailoring

We benchmarked a set of small models (under 4B parameters) to find those capable of reliable CV tailoring on an 8GB RAM MacBook.

## Benchmark Results (2026-10-20)

| Model | Status | Notes |
| :--- | :--- | :--- |
| `gemma2:2b` | Failed | Unstable/Unexpected behavior. |
| `granite4.2:3b` | **Success** | Stable and reliable. |
| `qwen2.5:3b` | **Success** | Stable and reliable. |
| `qwen2.5:1.5b` | **Success** | Stable and reliable. |
| `phi4-mini` | **Success** | Stable and reliable. |
| `llama3.2:1b` | Failed | Missing required output format. |
| `llama3.2:3b` | Failed | Missing required output format. |

## Recommendations

1.  **Use Successful Models**: For reliable CV tailoring, use `granite4.2:3b`, `qwen2.5:3b`, `qwen2.5:1.5b`, or `phi4-mini`.
2.  **Note on Model Size**: Models larger than 4B parameters (like `phi4`, which is 14B) are unsuitable for an 8GB RAM MacBook.
3.  **Stability**: Models < 1.5B may be unstable for this specific task due to output structure constraints. Note: `ModelOutputError` is expected for such models.

## Execution

To run the benchmark for the new model suite, use:

```bash
uv run job-assistant benchmark run \
  --cv data/cv.json --jobs data/job_descriptions/elc.txt \
  --models gemma4:e2b gemma4:e4b qwen3.5:2b qwen3.5:4b granite4.1:3b granite4.2:3b \
  --pull
```

The system now correctly handles missing model info, preventing `KeyError`.
The system now automatically maintains model-specific job analysis caches, ensuring accurate results without requiring `--refresh-job-analysis` on every run. If you suspect an analysis is stale or want to force re-parsing, you can add `--refresh-job-analysis` to the command.
