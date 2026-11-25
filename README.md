# AI Job Assistant

**PDF CV in, tailored PDF CV out.** Supply a job description to tailor the CV,
chat with your CV and job descriptions using local RAG, and compare local models
on the tailoring workload.

By default, inference and document processing run locally. Dependency, model
and initial TeX downloads need internet access; documents are not sent to cloud
APIs unless you enable Ollama Cloud.

Ollama Cloud is optional. Use `--cloud` with an Ollama model name and set
`OLLAMA_API_KEY` in your shell; CVs and job descriptions sent in that run are
processed by Ollama's cloud service. Local Ollama remains the default.

## Setup

Requires Python 3.14, uv, Ollama, and Tectonic or pdflatex. On macOS:

```bash
brew install uv ollama tectonic
uv python install 3.14
uv sync --python 3.14
```

Start Ollama in a separate terminal if it is not already running:

```bash
ollama serve
```

Then, from the repository root:

```bash
ollama pull granite4.2:3b
mkdir -p data/job_descriptions
```

The source CV may live in any private repository or directory. Configure its
path once in your environment or local `.env` file:

```bash
# Add JOB_ASSISTANT_CV_PATH to the existing .env, or export it in your shell:
export JOB_ASSISTANT_CV_PATH=/path/to/private/cv.json
```

`generate` and `benchmark run` use this configured path when `--cv` is omitted.
`--cv` selects a PDF or another JSON source for one run. Personal CV data is
ignored by this repository. A fresh checkout can copy `.env.example` when it
does not already have a `.env` file.

The same default model serves generation and chat. Use `--model` to choose
another installed model. The default is `granite4.2:3b`; 8GB+ RAM is
recommended. For less memory, select a smaller installed model such as
`qwen3.5:2b`.
With `--cloud`, generation and chat default to `deepseek-v4.1-flash:cloud`;
cloud calls still require explicitly passing `--cloud` and setting
`OLLAMA_API_KEY`.
The chat embedding model downloads on first use, and Tectonic may download
resources on the first PDF export.

## Use Ollama Cloud

Create an API key at [ollama.com/settings/keys](https://ollama.com/settings/keys)
and make it available to the current terminal without committing it to the
repository:

```bash
export OLLAMA_API_KEY="your-key"
```

Then pass `--cloud` and a cloud model. For example:

```bash
uv run job-assistant generate \
  --cv output/cv.source.json --job data/job_descriptions/target.txt \
  --output output/target-cv.pdf --cloud

uv run job-assistant chat --cloud
```

Ollama's available models, free starter usage, and limits can change; check the
Cloud model library and account usage page if a model is unavailable. Cloud
models do not currently support schema-constrained output, so this project asks
for JSON in the prompt and validates every response itself. Review the generated
CV and its report before using it. See [Ollama Cloud docs](https://docs.ollama.com/cloud).

To troubleshoot the provider without running the multi-call CV workflow, use
the one-request diagnostic (up to 64 generated tokens, no retries):

```bash
uv run job-assistant diagnose-cloud
```

It displays the HTTP status and raw response body while redacting the API key.

## Generate or Tailor a CV

Use `uv run job-assistant` for every workflow. `uv run python -m assistant`
is equivalent. The supported commands are `generate`, `chat`, and `benchmark`;
implementation modules are not standalone CLIs.

Put a target job description at `data/job_descriptions/target.txt` (or use your
own path), then run:

```bash
# Polish the private source CV
uv run job-assistant generate --output output/cv.pdf

# Tailor to a job description (TXT or PDF)
uv run job-assistant generate \
  --job data/job_descriptions/target.txt \
  --output output/target-cv.pdf

# Import a PDF for one run
uv run job-assistant generate \
  --cv data/cv.pdf --output output/imported-cv.pdf
```

The model extracts structured facts, selects relevant content when a job is supplied, and improves
wording without authorizing new achievements, metrics, or skills. The output
uses a clean, ATS-friendly layout; it does not reproduce the input PDF design.
Scanned PDFs need OCR before import.

Review the imported `*.source.json`, final PDF and `*.report.json` before use.
Import checks that extracted text exists in the PDF, but can still omit facts
or associate them with the wrong role. Reuse or correct the source JSON for
later runs without importing the PDF again:

```bash
uv run job-assistant generate \
  --cv output/cv.source.json --job data/job_descriptions/target.txt \
  --output output/target-cv.pdf
```

JSON input remains supported; `data/cv.example.json` shows the format. JSON
summaries may be provided as one string or as a list of sentences.
The match report stays separate from the application CV.

## Chat with Your CV and Jobs

```bash
uv run job-assistant chat \
  --cv "$JOB_ASSISTANT_CV_PATH" \
  --jobs data/job_descriptions/target.txt

# Return to the existing document index without re-ingesting
uv run job-assistant chat
```

Supplying `--cv` replaces the chat index before opening chat. Omit `--jobs` to
include all TXT/PDF files in `data/job_descriptions/`; pass an empty `--jobs`
to index only the CV. Without `--cv`, chat reuses the existing index.
Pass multiple paths after `--jobs` to index several jobs:

```bash
uv run job-assistant chat --cv data/cv.pdf \
  --jobs data/job_descriptions/first.txt data/job_descriptions/second.pdf

# Index only the CV
uv run job-assistant chat --cv data/cv.pdf --jobs
```

Chat retrieves source-labelled passages and distinguishes your experience
from job requirements. Retrieval can miss relevant passages; answers still
need review.

## Compare Models

```bash
uv run job-assistant benchmark models
uv run job-assistant benchmark run \
  --cv output/cv.source.json --jobs data/job_descriptions/target.txt \
  --models granite4.2:3b qwen3.5:2b --pull
uv run job-assistant benchmark report
```

To compare Ollama Cloud models on the Sanofi, HCLTech and Maisa descriptions,
set `OLLAMA_API_KEY` as above and run:

```bash
uv run job-assistant benchmark models --cloud
uv run job-assistant benchmark run --cloud \
  --refresh-job-analysis --no-retries \
  --jobs data/job_descriptions/sanofi.txt \
        data/job_descriptions/hcltech_senior_data_ai.txt \
        data/job_descriptions/maisa.txt \
  --database output/cloud-benchmarks.sqlite
uv run job-assistant benchmark report --database output/cloud-benchmarks.sqlite
```

With `--cloud` and no `--models`, the runner uses the cloud shortlist printed by
`benchmark models --cloud`; pass `--models model-a model-b` to choose 2–3
models. The database retains each tailored CV, evidence report and rewrite audit
so you can inspect individual outputs as well as aggregate scores. Cloud runs
send the source CV and job descriptions to Ollama for processing.
`--refresh-job-analysis` reparses each job once and reuses its analysis across
models; `--no-retries` makes one attempt per model call and skips the rewrite
summary recovery call.
Each future benchmark model call is appended to `output/model-calls.jsonl` with
token counts, benchmark run/model/job identifiers, and an estimated USD cost
when usage metadata and a known cloud rate are available. Estimates use the
Ollama rate snapshot dated 2026-11-18; see `pricing_source`, `pricing_tier`, and
`pricing_status` on each log record. They are estimates rather than billing
receipts, and earlier log records are not backfilled.

### Benchmark results (2026-11-18)

Run `20261118T104842Z-f2175339` compared three cloud models on the same CV and
Sanofi, HCLTech and Maisa job descriptions, with refreshed job analyses and no
retries. All 9 cases completed. Match coverage measures CV evidence matched to
job requirements; it is not a model-quality score.

| Model | Match / must-have | Rewrites accepted / rejected / unclear | Mean seconds | Estimated cost |
|---|---:|---:|---:|---:|
| DeepSeek V4.1 Flash | 49.7% / 56.8% | 67 / 2 / 0 | 25.0 | $0.0408 |
| GPT-OSS 120B | 42.8% / 58.1% | 59 / 4 / 1 | 23.0 | $0.0438 |
| Gemma 4 31B | 54.4% / 61.8% | 52 / 23 / 0 | 26.9 | $0.0639 |

DeepSeek V4.1 Flash is the default cloud model based on this comparison: it had
the lowest estimated cost and fewest rejected or unclear rewrite verdicts.
Gemma had the strongest match coverage. Rewrite approvals remain automated
signals and need human review. Cost estimates use the [Ollama pricing snapshot](https://ollama.com/pricing)
checked on 2026-11-18; they are not billing receipts.

Benchmarks retain timing, failures, evidence decisions and generated outputs in
`output/benchmarks.sqlite`. Prefer a reviewed source JSON so every model
uses the same facts. PDF input also works: the first selected model imports it
once, outside the measured tailoring cases. Match coverage is not a model-quality
score or a hiring probability. Without `--models`, the runner compares the
curated suite shown by `benchmark models`; `--pull` downloads missing models.
Benchmark responses are capped at 4096 tokens (generation and chat are not).
If a stale cache error occurs, use `--refresh-job-analysis` or manually
delete the offending file in `output/job-requirements/`.

Historical benchmark databases are not migrated; use `--database` with a new
file if an existing database has an unsupported schema. The previous
`output/model-benchmarks.sqlite` is not modified or used by the current CLI.

## Project Layout

- `src/assistant/cli.py`: unified `job-assistant` CLI.
- `src/assistant/cv_generator.py`: PDF/JSON input, LaTeX/PDF output, CLI.
- `src/assistant/cv_tailoring.py`: evidence matching, selection, safe rewriting.
- `src/assistant/job_requirements.py`, `match_cache.py`: reusable job and match analyses.
- `src/assistant/cv_parser.py`: shared PDF extraction and text chunking.
- `src/assistant/ingest_data.py`, `chat.py`, `rag_chain.py`, `vector_store.py`,
  `embeddings.py`: local document chat.
- `src/assistant/model_benchmark.py`: model comparison.
- `src/assistant/performance.py`: shared stage timings and model-call measurements.

Keep the structured CV in its configured private location. Temporary PDF imports
can stay in `data/`; job TXT/PDF files belong in `data/job_descriptions/`,
generated files in `output/`, and the chat index in `chroma_db/`. These
locations and file types are gitignored; arbitrary other files under `data/`
are not necessarily private. Do not publish personal artifacts.

Run `uv run job-assistant --help` or `uv run job-assistant generate --help`
for command options. See [the generator guide](docs/cv-generator.md) for source
fields, factual safeguards and advanced options. Run `uv run pytest` for development.
