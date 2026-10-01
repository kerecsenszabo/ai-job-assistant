# AI Job Assistant

**PDF CV in, tailored PDF CV out.** Supply a job description to tailor the CV,
chat with your CV and job descriptions using local RAG, and compare local models
on the tailoring workload.

Everything runs locally. Dependency, model and initial TeX downloads need
internet access; your documents are not sent to cloud APIs.

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
ollama pull gemma4:e2b-it-qat
mkdir -p data/job_descriptions
```

The same default model serves generation and chat. Use `--model` to choose
another installed model. The default is `gemma4:e2b-it-qat`; 8GB+ RAM is
recommended. For less memory, select a smaller installed model such as
`granite4.2:3b`.
The chat embedding model downloads on first use, and Tectonic may download
resources on the first PDF export.

## Generate or Tailor a CV

Use `uv run job-assistant` for every workflow. `uv run python -m assistant`
is equivalent. The supported commands are `generate`, `chat`, and `benchmark`;
implementation modules are not standalone CLIs.

Place your text-based CV PDF at `data/cv.pdf` and a target job description at
`data/job_descriptions/target.txt` (or use your own paths), then run:

```bash
# Polish the full CV
uv run job-assistant generate \
  --cv data/cv.pdf --output output/cv.pdf

# Tailor to a job description (TXT or PDF)
uv run job-assistant generate \
  --cv data/cv.pdf --job data/job_descriptions/target.txt \
  --output output/target-cv.pdf
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

JSON input remains supported; `data/cv.example.json` shows the format.
The match report stays separate from the application CV.

## Chat with Your CV and Jobs

```bash
uv run job-assistant chat \
  --cv data/cv.pdf --jobs data/job_descriptions/target.txt

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
  --models gemma4:e2b-it-qat granite4.2:3b --pull
uv run job-assistant benchmark report
```

Benchmarks retain timing, failures, evidence decisions and generated outputs in
`output/benchmarks.sqlite`. Prefer a reviewed source JSON so every model
uses the same facts. PDF input also works: the first selected model imports it
once, outside the measured tailoring cases. Match coverage is not a model-quality
score or a hiring probability. Without `--models`, the runner compares the
curated suite shown by `benchmark models`; `--pull` downloads missing models.
Benchmark responses are capped at 4096 tokens (generation and chat are not).

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

Keep personal CV PDFs/JSON in `data/`, job TXT/PDF files in
`data/job_descriptions/`, generated files in `output/`, and the chat index in
`chroma_db/`. These locations and file types are gitignored; arbitrary other
files under `data/` are not necessarily private. Do not publish personal artifacts.

Run `uv run job-assistant --help` or `uv run job-assistant generate --help`
for command options. See [the generator guide](docs/cv-generator.md) for source
fields, factual safeguards and advanced options. Run `uv run pytest` for development.
