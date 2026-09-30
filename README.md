# AI Job Assistant - Local, Evidence-First CV Generator

Turn a structured source CV into a polished PDF, or tailor it to a job description
using a local language model. **CV generation is the core workflow**: keep your
career facts in one JSON file, generate application-ready documents, and inspect
an evidence and rewrite audit before sending them.

The generator prioritizes relevant experience and improves wording without
authorizing new achievements, metrics, or skills. Job-match scores come from the
original CV evidence, not the rewritten output, and stay out of the application
CV. Automated safeguards reduce risk; they do not replace your final review.

This is also a practical GenAI learning project. Start with a useful CV export,
then explore structured extraction, evidence matching, evaluation, and retrieval.
CV chat is an optional companion; interview coaching, message drafting, an app,
and fine-tuning are later extensions, not prerequisites.

Inference and document processing run locally. Initial dependency and model
downloads require internet access; candidate data is not sent to cloud APIs.

## Start Here

1. [Setup](#setup)
2. [Tools and when you need them](#tools-and-when-you-need-them)
3. [Phases](#phases)
4. [Project structure](#project-structure)
5. [Commands cheatsheet](#commands-cheatsheet)

For source format, scoring, rewrite safeguards, caches, and benchmark details,
see the [CV generator guide](docs/cv-generator.md).

## Setup

### Requirements

- **Python 3.14**: the project requires `>=3.14,<3.15`, not Python 3.12 or 3.13.
- **uv** for Python installation and dependency management.
- **Ollama** for local inference.
- **Tectonic or pdflatex** for PDF compilation.
- Apple Silicon is a good fit for the local workflow. Start with the default
  3B model; 16GB+ RAM is recommended, with more memory needed for larger models.
  Allow disk space for the Python environment and downloaded model weights.

### 1. Install and sync

From the repository root, on macOS:

```bash
brew install uv ollama tectonic
uv python install 3.14
uv sync --python 3.14
```

`uv sync` installs the declared dependencies, including the optional chat stack.
You do not need to configure a vector database, UI, or fine-tuning toolchain to
generate a CV.

The generator uses `pdflatex` when installed, otherwise Tectonic. MacTeX is an
alternative if you already use it; you do not need both compilers. Tectonic may
download TeX resources on the first compilation, so complete that first export
before expecting an offline workflow.

### 2. Start Ollama and download the generator model

In a separate terminal, run this only if Ollama is not already running:

```bash
ollama serve
```

Then, in your project terminal:

```bash
ollama pull granite4.2:3b
ollama list
```

`granite4.2:3b` is the generator default. Use `--model` to select another installed
model; benchmark alternatives in Phase 3 rather than downloading the whole suite
up front.

### 3. Prepare your source CV

```bash
cp data/cv.example.json data/cv.json
```

Replace the example content with your own facts. Keep the source detailed:
tailoring selects from it, so retain relevant projects, responsibilities,
technologies, and truthful metrics. Add language proficiency explicitly if you
want it assessed against job requirements.

`data/cv.json` is gitignored. The generator reads JSON, **not PDF**; PDF parsing
belongs to the separate chat workflow. See the [source CV guide](docs/cv-generator.md#source-cv).

### 4. Generate your first CV

```bash
uv run python -m assistant.cv_generator \
  --cv data/cv.json \
  --output output/cv.pdf
```

This polishes every experience bullet while retaining every source section.
It writes `cv.pdf`, `cv.tex`, `cv.json`, and `cv.report.json` under `output/`.
Review the PDF and the rewrite audit; keep these personal artifacts private.

## Tools and When You Need Them

| Tool | Role in this repo | Introduced |
|---|---|---|
| **uv + Python** | Reproducible environment and module commands | Setup |
| **Ollama** | Serves local models; the generator defaults to `granite4.2:3b` | Setup |
| **Tectonic / pdflatex** | Compiles the generated LaTeX into a PDF | Setup |
| **Pydantic** | Validates the source CV, job criteria, and structured model responses | Phases 1-2 |
| **LangChain + langchain-ollama** | Connects prompts and structured operations to Ollama | Phases 1-2 |
| **httpx** | Local Ollama HTTP calls, including installed-model identity | Phases 2-3 |
| **SQLite** | Stores benchmark results; built into Python | Phase 3 |
| **pdfplumber** | Extracts text from a PDF for the optional chat workflow | Phase 4 |
| **sentence-transformers** | Creates local semantic-search embeddings | Phase 4 |
| **ChromaDB** | Persists the optional chat document index locally | Phase 4 |
| **Rich** | Terminal presentation for CV chat | Phase 4 |
| **Gradio** | Installed UI library; no app is implemented yet | Planned Phase 6 |
| **FastAPI** | Possible API layer; not installed or implemented | Planned Phase 6 |
| **Hugging Face training tools / PEFT** | Optional model adaptation, not part of current setup | Planned Phase 7 |

The generator does not require the ChromaDB chat index. Its evidence matching
and audits operate on the structured source CV.

## Phases

The phases follow product value, not a fixed weekly schedule. Phases 1-4 have
working code; Phases 5-7 describe future work.

### Phase 1 - Generate a Complete CV

**Status:** Implemented. **Outcome:** A general-purpose CV PDF from one source JSON.

Start with the setup above. Learn the source schema, local inference, structured
output, LaTeX rendering, and evidence-checked bullet rewriting. The general export
does not filter content or rewrite the summary.

**Milestone:** A complete CV whose facts you can trace back to your source.

### Phase 2 - Tailor to a Job with an Evidence Audit

**Status:** Implemented. **Outcome:** A job-specific CV plus a separate match report.

Save a job description as `data/job_descriptions/target.txt`, then run:

```bash
uv run python -m assistant.cv_generator \
  --cv data/cv.json \
  --job data/job_descriptions/target.txt \
  --output output/target-cv.pdf
```

The pipeline parses atomic job criteria, matches them against the complete
original CV, computes weighted coverage, prioritizes relevant content, and
reviews proposed rewrites. Unsupported or unclear rewrites retain the original
wording. Missing CV evidence is not proof that you lack a skill, and the score
is not a hiring probability.

**Milestone:** Inspect `output/target-cv.report.json` for supported matches, gaps,
unresolved eligibility constraints, and accepted or rejected claims. Learn
grounded generation, deterministic scoring, validation, and cache design.

See [the pipeline and safeguards](docs/cv-generator.md#job-tailoring-pipeline)
before relying on an export.

### Phase 3 - Evaluate Models and Improve Reliability

**Status:** Implemented. **Outcome:** Choose a model using the actual CV workload.

```bash
uv run python -m assistant.model_benchmark models

uv run python -m assistant.model_benchmark run \
  --cv data/cv.json \
  --jobs data/job_descriptions/target.txt \
  --models granite4.2:3b granite4.2:8b \
  --pull

uv run python -m assistant.model_benchmark report
```

`--pull` downloads missing models. Start with a small comparison and add
`--repeat 2` when you want repeated cases. Results persist in
`output/model-benchmarks.sqlite`. Compare latency, failures, retries, evidence
decisions, and wording quality; a higher match percentage alone does not mean a
better model.

**Milestone:** Pick a model based on factual fidelity, useful writing, runtime,
and memory needs. Learn workload evaluation and reproducible comparisons.

See [benchmark interpretation](docs/cv-generator.md#benchmarking-local-models).

### Phase 4 - Explore Your CV with Retrieval

**Status:** Implemented CLI and RAG modules. **Outcome:** Ask questions over indexed documents.

This is a separate, optional workflow. It uses `data/cv.pdf` and
`granite4.2:8b`, rather than the generator's JSON input and 3B default:

```bash
cp output/cv.pdf data/cv.pdf
ollama pull granite4.2:8b
uv run python -m assistant.cv_parser

uv run python -c "from assistant.cv_parser import load_cv_chunks; from assistant.vector_store import ingest; ingest(load_cv_chunks(), source='cv')"
uv run python -m assistant.chat
```

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) downloads on
first use. Chunks and embeddings stay in `chroma_db/`. Ask about your skills,
recent roles, or project experience.

**Milestone:** Understand chunking, embeddings, semantic search, and
retrieval-augmented generation (RAG). Keep the source JSON authoritative:
chat answers do not update it or authorize generator claims.

`ingest_data.py` can rebuild an existing index from PDFs, cover letters,
messages, and job descriptions. It clears the collection first, so use it only
when deliberately replacing that index. The chat retriever searches the whole
collection, not just CV documents.

### Phase 5 - Reuse Evidence for Interview Prep and Outreach

**Status:** Planned; no interview coach or message drafter exists yet.

Build interview questions and talking points from the parsed job criteria,
supported experience, and unresolved gaps already captured by the generator.
Add recruiter or LinkedIn reply drafts using confirmed facts and writing samples.
Do not build a separate opaque fit score or infer skills from missing evidence.

**Milestone:** Human-reviewed preparation and drafts grounded in the same career
source. Learn few-shot prompting, conversation state, and multi-step evaluation.

### Phase 6 - Package the Generator as a Local App

**Status:** Planned; no API or UI entry point exists yet.

Prioritize editing the source CV, supplying a job description, previewing the
PDF, and reviewing claims before export. Then expose benchmarks, chat, and
preparation as companion tools. Reuse the existing pipeline rather than
duplicating it behind a UI.

**Milestone:** One local workflow with visible errors, audit history, and private
storage. Introduce Gradio or an API/frontend only when implementing this phase;
Docker and a separate vector service are not current requirements.

### Phase 7 - Optional Personalization and Advanced Engineering

**Status:** Planned experiments, not required for usable CV generation.

Explore style fine-tuning only after collecting reviewed examples and establishing
a baseline. Keep factual review and original-source scoring independent of any
personalized model. Choose training tools for the actual hardware; do not assume
a CUDA-oriented QLoRA setup works on Apple Silicon.

Other directions include retrieval evaluation, hybrid search, re-ranking,
stateful agents, multimodal input, and alternative local model serving.

**Milestone:** Demonstrate a measurable quality or performance improvement before
adding infrastructure or training dependencies.

## Project Structure

This reflects the current repository, not hypothetical future modules:

```text
ai-job-assistant/
|-- README.md                    # CV-first setup and phased roadmap
|-- docs/
|   `-- cv-generator.md           # Source format, safeguards, caches, benchmarks
|-- data/
|   |-- cv.example.json           # Portable source CV template (tracked)
|   |-- cv.json                   # Your authoritative CV (private)
|   |-- cv.pdf                    # Optional chat input (private)
|   |-- job_descriptions/         # Target job text files (private)
|   |-- cover_letters/            # Optional retrieval documents (private)
|   `-- messages/                 # Optional writing samples (private)
|-- src/assistant/
|   |-- cv_generator.py           # JSON loading, local model, LaTeX/PDF CLI
|   |-- cv_tailoring.py           # Evidence matching, scoring, rewriting, review
|   |-- job_requirements.py       # Parsed job criteria and reusable job cache
|   |-- match_cache.py            # Private completed-match cache
|   |-- performance.py           # Stage timing and model-call measurements
|   |-- model_benchmark.py        # CV-workload evaluation and SQLite reports
|   |-- cv_parser.py              # PDF extraction and chunking for chat
|   |-- embeddings.py            # Semantic-search embedding model
|   |-- vector_store.py          # Local ChromaDB index
|   |-- ingest_data.py           # Multi-source index rebuild
|   |-- rag_chain.py             # Retrieval-augmented question answering
|   `-- chat.py                  # Optional terminal chat
|-- tests/                       # Generator, evidence, cache, and benchmark tests
|-- output/                      # Generated files, audits, caches, benchmarks (private)
|-- chroma_db/                   # Optional retrieval index (private)
|-- pyproject.toml               # Python requirement and dependencies
`-- uv.lock                      # Locked environment
```

Personal source files, outputs, and indexes are gitignored. Do not move them into
tracked documentation or attach them to public issues.

## Commands Cheatsheet

Run these from the repository root after setup:

```bash
# Environment and local models
uv sync --python 3.14
ollama serve
ollama list
ollama pull granite4.2:3b

# General and job-specific CVs
uv run python -m assistant.cv_generator --cv data/cv.json --output output/cv.pdf
uv run python -m assistant.cv_generator --cv data/cv.json --job data/job_descriptions/target.txt --output output/target-cv.pdf
uv run python -m assistant.cv_generator --help

# Model evaluation
uv run python -m assistant.model_benchmark models
uv run python -m assistant.model_benchmark report
uv run python -m assistant.model_benchmark run --help

# Optional chat (requires Phase 4 setup and ingestion)
uv run python -m assistant.cv_parser
uv run python -m assistant.chat

# Development
uv run pytest
```
