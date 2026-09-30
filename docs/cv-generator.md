# CV Generator Guide

The CV generator is the main workflow in AI Job Assistant. For installation,
first export, and the phased roadmap, start with the [README](../README.md).
Run the commands below from the repository root.

## Source CV

Keep your source CV in the small, portable JSON format shown in
[`data/cv.example.json`](../data/cv.example.json). Copy it to `data/cv.json` and
replace the example content. To generate a complete general-purpose CV with
every source section and experience bullet, run:

```bash
uv run python -m assistant.cv_generator \
  --cv data/cv.json \
  --output output/cv.pdf
```

To tailor it to a particular job, save the job description as a text file and
run:

```bash
uv run python -m assistant.cv_generator \
  --cv data/cv.json \
  --job data/job_descriptions/target.txt \
  --output output/target-cv.pdf
```

Both commands use the local Ollama model (`--model` overrides the default
`granite4.2:3b`). The JSON CV is the authoritative source of candidate facts;
the job description only determines what to emphasize. All processing stays local.

Both commands rewrite work-experience bullets for clearer, more concise language
and direct action verbs, without inventing achievements or metrics. General-purpose
generation rewrites every bullet; job-tailored generation rewrites the selected
bullets. Original wording is retained when no safe improvement is possible or a
rewrite fails its evidence checks.
Keep employer names in the experience `company` fields, but describe client work
without naming client companies in the bullets. Include project durations and
technical details directly in the bullet text.

Without `--job`, the same rewrite safeguards polish every experience bullet
without filtering content or rewriting the summary. Contact details, employer
names, roles, dates, education, publications and certifications are copied from
the source in both modes.

An optional `"languages": [{"name": "English", "proficiency": "Native"}]`
list renders a separate Languages section, is preserved in both export modes,
and provides explicit evidence for job-language requirements. Proficiency is
copied exactly from the source CV, never inferred or upgraded by the model.

An optional `"ai_native": ["..."]` list in the source JSON provides a separate
AI-Native Practice section for cross-role AI tooling and projects.

PDF import is not part of this pipeline yet: `cv_parser.py` extracts text, but
imported facts must first be confirmed and placed in the source JSON.

## Job-Tailoring Pipeline

With `--job`, generation follows an evidence-first pipeline:

1. Parse atomic job requirements with source quotes, distinguishing required,
   preferred, responsibilities, and explicit eligibility constraints. Explicit
   source sections determine priority: English under required qualifications
   cannot be reclassified as a nice-to-have.
2. Match requirements against the **complete original CV**, with stable evidence
   IDs and explanations. Explicit technology and language criteria are resolved
   locally without model calls. Only unresolved semantic criteria go to the
   model, with retrieved source evidence and a separate verification pass,
   rather than repeatedly resending the complete CV. Missing evidence does not
   mean you lack the skill; retrieved context can also miss relevant evidence.
   Requirements contain individual criteria with explicit **any-of** and
   **all-of** technology groups. One supported option fully satisfies an any-of
   group (for example, scikit-learn or XGBoost), while all-of groups require
   every option. Production framework credit requires the named technology and
   explicit production use together in work evidence. Language proficiency
   requires an explicit language statement; stakeholder collaboration or an
   English-written CV cannot establish it. Criterion decisions and rule
   corrections are included in the report.
3. Calculate deterministic **CV-evidenced job match** and must-have coverage.
   Required and eligibility items have weight 3; preferred items and
   responsibilities have weight 1. Direct matches earn full credit, partial
   matches half, and unsupported/unclear matches zero. No assessable requirements
   means insufficient information, not a 0% match. Unresolved eligibility
   constraints are reported separately. This is not a hiring probability.
4. Prioritize relevant skills and bullets (maximum 14 skills, 4 AI-native items,
   and 6 bullets per matched role). Use remaining role space for original career
   context rather than discarding everything not cited by a requirement.
   Preserve role chronology and client context; keep up to two original bullets
   for unmatched roles. Contextual bullets are recorded separately from positive
   job matches and never increase match scores. The compact 11pt layout
   aims for roughly two pages for a detailed tailored CV such as XR; the exact
   length depends on the source material and accepted rewrites.
5. Propose rewritten bullets and a summary, each linked to source evidence.
   An independent claim-level review checks factual support and qualifiers,
   including ownership, scale, metrics, technologies, and prototype versus
   production work. Mechanical checks reject new numbers, unsupported named
   terms, and changed client prefixes. A global skill never authorizes adding
   that technology to a particular role.

Supported rewrites export automatically. Draft and review schemas constrain
exact target keys. A missing, rejected, uncertain, or malformed bullet retains
only that bullet's original wording; other valid bullets still export after
review. Unchanged originals need no review call, and mechanical rejections
happen before model review. Whole unreadable responses retain original wording
and produce visible warnings.

The generator accepts only criterion-based job analyses and keyed rewrite and
review responses; older unkeyed formats are not used for generation. Reparse a
cached job without criteria using `--refresh-job-analysis`.
If any proposed summary sentence fails, the whole original summary is retained.
Service/transport errors still propagate; they are not disguised as successful
generation. Model-based evidence review reduces risk but **cannot guarantee
perfect factual accuracy**; inspect the audit before submitting an application.
Rewriting never changes the match score, which comes only from original evidence.

Ollama receives a Pydantic-generated JSON schema for each operation. Schema,
source-quote, evidence-ID, target-coverage, and review-coverage validation prevent
malformed responses from authorizing claims.

To customize scoring, pass `--rubric path/to/rubric.json`, containing any
overrides such as `{"required_weight": 4, "partial_credit": 0.25}`.
All weights must be positive; partial credit must be between 0 and 1.

## Outputs and Privacy

The generated JSON and `.tex` are saved beside the PDF. A separate
`target-cv.report.json` contains the scoring rubric, parsed requirements, full
source evidence, requirement matches, selected/contextual evidence IDs, unresolved
eligibility constraints, and every proposed/exported rewrite with its verdict
and reason. Malformed rewrite/review responses are also retained for troubleshooting.
The match percentage and gaps are **not included in the application CV**.
All these files contain personal data; keep them in the gitignored `output/`
directory.

PDF creation uses `pdflatex` when available, or Tectonic as a user-level alternative:

```bash
brew install tectonic
```

MacTeX also works if an administrator can install it:
`brew install --cask mactex`. Tectonic may download TeX resources on first use.

## Caches and Performance

The CLI saves parsed job requirements in `output/job-requirements/`, keyed by
the job description's content hash and a cache format version. CV edits and
model changes reuse the same parsed job rubric instead of redefining it on
every run. Changes to the job description create a new analysis. Use
`--job-cache path/to/cache` to choose a different directory or
`--refresh-job-analysis` to deliberately reparse the job. Corrupt or incompatible
cache records produce an actionable error, not a silent reparse. The cache stores
job requirements only, never candidate evidence. General semantic criteria still
use model-reviewed judgments; caching stabilizes requirements and weighting,
not a guarantee of identical model judgments or percentages.
The benchmark runner shares this cache (`--job-cache` overrides it), so models
and repeated runs assess the same parsed requirements rather than different
model-specific scoring rubrics.

Completed matching is cached separately in `output/cv-matches/`. Its key includes
the source CV, job description, parsed criteria, rubric, model name and installed
weights digest, inference settings, schema, and matching-analysis version.
Unchanged exports reuse the completed matches and score; CV edits, new weights,
or a changed rubric invalidate them. Use `--matching-cache path/to/cache` to
choose a directory, `--refresh-matching` to recompute, or `--no-matching-cache`
for an uncached run. Refreshing job analysis also refreshes matching. These
files contain personal match explanations and must stay private.

The CLI prints total elapsed time and per-stage model-call counts/durations.
The same measurements, including matching-cache reuse, are saved under
`performance` in the report. A warm matching cache removes matching calls,
but generation and verification of genuinely changed wording still run.

## Benchmarking Local Models

The benchmark runner compares Ollama models on the actual CV-tailoring workload
and stores every result in SQLite. The default shortlist targets machines with
8 GB total RAM: Qwen3.5 0.8B as a lightweight option, Granite 4.2 3B as the
baseline, and Qwen3.5 2B/4B Q4_K_M plus Ministral 3 3B Instruct Q4_K_M as
challengers. Run `models` for exact tags, download sizes and installed status.
Download sizes are not runtime RAM: the 16K context, runtime buffers, OS and
application also need memory. Peak memory and swap use are not yet measured;
the shortlist is not a guarantee of fitting within 8 GB. Larger models remain
selectable explicitly with `--models`.
Hidden reasoning is disabled where supported so structured JSON remains
in the output channel. `gpt-oss`, when selected explicitly, requires its default
reasoning mode.

```bash
# See the suite and which models are already installed
uv run python -m assistant.model_benchmark models

# Compare the default model to a smaller-memory challenger
uv run python -m assistant.model_benchmark run \
  --cv data/cv.json \
  --jobs data/job_descriptions/target.txt \
  --models granite4.2:3b qwen3.5:4b-q4_K_M \
  --pull

# Reprint the latest stored report
uv run python -m assistant.model_benchmark report
```

Use `--repeat 2` for repeated cases, multiple paths after `--jobs` for cross-job
comparisons, and `--database path/to/results.sqlite` for another private database.
The default is `output/model-benchmarks.sqlite`.

Each case records success or failure, selection and summary latency, retries,
summary fallback, selected item count, coarse job-keyword recall, the exact
Ollama model digest, and the full JSON output. Evidence-first runs also record
match and must-have coverage, accepted/rejected/unclear rewrite counts, and the
complete evidence audit in `report_json`, plus model-call count, matching and
rewrite durations, and matching-cache use; older records keep these fields null.
The benchmark reuses the parsed job requirements for all models, but computes
matching separately for each case rather than using completed-match caches.
This measures the full matching workload while preserving a consistent rubric.

The report also shows cross-job diversity of exported items. Wording changes can
increase diversity without changing evidence selection. Keyword overlap and
diversity are comparison aids, not correctness scores. Match coverage measures
source support for the job, not model quality, and accepted rewrite counts are
automated judgments, not proof of factual fidelity. Review close candidates
manually for relevance and writing quality.

Public benchmarks useful for choosing candidates include
[LiveBench](https://livebench.ai/) for broad current capability,
[IFEval](https://arxiv.org/abs/2311.07911) for instruction following,
[JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) for structured
output, [FACTS Grounding](https://arxiv.org/abs/2501.03200) for grounded
generation, and [LiveCodeBench](https://livecodebench.github.io/) for coding.
They are screening signals only; local workload quality, latency, retries, and
memory use determine the best model for this application.
