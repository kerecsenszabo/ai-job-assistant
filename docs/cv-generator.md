# CV Generator Guide

See the [README](../README.md) for setup, PDF generation, chat and benchmarks.

## Inputs

For `job-assistant generate`, `--cv` accepts a text-based PDF or structured JSON.
`--job` accepts a UTF-8 TXT file or text-based PDF; omit it to polish the full CV.
`--output` must end in `.pdf` and must not overwrite any input file.
Scanned PDFs require OCR before use. PDF import copies source text into the
CV schema and rejects extracted strings absent from the document after whitespace
normalization. Recognized section headings require nonempty extracted lists.
The email field is restricted to addresses found in the source (or left empty).
These checks do not detect all missing content or incorrect field/role assignment:
review the saved `*.source.json` before relying on it.

The imported JSON is an editable intermediate, not a second required input.
Reuse it for repeat exports and model comparisons to avoid repeating extraction.

[`data/cv.example.json`](../data/cv.example.json) shows the schema. It includes
contact details, summary, skills, experience, education, publications and
certifications. Optional `languages` entries contain `name` and `proficiency`;
proficiency must be explicit, never inferred. Optional `ai_native` items render
an AI-Native Practice section. Unknown fields are rejected.

## Tailoring and Factual Safeguards

The job describes what to emphasize, never new candidate facts. The pipeline:

1. Extracts source-linked, atomic job criteria, separating required, preferred,
   responsibilities and eligibility constraints. Long descriptions use bounded
   chunks; invalid extraction retries with feedback, then fails explicitly.
2. Matches criteria against the complete original CV. Explicit technology and
   language matches use local rules; unresolved semantic criteria use retrieved
   CV evidence and model verification. Any-of groups need one supported option;
   all-of groups need every option. Production claims need explicit production
   evidence for the named technology in work experience.
3. Computes weighted source-evidence coverage, then selects relevant skills and
   bullets while preserving chronology and context for unmatched roles.
4. Rewrites selected bullets and the summary with source citations. Independent
   claim review and mechanical checks reject unsupported numbers, named terms,
   ownership, scale and prototype-to-production upgrades. Rejected or unclear
   wording retains the original, with a visible warning and audit entry.
   When a proposed summary fails the mechanical checks, one focused summary-only
   retry uses the rejection reasons and cited candidate evidence. It is checked
   and reviewed by the same safeguards; if still unsupported, the original
   summary remains. Supported summaries may be retained or subtly adapted to
   the role rather than replaced wholesale.

Without a job, all sections and bullets remain, experience wording is polished,
and the summary is unchanged. Both modes preserve contact details, employers,
roles, dates, education, languages, publications and certifications from the
structured source.

Tailoring selects up to 14 skills, 4 AI-native items and 6 bullets per matched
role, with up to 2 original context bullets for unmatched roles. Layout is
compact 11pt A4; page count depends on content.

Missing CV evidence is not proof that you lack a skill. Automated review cannot
guarantee factual accuracy. Inspect the final document before sending it.

## Outputs

Beside the requested PDF, the generator writes:

- `*.json`: the exported structured CV.
- `*.tex`: the rendered LaTeX.
- `*.report.json`: requirements, source evidence, matches, rewrite verdicts,
  warnings and timings.
- `*.source.json`: the original extracted facts, when the input was a PDF.

The imported source JSON is saved before tailoring and PDF compilation, so it
may remain available even if a later step fails. Reusing JSON input does not
create another `*.source.json`.

The PDF uses pdflatex if installed, otherwise Tectonic. The input PDF's design
is not preserved. Generated files contain personal data and belong in `output/`.

The report's **CV-evidenced job match** is not a hiring probability and never
appears in the application CV. Required and eligibility items have weight 3;
preferred items and responsibilities have weight 1. Direct matches receive full
credit, partial matches half, and unsupported/unclear matches zero. No assessable
requirements means insufficient information. Unresolved eligibility is reported
separately. Rewriting does not change the score.

## Advanced Options

Run `uv run job-assistant generate --help` for all options.

| Option | Purpose |
|---|---|
| `--model NAME` | Choose an installed Ollama model; default `granite4.2:3b` |
| `--rubric FILE` | Override weights, e.g. `{"required_weight": 4, "partial_credit": 0.25}` |
| `--job-cache DIR` | Parsed-job cache; default `output/job-requirements/` |
| `--refresh-job-analysis` | Reparse the job and refresh matching |
| `--matching-cache DIR` | Completed-match cache; default `output/cv-matches/` |
| `--refresh-matching` | Recompute matching |
| `--no-matching-cache` | Disable completed-match caching |

Rubric overrides and refresh options require `--job`. Rubric weights must be
positive and `partial_credit` must be between 0 and 1.

Job analyses are shared across CV edits and models for a consistent rubric.
Repeated headings and soft-wrapped lines are normalized before extraction;
known benefits sections are excluded. Labeled requirements and responsibilities
must be covered, and obvious word-by-word or non-technology criteria are rejected
rather than scored. Existing job-analysis caches created before these parsing changes
require `--refresh-job-analysis`.
Matching-cache keys include the CV, job, parsed criteria, rubric, installed model
digest, inference settings and analysis version. Changed inputs invalidate
matching; rewriting still runs. Corrupt/incompatible records require explicit
refresh rather than silent reuse. Cached matches contain personal data.

## Benchmarks

Use `uv run job-assistant benchmark models`, `benchmark run`, or `benchmark report`.
Run `uv run job-assistant benchmark run --help` for workload options.
The default database is `output/benchmarks.sqlite`.

The benchmark runner compares tailoring, not PDF compilation. It reuses parsed
job criteria but recomputes matching for each case. Benchmark model responses
are capped at 4096 tokens so a runaway response cannot stall the suite indefinitely.
Add `--repeat 2` for repeated
runs, multiple paths after `--jobs` for cross-job comparisons, or `--database`
for a different SQLite file. Use the same `--database` path for both `run` and
`report`; `report` defaults to the latest run, or accepts `--run-id`.
If a job-analysis cache is stale, use `benchmark run --refresh-job-analysis`
to reparse each job once and share its updated analysis across models.
`--pull` downloads missing models.
When benchmarking a PDF, the first model imports it once and saves the extracted
facts as `<run-id>.source.json` beside the database.

Compare factual fidelity and useful wording alongside latency, failures,
model calls and rewrite verdicts. The report shows pass counts, mean successful
runtime, match/must-have coverage, accepted/rejected/unclear rewrites, model calls
and matching/rewriting stage times. Full CV outputs and audits remain in SQLite.
Keyword-overlap and output-diversity metrics are no longer part of the benchmark.
Match coverage is not a model-quality score. Model download sizes are not runtime
RAM; the 16K context, runtime buffers and OS need additional memory.
Peak RAM and swap are not measured.

Only the current benchmark schema is supported. Historical databases are kept
as archives, not automatically migrated. If a chosen database has an unsupported
schema, use a new path, for example `--database output/benchmarks-new.sqlite`,
for both run and report. The former `output/model-benchmarks.sqlite` is not used
by default.
