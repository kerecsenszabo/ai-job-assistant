# AI Job Assistant — Build Your Own Local Career AI

A hands-on project to build a **fully local AI assistant** that knows your CV, drafts LinkedIn messages in your voice, and coaches you through job interviews. Built with Python, HuggingFace, LangChain, and Ollama — no cloud APIs, no data leaving your machine.

> **Why this project?** It's immediately useful (you'll actually use it during your job search) and teaches every major GenAI engineering skill: RAG, embeddings, vector databases, fine-tuning, agentic workflows, and LLM serving.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Phase 1 — Local LLM + CV Chat](#phase-1--local-llm--cv-chat)
4. [Phase 2 — RAG Pipeline](#phase-2--rag-pipeline)
5. [Phase 3 — LinkedIn Message Assistant](#phase-3--linkedin-message-assistant)
6. [Phase 4 — Interview Preparation Engine](#phase-4--interview-preparation-engine)
7. [Phase 5 — Fine-Tuning on Your Style](#phase-5--fine-tuning-on-your-style)
8. [Phase 6 — Production System](#phase-6--production-system)
9. [Phase 7 — Advanced Topics](#phase-7--advanced-topics)
10. [Reference: Commands Cheatsheet](#reference-commands-cheatsheet)
11. [Reference: Concepts Glossary](#reference-concepts-glossary)

---

## Prerequisites

### Hardware
- MacBook Pro (Apple Silicon recommended — MPS backend accelerates PyTorch)
- 16GB+ RAM (8B models need ~8GB RAM; 32GB gives you headroom for fine-tuning)
- 20GB+ free disk space (models are large)

### Software to install before starting

```bash
# Package manager (already installed if you're reading this)
brew install uv

# Local LLM server
brew install ollama

# Pull the main model (~5GB, do this first — it runs in background)
ollama pull llama3.1:8b

# Verify
ollama list          # should show llama3.1:8b
uv --version         # should show 0.4+
python3 --version    # should show 3.12+
```

### Key technology choices explained

| Tool | What it is | Why we use it |
|---|---|---|
| **Ollama** | Local LLM server | Runs models like llama3.1 on your Mac with one command |
| **LangChain** | LLM orchestration framework | Chains together prompts, retrievers, memory, tools |
| **ChromaDB** | Embedded vector database | Stores and searches embeddings locally with zero config |
| **sentence-transformers** | Embedding models | Turns text into vectors for semantic search |
| **pdfplumber** | PDF text extraction | Parses your CV PDF reliably |
| **HuggingFace transformers** | Model library | Used in Phase 5 for fine-tuning |
| **PEFT** | Parameter-Efficient Fine-Tuning | Implements LoRA/QLoRA — fine-tune without huge GPU |
| **FastAPI** | Web framework | Backend for the final production app |
| **Gradio** | ML UI framework | Quick UI for testing features as you build |

### Generating a CV

Keep your source CV in the small, portable JSON format shown in
[`data/cv.example.json`](data/cv.example.json). Copy it to `data/cv.json` and
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
`granite4.2:8b`). The JSON CV is the authoritative source of candidate facts;
the job description only determines what to emphasize. All processing stays local.

With `--job`, generation follows an evidence-first pipeline:

1. Parse atomic job requirements with source quotes, distinguishing required,
   preferred, responsibilities, and explicit eligibility constraints.
2. Match requirements against the **complete original CV**, with stable evidence
   IDs and explanations. Independently review these matches as direct, partial,
   not evidenced, or unclear. Missing evidence does not mean you lack the skill.
3. Calculate deterministic **CV-evidenced job match** and must-have coverage.
   Required and eligibility items have weight 3; preferred items and
   responsibilities have weight 1. Direct matches earn full credit, partial
   matches half, and unsupported/unclear matches zero. No assessable requirements
   means insufficient information, not a 0% match. Unresolved eligibility
   constraints are reported separately. This is not a hiring probability.
4. Select relevant skills and bullets without forced minimum counts (maximum
   10 skills, 3 AI-native items, and 4 bullets per role). Preserve role chronology
   and client context; keep one original contextual bullet for unmatched roles.
5. Propose rewritten bullets and a summary, each linked to source evidence.
   An independent claim-level review checks factual support and qualifiers,
   including ownership, scale, metrics, technologies, and prototype versus
   production work. Mechanical checks reject new numbers, unsupported named
   terms, and changed client prefixes. A global skill never authorizes adding
   that technology to a particular role.

Supported rewrites export automatically. Rejected, uncertain, or malformed
rewrite/review responses retain original wording and produce visible warnings.
If any proposed summary sentence fails, the whole original summary is retained.
Service/transport errors still propagate; they are not disguised as successful
generation. Model-based evidence review reduces risk but **cannot guarantee
perfect factual accuracy**; inspect the audit before submitting an application.
Rewriting never changes the match score, which comes only from original evidence.

Without `--job`, the same rewrite safeguards polish every experience bullet
without filtering content or rewriting the summary. Contact details, employer
names, roles, dates, education, publications and certifications are copied from
the source in both modes.

An optional `"ai_native": ["..."]` list in the source JSON provides a separate
AI-Native Practice section for cross-role AI tooling and projects. The generated
JSON and `.tex` are saved beside the PDF. A separate `target-cv.report.json`
contains the scoring rubric, parsed requirements, full source evidence,
requirement matches, selected/contextual evidence IDs, unresolved eligibility
constraints, and every proposed/exported rewrite with its verdict and reason.
Malformed rewrite/review responses are also retained for troubleshooting.
The match percentage and gaps are **not included in the application CV**.
All these files contain personal data; keep them in the gitignored `output/`
directory. PDF import is not part of this pipeline yet: `cv_parser.py` extracts
text, but imported facts must first be confirmed and placed in the source JSON.

To customize scoring, pass `--rubric path/to/rubric.json`, containing any
overrides such as `{"required_weight": 4, "partial_credit": 0.25}`.
All weights must be positive; partial credit must be between 0 and 1.

PDF creation uses `pdflatex` when available, or Tectonic as a user-level alternative:

Ollama receives a Pydantic-generated JSON schema for each operation. Schema,
source-quote, evidence-ID, target-coverage, and review-coverage validation prevent
malformed responses from authorizing claims.

```bash
brew install tectonic
```

MacTeX also works if an administrator can install it:
`brew install --cask mactex`.

### Benchmarking local models

The benchmark runner compares Ollama models on the actual CV-tailoring workload
and stores every result in SQLite. The curated 10-model ladder covers Qwen,
Llama, Gemma and Mistral from 1.7B through 14B, plus `gpt-oss:20b` as the
largest practical candidate for a 24 GiB Mac. Hidden reasoning is disabled
wherever the model supports it, so structured JSON is read from the same output
channel and latency remains comparable; `gpt-oss` is the exception, as it
returns an empty reply when thinking is switched off.

```bash
# See the 10-model suite and which models are already installed
uv run python -m assistant.model_benchmark models

# Start with a representative small/balanced/advanced subset
uv run python -m assistant.model_benchmark run \
  --cv data/cv.json \
  --jobs data/job_descriptions/aldi.txt data/job_descriptions/xr.txt \
  --models qwen3:4b gemma3:4b llama3.1:8b qwen3:14b \
  --repeat 2 \
  --pull

# Reprint the latest stored report
uv run python -m assistant.model_benchmark report
```

Each case records success or failure, selection and summary latency, retries,
summary fallback, selected item count, coarse job-keyword recall, the exact
Ollama model digest, and the full JSON output. Evidence-first runs also record
match and must-have coverage, accepted/rejected/unclear rewrite counts, and the
complete evidence audit in `report_json`; older records keep these fields null.
The report also shows cross-job diversity of exported items. Wording changes can
increase diversity without changing evidence selection. Keyword overlap and
diversity are comparison aids, not correctness scores. Match coverage measures
source support for the job, not model quality, and accepted rewrite counts are
automated judgments, not proof of factual fidelity. Review close candidates
manually for relevance and writing quality. Public benchmarks useful for choosing candidates include
[LiveBench](https://livebench.ai/) for broad current capability,
[IFEval](https://arxiv.org/abs/2311.07911) for instruction following,
[JSONSchemaBench](https://github.com/guidance-ai/jsonschemabench) for structured
output, [FACTS Grounding](https://arxiv.org/abs/2501.03200) for grounded
generation, and [LiveCodeBench](https://livecodebench.github.io/) for coding.
They are screening signals only; local workload quality, latency, retries, and
memory use determine the best model for this application.

---

## Project Structure

The project grows incrementally. Here is the final structure you'll arrive at by the end:

```
ai-job-assistant/
│
├── data/
│   ├── cv.pdf                  # Your CV (gitignored)
│   ├── cover_letters/          # Past cover letters
│   ├── messages/               # LinkedIn messages you've sent
│   └── job_descriptions/       # JDs you're targeting
│
├── src/
│   └── assistant/
│       ├── __init__.py
│       ├── cv_parser.py        # Phase 1 — PDF → text chunks
│       ├── chat.py             # Phase 1 — naive CV chat (CLI)
│       ├── embeddings.py       # Phase 2 — embedding model wrapper
│       ├── vector_store.py     # Phase 2 — ChromaDB operations
│       ├── rag_chain.py        # Phase 2 — retrieval-augmented chat
│       ├── linkedin.py         # Phase 3 — message classifier + drafter
│       ├── interview.py        # Phase 4 — question generator + coach
│       ├── gap_analyzer.py     # Phase 4 — CV vs JD gap analysis
│       └── api.py              # Phase 6 — FastAPI backend
│
├── fine_tuning/
│   ├── dataset/
│   │   └── training_data.jsonl # Phase 5 — your style examples
│   ├── train.py                # Phase 5 — QLoRA fine-tuning script
│   └── evaluate.py             # Phase 5 — before/after comparison
│
├── ui/
│   └── app.py                  # Phase 6 — Streamlit frontend
│
├── tests/
│   └── test_cv_parser.py
│
├── pyproject.toml
├── .gitignore
└── TUTORIAL.md                 # ← you are here
```

---

## Phase 1 — Local LLM + CV Chat

**Duration:** Day 1  
**Goal:** Get a local model running and answering questions about your CV.  
**What you learn:** LLM inference, tokens, context windows, prompt engineering, system prompts.

### Step 1.1 — Set up the project

```bash
cd ai-job-assistant
uv sync                    # installs all dependencies into .venv/
```

### Step 1.2 — Drop in your CV

Copy your CV PDF into the `data/` folder:

```bash
cp ~/Desktop/my-cv.pdf data/cv.pdf
```

### Step 1.3 — Parse your CV

`src/assistant/cv_parser.py` is already written. Run it to verify parsing works:

```bash
uv run python -m assistant.cv_parser
```

You should see 3 sample chunks of your CV printed to the terminal.

**What's happening inside `cv_parser.py`:**
- `pdfplumber` opens the PDF and extracts raw text page by page
- `chunk_text()` splits it into overlapping windows (500 chars, 100 overlap)
- Overlap prevents information from being cut off at chunk boundaries

**Experiment:** Try changing `CHUNK_SIZE` and `CHUNK_OVERLAP` constants. Smaller chunks = more precise retrieval later. Larger chunks = more context per chunk. There is no universally correct value.

### Step 1.4 — Chat with your CV

```bash
# Make sure Ollama is running (it auto-starts on Mac after install)
ollama serve &             # or it may already be running as a service

# Launch the chat
uv run python -m assistant.chat
```

Try these questions:
- *"What is my most recent job title?"*
- *"What programming languages do I know?"*
- *"Summarize my experience in 3 bullet points."*

**What's happening inside `chat.py`:**
- The entire CV text is injected into the **system prompt** (called "prompt stuffing")
- LangChain's `OllamaLLM` sends the prompt to your local Ollama server
- The model never sees the internet — everything runs on your Mac

**Limitation to notice:** If your CV is long, the model's context window fills up. This is exactly why Phase 2 (RAG) exists.

### Phase 1 Concepts to Study

- **Tokens:** LLMs don't see characters, they see tokens (~4 chars each). `llama3.1:8b` has a 128k token context window.
- **System prompt vs user prompt:** System sets behavior/persona, user is the actual question.
- **Temperature:** Controls randomness. `0.0` = deterministic, `1.0` = creative. For factual CV questions, use low temperature (0.1–0.3).
- **Prompt engineering:** The art of writing instructions that get the model to do what you want. Read: [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

## Phase 2 — RAG Pipeline

**Duration:** Week 2–3  
**Goal:** Replace naive prompt stuffing with proper semantic retrieval.  
**What you learn:** Embeddings, vector databases, semantic search, RAG architecture, chunking strategies.

### Why RAG?

Prompt stuffing breaks when:
- Your CV + cover letters + past JDs exceed the context window
- You want to retrieve only the *relevant* sections per question
- You add more documents (certifications, projects, references)

RAG (Retrieval-Augmented Generation) fixes this:
```
Query → Embed query → Search vector DB → Retrieve top-k chunks → Augment prompt → LLM answers
```

### Step 2.1 — Create the embedding module

Create `src/assistant/embeddings.py`:

```python
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"   # 80MB, fast, good quality

_model = None   # lazy-load singleton

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of text strings into vectors."""
    model = get_model()
    return model.encode(texts, show_progress_bar=True).tolist()
```

**What's an embedding?** A list of ~384 floats that encodes the *meaning* of a piece of text. Two semantically similar sentences will have vectors that are close together (high cosine similarity). The model was trained on millions of sentence pairs to learn this mapping.

**Experiment:** Open a Python shell and compare:
```python
from assistant.embeddings import embed
import numpy as np

v1 = embed(["I worked as a machine learning engineer"])
v2 = embed(["My role was ML engineering"])
v3 = embed(["I enjoy hiking on weekends"])

# cosine similarity
def cos_sim(a, b):
    a, b = np.array(a[0]), np.array(b[0])
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cos_sim(v1, v2))   # should be ~0.92 (similar)
print(cos_sim(v1, v3))   # should be ~0.35 (different)
```

### Step 2.2 — Create the vector store module

Create `src/assistant/vector_store.py`:

```python
import chromadb
from pathlib import Path
from assistant.embeddings import embed

DB_PATH = Path(__file__).parent.parent.parent / "chroma_db"
COLLECTION_NAME = "career_docs"

def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(COLLECTION_NAME)

def ingest(chunks: list[str], source: str = "cv") -> None:
    """Embed and store chunks in ChromaDB."""
    collection = get_collection()
    vectors = embed(chunks)
    ids = [f"{source}-{i}" for i in range(len(chunks))]
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=chunks,
        metadatas=[{"source": source}] * len(chunks),
    )
    print(f"Ingested {len(chunks)} chunks from '{source}'")

def search(query: str, top_k: int = 5, source_filter: str | None = None) -> list[str]:
    """Semantic search — returns top_k relevant chunks."""
    collection = get_collection()
    query_vector = embed([query])[0]
    where = {"source": source_filter} if source_filter else None
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
    )
    return results["documents"][0]
```

**Ingest your CV:**
```bash
uv run python -c "
from assistant.cv_parser import load_cv_chunks
from assistant.vector_store import ingest
ingest(load_cv_chunks(), source='cv')
print('Done!')
"
```

### Step 2.3 — Build the RAG chain

Create `src/assistant/rag_chain.py`:

```python
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from assistant.vector_store import search

MODEL = "llama3.1:8b"

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a career assistant. Answer questions about the candidate using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the CV."

Context:
{context}"""),
    ("human", "{question}"),
])

def ask(question: str, top_k: int = 5) -> str:
    context_chunks = search(question, top_k=top_k)
    context = "\n\n---\n\n".join(context_chunks)
    chain = RAG_PROMPT | OllamaLLM(model=MODEL) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
```

**Run it:**
```bash
uv run python -c "from assistant.rag_chain import ask; print(ask('What are my top 3 technical skills?'))"
```

**Compare** the answer quality vs Phase 1 (prompt stuffing). Notice: RAG retrieves only the *relevant* chunks — the model has more focused context and fewer distractions.

### Step 2.4 — Add a Gradio UI

```python
# ui/app.py (Phase 2 version)
import gradio as gr
from assistant.rag_chain import ask

def respond(message, history):
    return ask(message)

gr.ChatInterface(respond, title="AI Job Assistant — CV Chat").launch()
```

```bash
uv run python ui/app.py
# opens http://localhost:7860
```

### Step 2.5 — Upgrade: swap ChromaDB for Qdrant

Once ChromaDB is working, repeat the process with **Qdrant** to learn the differences:

```bash
# Run Qdrant locally with Docker
docker run -p 6333:6333 qdrant/qdrant

# Add client
uv add qdrant-client
```

Key differences you'll discover:
- Qdrant uses **collections** with explicit vector size configuration
- Richer **payload filtering** (filter by metadata at query time)
- Better performance at scale
- Separate REST and gRPC interfaces

### Phase 2 Concepts to Study

- **HNSW index:** The algorithm Chroma and Qdrant use for approximate nearest-neighbor search. Understand the `ef_construction` and `m` parameters.
- **Chunking strategies:** Fixed-size (what we use), semantic (split on sentences/paragraphs), recursive (LangChain's `RecursiveCharacterTextSplitter`). Try all three.
- **Re-ranking:** After vector search, run a cross-encoder re-ranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score results by relevance. Often improves quality significantly.
- **Hybrid search:** Combine vector search (semantic) + BM25 keyword search. Qdrant supports this natively.

---

## Phase 3 — LinkedIn Message Assistant

**Duration:** Week 3–4  
**Goal:** Classify incoming messages and draft replies in your own voice.  
**What you learn:** Few-shot prompting, structured output with Pydantic, conversation memory.

### Step 3.1 — Build a message corpus

Create `data/messages/` and add ~30–50 real messages you've written:

```
data/messages/
├── recruiter_replies.txt    # your replies to recruiters
├── networking.txt           # messages to people you contacted
└── thank_you_notes.txt      # post-interview thank you notes
```

Format: one message per line, or use JSONL:
```json
{"context": "recruiter reached out about senior ML role", "my_reply": "Hi Sarah, thanks for reaching out..."}
```

### Step 3.2 — Message classifier

Create `src/assistant/linkedin.py`:

```python
from enum import Enum
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class MessageType(str, Enum):
    RECRUITER_COLD = "recruiter_cold"
    INTERVIEW_INVITE = "interview_invite"
    JOB_OFFER = "job_offer"
    NETWORKING = "networking"
    FOLLOW_UP = "follow_up"
    OTHER = "other"

class ClassifiedMessage(BaseModel):
    type: MessageType
    urgency: str          # "high" | "medium" | "low"
    key_points: list[str] # bullet points extracted from message
    suggested_action: str # what you should do

def classify_message(message: str) -> ClassifiedMessage:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Classify the following LinkedIn message and extract key information.
Respond with valid JSON matching this schema:
{{"type": "<type>", "urgency": "<urgency>", "key_points": ["..."], "suggested_action": "..."}}

Message types: recruiter_cold, interview_invite, job_offer, networking, follow_up, other"""),
        ("human", "{message}"),
    ])
    chain = prompt | OllamaLLM(model="llama3.1:8b") | JsonOutputParser()
    result = chain.invoke({"message": message})
    return ClassifiedMessage(**result)
```

**What you learn:** Structured output — getting an LLM to return parseable JSON rather than free text. This is fundamental to building reliable pipelines.

### Step 3.3 — Reply drafter with few-shot examples

```python
def draft_reply(incoming: str, my_style_examples: list[str], cv_context: str) -> list[str]:
    """Generate 3 reply drafts in your personal style."""
    examples_text = "\n---\n".join(my_style_examples[:5])
    
    prompt = f"""You are ghostwriting a reply for a professional.
    
Here are examples of how they write:
{examples_text}

Their CV summary:
{cv_context}

Incoming LinkedIn message:
{incoming}

Write 3 different reply options (brief, medium, detailed). Match their writing style exactly.
Format: OPTION 1:\n...\n\nOPTION 2:\n...\n\nOPTION 3:\n..."""
    
    # Direct Ollama call for flexibility
    import httpx
    response = httpx.post("http://localhost:11434/api/generate", json={
        "model": "llama3.1:8b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7}
    })
    return response.json()["response"]
```

### Step 3.4 — Add conversation memory

For multi-turn message threads, use LangChain memory:

```python
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

memory = ConversationBufferWindowMemory(k=10)   # remember last 10 exchanges
```

### Phase 3 Concepts to Study

- **Few-shot prompting:** Providing examples in the prompt to steer output style/format. The more targeted examples, the better.
- **JSON mode / function calling:** Newer models support forcing JSON output. Try `format="json"` in Ollama.
- **Pydantic output parsers:** LangChain's `PydanticOutputParser` auto-generates the JSON schema instruction and validates the output.
- **Temperature tuning:** Low (0.1) for factual tasks, medium (0.5–0.7) for creative writing, high (0.9) for brainstorming.

---

## Phase 4 — Interview Preparation Engine

**Duration:** Week 4–5  
**Goal:** Given a job description, simulate a full interview and give feedback.  
**What you learn:** Structured extraction, agentic patterns (planner → executor → evaluator), multi-step chains.

### Step 4.1 — Job description ingester

Create `src/assistant/gap_analyzer.py`:

```python
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class JobDescription(BaseModel):
    title: str
    company: str
    required_skills: list[str]
    nice_to_have_skills: list[str]
    responsibilities: list[str]
    seniority: str
    tech_stack: list[str]

def parse_jd(raw_text: str) -> JobDescription:
    """Extract structured data from a raw job description."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract structured information from this job description. Return valid JSON."),
        ("human", "{jd}"),
    ])
    chain = prompt | OllamaLLM(model="llama3.1:8b") | JsonOutputParser()
    result = chain.invoke({"jd": raw_text})
    return JobDescription(**result)
```

### Step 4.2 — Gap analyzer

```python
class GapReport(BaseModel):
    strong_matches: list[str]   # your strengths for this role
    gaps: list[str]             # things the JD needs that you lack
    talking_points: list[str]   # how to address gaps in an interview
    overall_fit_score: int      # 0–100

def analyze_gap(jd: JobDescription, cv_chunks: list[str]) -> GapReport:
    cv_text = "\n".join(cv_chunks)
    prompt = f"""Compare this candidate's CV against the job requirements.

JD Requirements:
- Skills: {jd.required_skills}
- Tech stack: {jd.tech_stack}
- Seniority: {jd.seniority}

Candidate CV:
{cv_text}

Return JSON with: strong_matches, gaps, talking_points, overall_fit_score (0-100)"""
    # ... chain invocation
```

### Step 4.3 — Question generator

```python
def generate_questions(jd: JobDescription, gap_report: GapReport) -> dict:
    """Generate role-specific interview questions."""
    return {
        "behavioral": [...],    # STAR-format questions based on responsibilities
        "technical": [...],     # based on required tech stack
        "gap_probing": [...],   # questions likely to surface your gaps
        "questions_to_ask": [], # smart questions for YOU to ask the interviewer
    }
```

### Step 4.4 — Mock interviewer (multi-turn agent)

The mock interviewer is a stateful conversation:

```
System: You are an interviewer for [company]. You are hiring for [role].
        Ask questions one at a time. After each answer, give brief feedback.
        Focus on: [gap_report.gaps] — probe these areas.
        
Flow:
1. Ask question from generated list
2. Candidate answers
3. Agent evaluates: STAR completeness, relevance, red flags
4. Agent gives feedback + asks follow-up or next question
5. After N questions: generate session summary report
```

```python
# src/assistant/interview.py
class MockInterviewer:
    def __init__(self, jd: JobDescription, gap_report: GapReport):
        self.jd = jd
        self.gap_report = gap_report
        self.questions = generate_questions(jd, gap_report)
        self.history = []
        self.current_q_index = 0
    
    def next_question(self) -> str: ...
    def evaluate_answer(self, answer: str) -> str: ...
    def generate_report(self) -> str: ...   # PDF summary
```

### Phase 4 Concepts to Study

- **Agentic patterns:** Planner (decide what to do) → Executor (do it) → Evaluator (was it good?). This loop underlies most AI agents.
- **Structured extraction:** Using LLMs to turn unstructured text into typed data. The reliability of your whole pipeline depends on getting this right.
- **Chain-of-thought prompting:** Adding "Think step by step" improves complex reasoning. Try it on the gap analyzer.
- **Self-consistency:** Run the same prompt 3 times and take a majority vote — reduces hallucinations for factual tasks.

---

## Phase 5 — Fine-Tuning on Your Style

**Duration:** Week 6–7  
**Goal:** Fine-tune a small model to genuinely write and reason like you.  
**What you learn:** LoRA/QLoRA, HuggingFace Trainer, dataset preparation, model quantization, GGUF conversion.

### Why fine-tune?

Few-shot prompting approximates your style. Fine-tuning *bakes it in*. After fine-tuning:
- Smaller prompts needed (no style examples required)
- More consistent tone across outputs
- Model can generalize your style to new situations

### Step 5.1 — Build your training dataset

Create `fine_tuning/dataset/training_data.jsonl` — each line is one example:

```json
{"instruction": "Reply to this recruiter message professionally but briefly", "input": "Hi, I saw your profile and think you'd be great for a Senior ML Engineer role at our startup...", "output": "Hi [Name], thanks for reaching out. I'm open to hearing more — could you share the job description and tech stack? Happy to schedule a quick call if there's a good fit."}
{"instruction": "Explain your experience with recommendation systems", "input": "", "output": "I spent 2 years building real-time recommendation systems at [Company], starting with collaborative filtering and eventually moving to a two-tower neural network architecture..."}
```

**Target:** 100–500 examples for style transfer. Quality beats quantity — only include examples you're proud of.

**Dataset sources:**
- LinkedIn messages you've sent (paste + clean)
- Email drafts and cover letters
- Interview answers you'd give (write from scratch)
- Technical explanations you've written

### Step 5.2 — QLoRA fine-tuning script

Install fine-tuning dependencies:

```bash
uv add transformers peft bitsandbytes accelerate trl datasets
```

Create `fine_tuning/train.py`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset
import torch

MODEL_NAME = "mistralai/Mistral-7B-v0.1"   # or "meta-llama/Meta-Llama-3.1-8B"
OUTPUT_DIR = "fine_tuning/output"

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # rank — higher = more parameters, more capacity
    lora_alpha=32,      # scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],   # which layers to adapt
)

# Load base model in 4-bit (QLoRA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto",   # uses MPS on Apple Silicon
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # expect ~0.1–1% of total params

# Load dataset
dataset = load_dataset("json", data_files="fine_tuning/dataset/training_data.jsonl")

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    dataset_text_field="output",
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    ),
)
trainer.train()
model.save_pretrained(OUTPUT_DIR)
```

```bash
uv run python fine_tuning/train.py
```

**What's happening:**
- **LoRA (Low-Rank Adaptation):** Instead of updating all 7 billion weights, we add small adapter matrices to specific layers. Only ~0.5% of weights are trained. Much faster, much less memory.
- **QLoRA:** Quantize the base model to 4-bit (reduces memory from ~14GB to ~5GB), then apply LoRA adapters in full precision. You can fine-tune a 7B model on a MacBook Pro.
- **SFT (Supervised Fine-Tuning):** Show the model many `(instruction, output)` pairs. It learns to match those patterns.

### Step 5.3 — Evaluate before/after

```python
# fine_tuning/evaluate.py
EVAL_PROMPTS = [
    "Reply to a recruiter asking about your availability",
    "Explain why you're interested in transitioning to GenAI engineering",
    "Describe your biggest professional achievement",
]

# Run each prompt through: base model, fine-tuned model
# Compare outputs qualitatively + compute ROUGE score
```

### Step 5.4 — Convert to GGUF and load in Ollama

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements.txt

# Merge LoRA adapter into base model
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1')
model = PeftModel.from_pretrained(base, 'fine_tuning/output')
model.merge_and_unload().save_pretrained('fine_tuning/merged')
"

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py fine_tuning/merged --outfile data/my-model.gguf

# Create Modelfile for Ollama
echo 'FROM ./data/my-model.gguf' > Modelfile
ollama create my-job-assistant -f Modelfile
ollama run my-job-assistant
```

### Phase 5 Concepts to Study

- **LoRA math:** The adapter adds `W = W_0 + BA` where `B` and `A` are low-rank matrices. If `W` is `d×d` and rank is `r`, we go from `d²` to `2dr` parameters.
- **Quantization types:** `float32` → `float16` → `int8` → `int4`. Each halves memory at some quality cost. `bitsandbytes` handles this automatically.
- **Overfitting risk:** With small datasets (< 100 examples), the model can memorize instead of generalize. Watch train vs validation loss.
- **Learning rate:** Too high → diverges. Too low → doesn't learn. `2e-4` is a safe start for LoRA.

---

## Phase 6 — Production System

**Duration:** Week 8–9  
**Goal:** Clean, runnable local app with a proper API and UI.  
**What you learn:** FastAPI, async LLM calls, Streamlit, Docker Compose, SQLite persistence.

### Step 6.1 — FastAPI backend

```bash
uv add fastapi uvicorn sqlmodel
```

Create `src/assistant/api.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from assistant.rag_chain import ask
from assistant.linkedin import classify_message, draft_reply
from assistant.gap_analyzer import parse_jd, analyze_gap
from assistant.interview import MockInterviewer

app = FastAPI(title="AI Job Assistant API")

@app.post("/cv/ask")
async def ask_cv(query: str) -> dict:
    return {"answer": ask(query)}

@app.post("/linkedin/classify")
async def classify(message: str) -> dict:
    return classify_message(message).model_dump()

@app.post("/linkedin/draft")
async def draft(message: str) -> dict:
    return {"drafts": draft_reply(message)}

@app.post("/interview/start")
async def start_interview(jd_text: str) -> dict:
    jd = parse_jd(jd_text)
    # store session, return session_id
    ...
```

```bash
uv run uvicorn assistant.api:app --reload
# API docs at http://localhost:8000/docs
```

### Step 6.2 — Streamlit frontend

```bash
uv add streamlit
```

Create `ui/app.py` with tabs:
- **CV Chat** — RAG-powered chat
- **LinkedIn** — paste message → classify + get 3 drafts
- **Interview Prep** — paste JD → gap analysis + mock interview
- **Job Tracker** — list of JDs you've applied to (SQLite)

### Step 6.3 — Docker Compose

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333"]
    volumes: ["./qdrant_data:/qdrant/storage"]

  api:
    build: .
    ports: ["8000:8000"]
    depends_on: [qdrant]
    volumes: ["./data:/app/data"]
```

```bash
docker compose up
```

---

## Phase 7 — Advanced Topics

Pick what interests you most after Phase 6. These are open-ended explorations.

### 7.1 LangGraph Agents

LangGraph lets you build stateful, multi-step agents with explicit control flow (loops, branches, human-in-the-loop):

```bash
uv add langgraph
```

**Build:** A company research agent that:
1. Takes a company name
2. Searches the web (with a search tool)
3. Reads their engineering blog
4. Summarizes tech stack, culture, interview process
5. Generates tailored talking points for your interview

### 7.2 RAG Evaluation with RAGAS

Measure your RAG quality objectively:

```bash
uv add ragas
```

Metrics to track:
- **Faithfulness:** Does the answer stay within the retrieved context?
- **Answer relevance:** Does the answer address the question?
- **Context precision:** Are retrieved chunks actually relevant?
- **Context recall:** Did retrieval find all necessary information?

### 7.3 Advanced LLM Serving

Replace Ollama with lower-level serving for performance:

```bash
# llama-cpp-python (direct Python bindings)
pip install llama-cpp-python

# vLLM (production-grade, GPU-optimized)
pip install vllm   # requires NVIDIA GPU
```

### 7.4 MLflow Experiment Tracking

Track your fine-tuning experiments properly:

```bash
uv add mlflow
mlflow ui   # experiment dashboard at http://localhost:5000
```

Log: hyperparameters, training loss curves, eval metrics, model artifacts.

### 7.5 Multimodal Input

Accept screenshots of LinkedIn messages or job postings using a vision model:

```bash
ollama pull llava:13b   # multimodal model
```

---

## Reference: Commands Cheatsheet

```bash
# ─── Ollama ────────────────────────────────────────────────────
ollama serve                        # start local LLM server
ollama pull llama3.1:8b             # download 8B model (~5GB)
ollama pull mistral:7b              # alternative model
ollama pull nomic-embed-text        # embedding model via Ollama
ollama list                         # list downloaded models
ollama run llama3.1:8b              # quick interactive chat
ollama rm llama3.1:8b               # remove model

# ─── uv (Python package manager) ───────────────────────────────
uv sync                             # install all dependencies
uv add <package>                    # add a new dependency
uv add --dev <package>              # add a dev-only dependency
uv run python <script.py>           # run script in project venv
uv run pytest                       # run tests
uv run ipython                      # interactive Python shell

# ─── Running project modules ───────────────────────────────────
uv run python -m assistant.cv_parser    # parse & preview CV
uv run python -m assistant.chat         # Phase 1 CLI chat
uv run python ui/app.py                 # Gradio/Streamlit UI
uv run uvicorn assistant.api:app --reload   # FastAPI backend

# ─── Fine-tuning ───────────────────────────────────────────────
uv run python fine_tuning/train.py      # run QLoRA training
uv run python fine_tuning/evaluate.py   # compare base vs fine-tuned

# ─── Docker ────────────────────────────────────────────────────
docker compose up -d                # start Qdrant + API
docker compose down                 # stop all services
docker compose logs -f api          # follow API logs
```

---

## Reference: Concepts Glossary

| Term | Definition |
|---|---|
| **LLM** | Large Language Model — a neural network trained to predict the next token |
| **Token** | The unit LLMs process — roughly 4 characters or 0.75 words |
| **Context window** | Maximum tokens an LLM can process at once (llama3.1:8b = 128k) |
| **System prompt** | Instructions given to the model before the conversation starts |
| **Temperature** | Randomness of output. 0 = deterministic, 1 = creative |
| **Embedding** | A vector (list of floats) encoding the semantic meaning of text |
| **Vector database** | Database optimized for storing and searching embeddings |
| **Cosine similarity** | Measure of angle between two vectors — how semantically similar two texts are |
| **RAG** | Retrieval-Augmented Generation — retrieve relevant context, then generate |
| **Chunking** | Splitting documents into smaller pieces for embedding and retrieval |
| **LoRA** | Low-Rank Adaptation — fine-tuning by adding small adapter matrices to model weights |
| **QLoRA** | LoRA on a quantized (4-bit) base model — enables fine-tuning on consumer hardware |
| **Quantization** | Reducing model weight precision (float32→int4) to reduce memory usage |
| **GGUF** | File format for quantized LLMs used by llama.cpp and Ollama |
| **SFT** | Supervised Fine-Tuning — training on `(instruction, output)` pairs |
| **Few-shot prompting** | Including example inputs/outputs in the prompt to guide model behavior |
| **Chain-of-thought** | Prompting the model to reason step-by-step before answering |
| **Agent** | An LLM that can use tools, make decisions, and take multi-step actions |
| **LangChain** | Python framework for building LLM pipelines and agents |
| **LangGraph** | Extension of LangChain for stateful, graph-based agentic workflows |
| **HNSW** | Hierarchical Navigable Small World — the graph index used by most vector DBs |
| **BM25** | Classical keyword-based search algorithm — complements vector search |
| **ROUGE** | Metric for text generation quality based on n-gram overlap |
| **MPS** | Metal Performance Shaders — Apple Silicon's GPU acceleration for PyTorch |
