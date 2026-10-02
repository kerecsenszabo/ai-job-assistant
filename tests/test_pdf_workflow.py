import json
from argparse import Namespace
from unittest.mock import Mock

import pytest
from langchain_core.runnables import RunnableLambda

from assistant import (
    cli,
    cv_generator,
    cv_parser,
    cv_tailoring,
    ingest_data,
    model_benchmark,
)
from assistant.cv_generator import CV, load_cv
from assistant.cv_tailoring import TailoringReport


@pytest.fixture
def pdf(tmp_path):
    stream = (
        b"BT /F1 12 Tf 50 750 Td (Example Candidate) Tj "
        b"0 -20 Td (Engineer at Example Co) Tj "
        b"0 -20 Td (2024) Tj "
        b"0 -20 Td (Built Python pipelines.) Tj ET"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    ]
    content = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    startxref = len(content)
    content += b"xref\n0 6\n0000000000 65535 f \n"
    for offset in offsets:
        content += f"{offset:010} 00000 n \n".encode()
    content += (
        f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{startxref}\n%%EOF\n"
    ).encode()
    path = tmp_path / "cv.pdf"
    path.write_bytes(content)
    return path


@pytest.fixture
def source():
    return CV(
        name="Example Candidate",
        experience=[
            {
                "company": "Example Co",
                "role": "Engineer",
                "dates": "2024",
                "bullets": ["Built Python pipelines."],
            }
        ],
    )


def responder(cv):
    return RunnableLambda(lambda prompt, **kwargs: cv.model_dump_json())


def test_pdf_extraction_and_import_preserve_verbatim_facts(pdf, source):
    assert "Built Python pipelines." in cv_parser.extract_text(pdf)
    assert load_cv(pdf, llm=responder(source)) == source


@pytest.mark.parametrize(
    "changes",
    [
        {"skills": ["Kubernetes"]},
        {"summary": "Managed a team of 20."},
        {"languages": [{"name": "English", "proficiency": "Native"}]},
    ],
)
def test_pdf_import_rejects_fabricated_fields(pdf, source, changes):
    with pytest.raises(ValueError, match="unsupported text"):
        load_cv(pdf, llm=responder(CV.model_validate(source.model_dump() | changes)))


def test_pdf_import_cannot_silently_default_missing_sections(pdf):
    llm = RunnableLambda(lambda prompt, **kwargs: '{"name": "Example Candidate"}')
    with pytest.raises(ValueError, match="omitted fields"):
        load_cv(pdf, llm=llm)


def test_pdf_import_rejects_empty_visible_section(pdf, source, monkeypatch):
    monkeypatch.setattr(
        cv_parser,
        "extract_text",
        lambda path: "Example Candidate\nExperience\nEngineer at Example Co\n"
        "2024\nBuilt Python pipelines.",
    )
    with pytest.raises(ValueError, match="omitted the experience section"):
        load_cv(pdf, llm=responder(source.model_copy(update={"experience": []})))


def test_pdf_import_rejects_missing_name(pdf, source):
    with pytest.raises(ValueError, match="candidate name"):
        load_cv(pdf, llm=responder(source.model_copy(update={"name": ""})))


def test_pdf_import_cannot_assign_candidate_name_to_email(pdf, source):
    with pytest.raises(ValueError, match="invalid email"):
        load_cv(pdf, llm=responder(source.model_copy(update={"email": source.name})))


def test_pdf_import_errors_propagate(pdf):
    def fail(prompt, **kwargs):
        raise ConnectionError("Ollama unavailable")

    with pytest.raises(ConnectionError, match="Ollama unavailable"):
        load_cv(pdf, llm=RunnableLambda(fail))


def test_pdf_without_text_requires_ocr(monkeypatch, pdf):
    document = Mock()
    document.__enter__ = Mock(return_value=document)
    document.__exit__ = Mock(return_value=False)
    document.pages = [Mock(extract_text=lambda: None)]
    monkeypatch.setattr(cv_parser.pdfplumber, "open", lambda path: document)
    with pytest.raises(ValueError, match="OCR"):
        cv_parser.extract_text(pdf)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_settings_fail(size, overlap):
    with pytest.raises(ValueError, match="Chunk size"):
        cv_parser.chunk_text("document", size, overlap)


def test_document_reader_supports_pdf_and_txt(pdf, tmp_path):
    assert "Example Candidate" in cv_parser.read_document(pdf)
    job = tmp_path / "job.txt"
    job.write_text("Python required")
    assert cv_parser.read_document(job) == "Python required"
    job.write_text(" ")
    with pytest.raises(ValueError, match="empty"):
        cv_parser.read_document(job)
    with pytest.raises(ValueError, match="PDF or TXT"):
        cv_parser.read_document(tmp_path / "job.docx")


def test_cli_pdf_to_pdf_with_pdf_job(pdf, source, monkeypatch, tmp_path, capsys):
    output = tmp_path / "tailored.pdf"
    imported = responder(source)
    calls = []

    def tailor(cv, job, llm, **kwargs):
        calls.append((cv, job, llm))
        return cv.model_copy(update={"summary": "Engineer"}), TailoringReport()

    monkeypatch.setattr(cv_generator, "local_llm", lambda model: imported)
    monkeypatch.setattr(cv_tailoring, "tailor_with_report", tailor)
    monkeypatch.setattr(
        cv_generator, "write_pdf", lambda latex, path: path.write_text(latex)
    )
    cli.main(
        [
            "generate",
            "--cv",
            str(pdf),
            "--job",
            str(pdf),
            "--output",
            str(output),
            "--no-matching-cache",
        ]
    )
    assert output.exists()
    assert calls == [(source, cv_parser.extract_text(pdf), imported)]
    assert (
        CV.model_validate_json(output.with_suffix(".source.json").read_text()) == source
    )
    exported = CV.model_validate_json(output.with_suffix(".json").read_text())
    assert exported.summary == "Engineer"
    report = json.loads(output.with_suffix(".report.json").read_text())
    assert report["performance"]["stage_model_calls"]["import"] == 1
    assert "Review the imported source JSON" in capsys.readouterr().err


@pytest.mark.parametrize(
    "source_name", ["cv.pdf", "cv.json", "cv.tex", "cv.report.json"]
)
def test_cli_cannot_overwrite_input(tmp_path, source_name):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "generate",
                "--cv",
                str(tmp_path / source_name),
                "--output",
                str(tmp_path / "cv.pdf"),
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize("input_flag", ["--job", "--rubric"])
def test_cli_sidecars_cannot_overwrite_other_inputs(tmp_path, input_flag, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "generate",
                "--cv",
                str(tmp_path / "source.json"),
                "--job",
                str(tmp_path / "target.txt"),
                input_flag,
                str(tmp_path / "cv.json"),
                "--output",
                str(tmp_path / "cv.pdf"),
            ]
        )
    assert exc.value.code == 2
    assert "must not overwrite" in capsys.readouterr().err


def test_imported_source_remains_available_if_pdf_compilation_fails(
    pdf,
    source,
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "generated.pdf"
    monkeypatch.setattr(cv_generator, "local_llm", lambda model: responder(source))
    monkeypatch.setattr(
        cv_tailoring,
        "polish_with_report",
        lambda cv, llm: (cv, TailoringReport()),
    )
    monkeypatch.setattr(
        cv_generator,
        "write_pdf",
        Mock(side_effect=FileNotFoundError("Compiler unavailable")),
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(["generate", "--cv", str(pdf), "--output", str(output)])
    assert exc.value.code == 1
    assert "Compiler unavailable" in capsys.readouterr().err
    assert (
        CV.model_validate_json(output.with_suffix(".source.json").read_text()) == source
    )
    assert not output.exists()
    assert not output.with_suffix(".report.json").exists()


def test_benchmark_accepts_pdf_job(pdf, source, monkeypatch):
    calls = []
    monkeypatch.setattr(
        model_benchmark,
        "tailor_cv",
        lambda cv, job, **kwargs: calls.append(job) or cv,
    )
    result = model_benchmark.benchmark_one("run", "model", pdf, 1, source)
    assert result.status == "ok"
    assert calls == [cv_parser.extract_text(pdf)]


def test_benchmark_pdf_import_is_shared_and_saved(pdf, source, tmp_path, monkeypatch):
    imports = []
    cases = []
    monkeypatch.setattr(
        model_benchmark,
        "installed_model_info",
        lambda: {"first": ("digest-1", "1 GB"), "second": ("digest-2", "2 GB")},
    )
    monkeypatch.setattr(model_benchmark, "command_output", lambda *args: "test")
    monkeypatch.setattr(
        model_benchmark,
        "load_cv",
        lambda path, model: imports.append((path, model)) or source,
    )
    monkeypatch.setattr(model_benchmark, "unload_model", lambda model: None)
    monkeypatch.setattr(model_benchmark, "print_report", lambda *args: None)

    def benchmark(run_id, model, job, repetition, cv, **kwargs):
        cases.append((model, cv))
        return model_benchmark.BenchmarkResult(
            run_id=run_id,
            model=model,
            job=job.name,
            repetition=repetition,
            status="error",
            total_seconds=0,
            error="synthetic case",
        )

    monkeypatch.setattr(model_benchmark, "benchmark_one", benchmark)
    model_benchmark.run_benchmark(
        Namespace(
            cv=pdf,
            jobs=[pdf],
            models=["first", "second"],
            repeat=1,
            pull=False,
            database=tmp_path / "benchmark.sqlite",
            job_cache=tmp_path / "jobs",
            refresh_job_analysis=False,
        )
    )
    assert imports == [(pdf, "first")]
    assert cases == [("first", source), ("second", source)]
    sources = list(tmp_path.glob("*.source.json"))
    assert len(sources) == 1
    assert CV.model_validate_json(sources[0].read_text()) == source


def test_ingestion_indexes_cv_and_jobs_with_source_labels(pdf, tmp_path, monkeypatch):
    from assistant import vector_store

    job = tmp_path / "job.txt"
    job.write_text("Python required")
    calls = []
    monkeypatch.setattr(vector_store, "clear_db", lambda: calls.append("clear"))
    monkeypatch.setattr(
        vector_store,
        "ingest",
        lambda chunks, source: calls.append((chunks, source)),
    )
    ingest_data.index_documents(pdf, [job])
    assert calls == [
        "clear",
        (cv_parser.chunk_text(cv_parser.extract_text(pdf)), "cv:cv.pdf"),
        (["Python required"], "jd:1:job.txt"),
    ]


def test_invalid_ingestion_input_does_not_clear_index(pdf, tmp_path, monkeypatch):
    from assistant import vector_store

    monkeypatch.setattr(
        vector_store, "clear_db", lambda: pytest.fail("Cleared existing index")
    )
    with pytest.raises(FileNotFoundError):
        ingest_data.index_documents(pdf, [tmp_path / "missing.txt"])


def test_ingestion_default_jobs_include_only_supported_documents(
    pdf, tmp_path, monkeypatch
):
    from assistant import vector_store

    jobs = tmp_path / "job_descriptions"
    jobs.mkdir()
    (jobs / "first.txt").write_text("Python required")
    (jobs / "second.pdf").write_bytes(pdf.read_bytes())
    (jobs / "ignored.json").write_text("{}")
    (jobs / "directory.txt").mkdir()
    monkeypatch.setattr(ingest_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(vector_store, "clear_db", lambda: None)
    ingest = Mock()
    monkeypatch.setattr(vector_store, "ingest", ingest)
    ingest_data.index_documents(pdf)
    assert [call.kwargs["source"] for call in ingest.call_args_list] == [
        "cv:cv.pdf",
        "jd:1:first.txt",
        "jd:2:second.pdf",
    ]
    ingest.reset_mock()
    ingest_data.index_documents(pdf, [])
    ingest.assert_called_once_with(
        cv_parser.chunk_text(cv_parser.extract_text(pdf)),
        source="cv:cv.pdf",
    )


def test_clear_index_also_works_on_first_run(monkeypatch):
    from assistant import vector_store

    collection = Mock()
    collection.get.return_value = {"ids": []}
    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)
    vector_store.clear_db()
    collection.delete.assert_not_called()


def test_retrieval_labels_cv_and_job_passages(monkeypatch):
    from assistant import vector_store

    collection = Mock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "documents": [["CV fact", "Job requirement"]],
        "metadatas": [[{"source": "cv:cv.pdf"}, {"source": "jd:1:job.txt"}]],
    }
    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)
    monkeypatch.setattr(vector_store, "embed", lambda texts: [[0.1]])
    assert vector_store.search("Python") == [
        "[cv:cv.pdf]\nCV fact",
        "[jd:1:job.txt]\nJob requirement",
    ]


def test_empty_chat_index_reports_ingestion_instruction(monkeypatch):
    from assistant import vector_store

    collection = Mock()
    collection.count.return_value = 0
    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)
    with pytest.raises(ValueError, match="job-assistant chat --cv"):
        vector_store.search("Python")


def test_chat_uses_shared_model_and_source_labels(monkeypatch):
    from assistant import rag_chain

    prompts = []
    monkeypatch.setattr(
        rag_chain, "search", lambda question, top_k: ["[jd:job.txt]\nPython required"]
    )
    monkeypatch.setattr(
        rag_chain,
        "local_llm",
        lambda model: RunnableLambda(lambda prompt: prompts.append(prompt) or model),
    )
    assert rag_chain.MODEL == cv_generator.MODEL
    assert rag_chain.ask("What is required?", model="custom-model") == "custom-model"
    assert "[jd:job.txt]" in prompts[0].to_string()
    assert "NOT candidate facts" in prompts[0].to_string()
