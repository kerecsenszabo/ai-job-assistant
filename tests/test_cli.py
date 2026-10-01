import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from assistant import chat, cli


@pytest.mark.parametrize("command,module,arguments", [
    ("generate", "assistant.cv_generator", [
        "--cv", "my cv.pdf", "--job", "job.txt", "--output", "out.pdf", "--model", "local",
    ]),
    ("chat", "assistant.chat", ["--cv", "cv.pdf", "--jobs", "first.txt", "second.pdf"]),
    ("benchmark", "assistant.model_benchmark", [
        "run", "--cv", "cv.pdf", "--jobs", "job.txt", "--models", "first", "second",
    ]),
    ("generate", "assistant.cv_generator", ["--help"]),
    ("chat", "assistant.chat", ["--help"]),
    ("benchmark", "assistant.model_benchmark", ["--help"]),
])
def test_unified_commands_forward_arguments_unchanged(monkeypatch, command, module, arguments):
    handler = Mock()
    imports = []
    monkeypatch.setattr(
        cli, "import_module",
        lambda name: imports.append(name) or SimpleNamespace(main=handler),
    )
    cli.main([command, *arguments])
    assert imports == [module]
    handler.assert_called_once_with(arguments)


def test_main_help_does_not_import_workflows(monkeypatch, capsys):
    monkeypatch.setattr(cli, "import_module", lambda name: pytest.fail("Imported workflow"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert all(command in output for command in cli.COMMANDS)


@pytest.mark.parametrize("arguments", [[], ["invalid"]])
def test_command_is_required_and_valid(arguments):
    with pytest.raises(SystemExit) as exc:
        cli.main(arguments)
    assert exc.value.code == 2


@pytest.mark.parametrize("error", [
    FileNotFoundError("Missing CV"),
    ValueError("Invalid CV"),
    subprocess.CalledProcessError(1, ["tectonic", "cv.tex"]),
])
def test_user_errors_are_reported_without_traceback(monkeypatch, capsys, error):
    handler = Mock(side_effect=error)
    monkeypatch.setattr(cli, "import_module", lambda name: SimpleNamespace(main=handler))
    with pytest.raises(SystemExit) as exc:
        cli.main(["generate"])
    assert exc.value.code == 1
    assert capsys.readouterr().err == f"Error: {error}\n"


def test_unexpected_errors_are_not_hidden(monkeypatch):
    handler = Mock(side_effect=RuntimeError("Unexpected failure"))
    monkeypatch.setattr(cli, "import_module", lambda name: SimpleNamespace(main=handler))
    with pytest.raises(RuntimeError, match="Unexpected failure"):
        cli.main(["generate"])


@pytest.mark.parametrize("command", ["generate", "chat", "benchmark"])
def test_public_command_help_uses_product_name(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--help"])
    assert exc.value.code == 0
    assert f"usage: job-assistant {command}" in capsys.readouterr().out


@pytest.mark.parametrize("command", ["generate", "chat", "benchmark"])
def test_public_commands_reject_unknown_options(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([command, "--unsupported-option"])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_chat_jobs_require_cv_before_indexing(capsys):
    with pytest.raises(SystemExit) as exc:
        chat.main(["--jobs", "job.txt"])
    assert exc.value.code == 2
    assert "--jobs requires --cv" in capsys.readouterr().err


@pytest.fixture
def chat_dependencies(monkeypatch):
    from assistant import ingest_data, rag_chain, vector_store

    index = Mock()
    ask = Mock(return_value="Source-grounded answer")
    collection = Mock()
    collection.count.return_value = 1
    monkeypatch.setattr(ingest_data, "index_documents", index)
    monkeypatch.setattr(rag_chain, "ask", ask)
    monkeypatch.setattr(vector_store, "get_collection", lambda: collection)
    return index, ask, collection


def test_chat_indexes_and_opens_session_in_one_command(monkeypatch, chat_dependencies, tmp_path):
    index, ask, _ = chat_dependencies
    questions = iter(["What does the job require?", "exit"])
    monkeypatch.setattr(chat.Prompt, "ask", lambda *args: next(questions))
    cv, job = tmp_path / "cv.pdf", tmp_path / "job.txt"
    chat.main(["--cv", str(cv), "--jobs", str(job), "--model", "chosen-model"])
    index.assert_called_once_with(cv, [job])
    ask.assert_called_once_with("What does the job require?", model="chosen-model")


@pytest.mark.parametrize("arguments,expected_jobs", [([], None), (["--jobs"], [])])
def test_chat_selects_default_jobs_or_cv_only(
    monkeypatch, chat_dependencies, tmp_path, arguments, expected_jobs,
):
    index, _, _ = chat_dependencies
    monkeypatch.setattr(chat.Prompt, "ask", lambda *args: "quit")
    cv = tmp_path / "cv.pdf"
    chat.main(["--cv", str(cv), *arguments])
    index.assert_called_once_with(cv, expected_jobs)


def test_chat_reuses_existing_index(monkeypatch, chat_dependencies):
    index, _, _ = chat_dependencies
    monkeypatch.setattr(chat.Prompt, "ask", lambda *args: "exit")
    chat.main([])
    index.assert_not_called()


def test_chat_empty_index_explains_first_use(chat_dependencies, capsys):
    _, _, collection = chat_dependencies
    collection.count.return_value = 0
    with pytest.raises(SystemExit) as exc:
        chat.main([])
    assert exc.value.code == 2
    assert "job-assistant chat --cv" in capsys.readouterr().err


@pytest.mark.parametrize("interrupt", [EOFError, KeyboardInterrupt])
def test_chat_exits_cleanly_on_terminal_interrupt(monkeypatch, chat_dependencies, interrupt):
    _, ask, _ = chat_dependencies
    monkeypatch.setattr(chat.Prompt, "ask", Mock(side_effect=interrupt))
    cli.main(["chat"])
    ask.assert_not_called()
