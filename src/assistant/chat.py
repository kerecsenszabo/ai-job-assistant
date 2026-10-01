"""Chat locally with your indexed CV and job descriptions."""

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from assistant.cv_generator import MODEL

console = Console()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="job-assistant chat",
        description=__doc__,
    )
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    parser.add_argument("--cv", type=Path, help="CV PDF to index before chatting")
    parser.add_argument(
        "--jobs",
        type=Path,
        nargs="*",
        help="Jobs to index with --cv (default: data/job_descriptions; empty for CV only)",
    )
    args = parser.parse_args(argv)
    if args.jobs is not None and args.cv is None:
        parser.error("--jobs requires --cv; omit both to chat with the existing index")
    if args.cv is not None:
        from assistant.ingest_data import index_documents

        index_documents(args.cv, args.jobs)

    from assistant.rag_chain import ask
    from assistant.vector_store import get_collection

    if get_collection().count() == 0:
        parser.error(
            "No documents indexed. Run job-assistant chat --cv data/cv.pdf first."
        )
    console.print(
        Panel.fit(
            "[bold cyan]AI Job Assistant[/bold cyan] — CV & Job Chat\n"
            "Type [yellow]exit[/yellow] or [yellow]quit[/yellow] to stop.",
            border_style="cyan",
        )
    )

    console.print("[dim]Retrieves indexed CV and job-description evidence.[/dim]\n")

    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]")
        except KeyboardInterrupt, EOFError:
            break

        if question.strip().lower() in {"exit", "quit", "q"}:
            break

        console.print("\n[bold blue]Assistant[/bold blue]")
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            response = ask(question, model=args.model)

        console.print(Panel(response, border_style="blue", padding=(0, 1)))

    console.print("\n[dim]Bye![/dim]")
