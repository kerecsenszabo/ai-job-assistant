"""CLI chat with your CV — RAG pipeline with semantic retrieval, Phase 2."""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from assistant.rag_chain import ask

console = Console()


def main() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]AI Job Assistant[/bold cyan] — CV Chat (Phase 2 - RAG)\n"
            "Type [yellow]exit[/yellow] or [yellow]quit[/yellow] to stop.",
            border_style="cyan",
        )
    )

    console.print("[dim]✓ RAG pipeline ready (retrieves relevant CV chunks)[/dim]\n")

    while True:
        try:
            question = Prompt.ask("\n[bold green]You[/bold green]")
        except KeyboardInterrupt, EOFError:
            break

        if question.strip().lower() in {"exit", "quit", "q"}:
            break

        console.print("\n[bold blue]Assistant[/bold blue]")
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            response = ask(question)

        console.print(Panel(response, border_style="blue", padding=(0, 1)))

    console.print("\n[dim]Bye![/dim]")


if __name__ == "__main__":
    main()
