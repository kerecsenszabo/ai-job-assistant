"""One entry point for local CV generation, document chat and model benchmarks."""

import argparse
import subprocess
from importlib import import_module

COMMANDS = {
    "generate": ("assistant.cv_generator", "Generate or tailor a CV PDF"),
    "chat": ("assistant.chat", "Chat with a CV and job descriptions"),
    "benchmark": ("assistant.model_benchmark", "Compare local models on CV tailoring"),
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="job-assistant",
        description="Local CV assistant: PDF in, tailored PDF out, with document chat.",
        epilog="Run job-assistant COMMAND --help for command-specific options.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, (_, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    args, remaining = parser.parse_known_args(argv)
    module = import_module(COMMANDS[args.command][0])
    try:
        module.main(remaining)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Error: {exc}\n")
