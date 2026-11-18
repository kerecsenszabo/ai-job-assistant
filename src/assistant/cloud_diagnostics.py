"""Make one small, raw Ollama Cloud request for troubleshooting."""

from __future__ import annotations

import argparse
import os
import sys
import time

import httpx

from assistant.cv_generator import DEFAULT_CLOUD_MODEL


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="job-assistant diagnose-cloud",
        description="Send one tiny prompt and display Ollama Cloud's raw response.",
    )
    parser.add_argument(
        "--model", default=DEFAULT_CLOUD_MODEL, help="Ollama Cloud model"
    )
    parser.add_argument(
        "--job",
        help="Send the first actual job-parsing batch from this description (one call)",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=2000,
        help="Maximum response characters to display (default: 2000)",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.max_output_chars <= 10000:
        parser.error("--max-output-chars must be between 1 and 10000")

    if args.job:
        from pathlib import Path

        from assistant.cv_generator import local_llm
        from assistant.job_parsing import parse_job

        llm = local_llm(args.model, cloud=True, num_predict=4096)

        def show_raw(text: str) -> None:
            print(f"Raw LangChain text ({len(text)} characters):")
            print(repr(text[: args.max_output_chars]))
            if len(text) > args.max_output_chars:
                print(f"[truncated; total text size: {len(text)} characters]")

        try:
            parse_job(
                llm,
                Path(args.job).read_text(encoding="utf-8"),
                raw_response_callback=show_raw,
            )
            print("First batch parsed successfully.")
        except Exception as exc:
            print(f"First-batch parsing stopped: {type(exc).__name__}: {exc}")
        return

    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise ValueError("OLLAMA_API_KEY is not set.")

    payload = {
        "model": args.model,
        "messages": [
            {"role": "user", "content": 'Reply with exactly this JSON: {"ok":true}'}
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 64},
    }
    started = time.perf_counter()
    try:
        response = httpx.post(
            "https://ollama.com/api/chat",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=60,
        )
    except httpx.HTTPError as exc:
        raise ValueError(f"Cloud request failed before an HTTP response: {exc}") from exc

    body = response.text.replace(api_key, "[REDACTED]")
    print(f"HTTP status: {response.status_code}")
    print(f"Elapsed: {time.perf_counter() - started:.2f}s")
    print("Raw response body:")
    print(body[: args.max_output_chars])
    if len(body) > args.max_output_chars:
        print(f"[truncated; total response size: {len(body)} characters]")
    if response.is_error:
        sys.exit(1)

