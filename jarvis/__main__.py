"""Chat with the assistant.

    python -m jarvis           type to it
    python -m jarvis --voice   hold a key and speak to it
"""

from __future__ import annotations

import argparse
import sys

from jarvis.chat import Conversation
from jarvis.config import load_settings
from jarvis.llm import LLMError, OllamaClient
from jarvis.persona import build_system_prompt

HELP = "Commands: /reset forgets this conversation, /quit exits."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="jarvis", description="A fast, fully local companion.")
    parser.add_argument("--voice", action="store_true", help="hold a key and speak instead of typing")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
    except (ValueError, OSError) as exc:
        print(f"Could not load settings: {exc}", file=sys.stderr)
        return 2

    client = OllamaClient(
        host=settings.ollama_host,
        model=settings.model,
        keep_alive=settings.keep_alive,
        timeout=settings.request_timeout,
    )
    conversation = Conversation(
        client,
        system_prompt=build_system_prompt(settings.assistant_name, settings.user_name),
        max_turns=settings.max_turns,
        temperature=settings.temperature,
    )

    if args.voice:
        from jarvis.voice import run_voice_chat

        return run_voice_chat(conversation, settings)

    name = settings.assistant_name
    print(f"{name} is ready (model: {settings.model}). {HELP}")
    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if text in ("/quit", "/exit"):
            return 0
        if text == "/reset":
            conversation.reset()
            print("Conversation cleared.")
            continue
        if text == "/help":
            print(HELP)
            continue

        print(f"{name}: ", end="", flush=True)
        try:
            for piece in conversation.ask(text):
                print(piece, end="", flush=True)
            print()
        except LLMError as exc:
            print(f"\n[{exc}]", file=sys.stderr)
        except KeyboardInterrupt:
            # Ctrl+C while the assistant is talking stops the reply, not the program.
            print("\n[stopped]")


if __name__ == "__main__":
    raise SystemExit(main())
