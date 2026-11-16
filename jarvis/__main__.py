"""Chat with the assistant.

    python -m jarvis           type to it
    python -m jarvis --voice   hold a key and speak to it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jarvis.chat import Conversation
from jarvis.config import Settings, load_settings
from jarvis.llm import LLMError, OllamaClient
from jarvis.memory import FactExtractor, Memory, MemoryStore
from jarvis.persona import build_system_prompt
from jarvis.tasks import default_tools
from jarvis.tools import ToolBox, ToolCall

HELP = (
    "Commands: /memory shows what I remember, /remember <fact> adds one, "
    "/forget <words> removes matching facts, /reset clears this conversation, /quit exits."
)


def build_memory(settings: Settings, client: OllamaClient) -> Memory | None:
    if not settings.remember:
        return None
    store = MemoryStore(Path(settings.memory_path), max_facts=settings.max_facts)
    return Memory(
        store,
        FactExtractor(client, settings.user_name),
        on_learned=lambda fact: print(f"\n[remembered: {fact}]", flush=True),
    )


def build_toolbox(settings: Settings) -> ToolBox | None:
    if not settings.use_tools:
        return None
    return ToolBox(default_tools(home_city=settings.home_city, screenshot_folder=settings.screenshot_folder))


def announce_tool(call: ToolCall, result: str) -> None:
    """Show what was actually done, independent of how the model words it."""
    print(f"[{call.name}: {result}]", flush=True)


def build_conversation(settings: Settings, client: OllamaClient, memory: Memory | None) -> Conversation:
    def system_prompt() -> str:
        facts = memory.store.facts if memory else None
        return build_system_prompt(settings.assistant_name, settings.user_name, facts)

    return Conversation(
        client,
        system_prompt=system_prompt,
        max_turns=settings.max_turns,
        temperature=settings.temperature,
        on_exchange=memory.observe if memory else None,
        toolbox=build_toolbox(settings),
        on_tool=announce_tool,
    )


def handle_command(text: str, conversation: Conversation, memory: Memory | None) -> bool:
    """Run a slash command. Returns False when the program should exit."""
    command, _, argument = text.partition(" ")
    argument = argument.strip()
    if command in ("/quit", "/exit"):
        return False
    if command == "/help":
        print(HELP)
    elif command == "/reset":
        conversation.reset()
        print("Conversation cleared. What I remember about you is kept.")
    elif command in ("/memory", "/remember", "/forget") and memory is None:
        print("Memory is turned off (remember = false).")
    elif command == "/memory":
        facts = memory.store.facts
        print("\n".join(f"  - {fact}" for fact in facts) if facts else "I don't remember anything about you yet.")
    elif command == "/remember":
        if not argument:
            print("Usage: /remember <fact>")
        else:
            print("Got it." if memory.store.add(argument) else "I already knew that.")
    elif command == "/forget":
        removed = memory.store.forget(argument) if argument else []
        if removed:
            print("Forgotten:\n" + "\n".join(f"  - {fact}" for fact in removed))
        else:
            print("Nothing matched." if argument else "Usage: /forget <words>")
    else:
        print(f"Unknown command {command}. {HELP}")
    return True


def run_text_chat(conversation: Conversation, settings: Settings, memory: Memory | None) -> int:
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
        if text.startswith("/"):
            if not handle_command(text, conversation, memory):
                return 0
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
    memory = build_memory(settings, client)
    conversation = build_conversation(settings, client, memory)

    try:
        if args.voice:
            from jarvis.voice import run_voice_chat

            return run_voice_chat(conversation, settings)
        return run_text_chat(conversation, settings, memory)
    finally:
        if memory is not None:
            # Finish learning from the last message before exiting.
            memory.wait()
            memory.close()


if __name__ == "__main__":
    raise SystemExit(main())
