"""Voice input loop: hold a key, speak, release, read the reply."""

from __future__ import annotations

import sys
import time

from jarvis.chat import Conversation
from jarvis.config import Settings
from jarvis.llm import LLMError

MISSING_PACKAGES = 'Voice mode needs extra packages. Install them with:  pip install -e ".[voice]"'


def run_voice_chat(conversation: Conversation, settings: Settings) -> int:
    try:
        from jarvis.audio import AudioError, Recorder, duration_seconds
        from jarvis.ptt import PushToTalk
        from jarvis.stt import STTError, Transcriber, add_nvidia_dll_directories
    except ImportError as exc:
        print(f"{MISSING_PACKAGES}\n({exc})", file=sys.stderr)
        return 2

    add_nvidia_dll_directories()
    print("Loading speech recognition...", flush=True)
    try:
        transcriber = Transcriber(
            model=settings.stt_model,
            device=settings.stt_device,
            language=settings.stt_language,
        )
        transcriber.warm_up()
        talk_key = PushToTalk(settings.push_to_talk_key)
    except (STTError, ValueError, ImportError) as exc:
        print(f"Could not start voice mode: {exc}", file=sys.stderr)
        return 2

    recorder = Recorder()
    name = settings.assistant_name
    key = settings.push_to_talk_key
    talk_key.start()
    print(f"{name} is listening (speech on {transcriber.device}). Hold {key} to talk, Ctrl+C to quit.")

    try:
        while True:
            # Short waits keep Ctrl+C responsive on Windows.
            while not talk_key.wait_for_press(0.2):
                pass
            try:
                recorder.start()
            except AudioError as exc:
                print(f"[{exc}]", file=sys.stderr)
                talk_key.wait_for_release()
                continue
            print("\n[listening...]", end="", flush=True)
            while not talk_key.wait_for_release(0.2):
                pass
            audio = recorder.stop()

            if duration_seconds(audio) < settings.min_record_seconds:
                print("\r[too short, hold the key while you speak]")
                continue

            started = time.perf_counter()
            text = transcriber.transcribe(audio)
            heard_in = time.perf_counter() - started
            if not text:
                print("\r[didn't catch that]          ")
                continue
            print(f"\rYou: {text}   ({heard_in:.1f}s)")

            print(f"{name}: ", end="", flush=True)
            try:
                for piece in conversation.ask(text):
                    print(piece, end="", flush=True)
                print()
            except LLMError as exc:
                print(f"\n[{exc}]", file=sys.stderr)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        talk_key.stop()
        recorder.stop()
