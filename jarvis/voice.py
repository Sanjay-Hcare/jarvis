"""Voice loop: hold a key, speak, release, and hear the reply."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from jarvis.chat import Conversation
from jarvis.config import Settings
from jarvis.llm import LLMError
from jarvis.sentences import SentenceSplitter

MISSING_PACKAGES = 'Voice mode needs extra packages. Install them with:  pip install -e ".[voice]"'


def _start_speaker(settings: Settings) -> Any:
    """Load the voice and return a Speaker, or None when speech is off or unavailable."""
    if not settings.speak:
        return None
    try:
        from jarvis.tts import PiperSynthesizer, SoundDevicePlayer, Speaker, TTSError, ensure_voice
    except ImportError as exc:
        print(f"[speech disabled: {exc}]", file=sys.stderr)
        return None
    try:
        print("Loading voice...", flush=True)
        model_path = ensure_voice(settings.tts_voice, Path(settings.voices_dir))
        synthesizer = PiperSynthesizer(model_path, speed=settings.tts_speed)
        # Synthesize once so the first spoken sentence is not delayed by start-up work.
        synthesizer.synthesize("Ready.")
        return Speaker(synthesizer, SoundDevicePlayer(synthesizer.sample_rate))
    except (TTSError, ImportError, OSError, ValueError) as exc:
        print(f"[speech disabled: {exc}]", file=sys.stderr)
        return None


def reply_aloud(conversation: Conversation, text: str, name: str, speaker: Any) -> None:
    """Print the reply as it streams and speak each sentence as soon as it is complete."""
    splitter = SentenceSplitter()
    started = False
    for piece in conversation.ask(text):
        if not started:
            # Printed late so that tool lines come out above the reply.
            print(f"{name}: ", end="", flush=True)
            started = True
        print(piece, end="", flush=True)
        if speaker is not None:
            for sentence in splitter.feed(piece):
                speaker.say(sentence)
    print()
    if speaker is not None:
        speaker.say(splitter.flush())


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

    speaker = _start_speaker(settings)
    recorder = Recorder()
    name = settings.assistant_name
    key = settings.push_to_talk_key
    talk_key.start()
    voice_note = f"voice {settings.tts_voice}" if speaker else "text replies"
    print(
        f"{name} is listening (speech recognition on {transcriber.device}, {voice_note}). "
        f"Hold {key} to talk, Ctrl+C to quit."
    )

    try:
        while True:
            # Short waits keep Ctrl+C responsive on Windows.
            while not talk_key.wait_for_press(0.2):
                pass
            if speaker is not None:
                # Pressing the key while the assistant is talking interrupts it.
                speaker.stop()
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

            try:
                reply_aloud(conversation, text, name, speaker)
            except LLMError as exc:
                print(f"\n[{exc}]", file=sys.stderr)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        talk_key.stop()
        recorder.stop()
        if speaker is not None:
            speaker.close()
