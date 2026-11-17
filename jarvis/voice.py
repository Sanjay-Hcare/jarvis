"""Voice loop: hold a key or say "Hey Jarvis", speak, and hear the reply."""

from __future__ import annotations

import sys
import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from jarvis.chat import Conversation
from jarvis.config import Settings
from jarvis.llm import LLMError
from jarvis.sentences import SentenceSplitter

MISSING_PACKAGES = 'Voice mode needs extra packages. Install them with:  pip install -e ".[all]"'


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


def _answer(conversation: Conversation, text: str, name: str, speaker: Any) -> None:
    try:
        reply_aloud(conversation, text, name, speaker)
    except LLMError as exc:
        print(f"\n[{exc}]", file=sys.stderr)


def _default_microphone() -> str:
    try:
        import sounddevice as sd

        return str(sd.query_devices(kind="input")["name"])
    except Exception as exc:
        return f"unknown ({exc})"


def _beep() -> None:
    """A short, quiet tone that says "I'm listening"."""
    try:
        import numpy as np
        import sounddevice as sd

        rate = 16000
        t = np.arange(int(rate * 0.12)) / rate
        tone = 0.15 * np.sin(2 * np.pi * 880 * t) * np.hanning(len(t))
        sd.play(tone.astype(np.float32), rate)
    except Exception:
        pass  # the printed "[listening...]" is enough


def run_voice_chat(conversation: Conversation, settings: Settings, wake_word: bool = False) -> int:
    needed = ["faster_whisper", "sounddevice", "openwakeword" if wake_word else "pynput"]
    # Checked without importing, so a missing package gets the install hint
    # instead of a confusing "could not load the speech model".
    missing = [name for name in needed if find_spec(name) is None]
    if missing:
        print(f"{MISSING_PACKAGES}\n(missing: {', '.join(missing)})", file=sys.stderr)
        return 2
    from jarvis.stt import STTError, Transcriber, add_nvidia_dll_directories

    add_nvidia_dll_directories()
    print("Loading speech recognition...", flush=True)
    try:
        transcriber = Transcriber(
            model=settings.stt_model,
            device=settings.stt_device,
            language=settings.stt_language,
        )
        transcriber.warm_up()
    except (STTError, ValueError, ImportError) as exc:
        print(f"Could not start voice mode: {exc}", file=sys.stderr)
        return 2

    speaker = _start_speaker(settings)
    voice_note = f"voice {settings.tts_voice}" if speaker else "text replies"
    status = f"speech recognition on {transcriber.device}, {voice_note}"
    try:
        if wake_word:
            return _wake_word_loop(conversation, settings, transcriber, speaker, status)
        return _push_to_talk_loop(conversation, settings, transcriber, speaker, status)
    except KeyboardInterrupt:
        print()
        return 0
    finally:
        if speaker is not None:
            speaker.close()


def _push_to_talk_loop(conversation: Conversation, settings: Settings, transcriber: Any, speaker: Any, status: str) -> int:
    from jarvis.audio import AudioError, Recorder, duration_seconds
    from jarvis.ptt import PushToTalk

    try:
        talk_key = PushToTalk(settings.push_to_talk_key)
    except ValueError as exc:
        print(f"Could not start voice mode: {exc}", file=sys.stderr)
        return 2
    recorder = Recorder()
    name = settings.assistant_name
    talk_key.start()
    print(f"{name} is listening ({status}). Hold {settings.push_to_talk_key} to talk, Ctrl+C to quit.")
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
            _answer(conversation, text, name, speaker)
    finally:
        talk_key.stop()
        recorder.stop()


def _wake_word_loop(conversation: Conversation, settings: Settings, transcriber: Any, speaker: Any, status: str) -> int:
    from jarvis.audio import AudioError, MicStream
    from jarvis.tts import TTSError
    from jarvis.wake import FRAME_SAMPLES, ListenSettings, WakeListener, WakeWordModel, ensure_wake_models

    try:
        model = WakeWordModel(ensure_wake_models(Path(settings.wake_models_dir)))
    except (TTSError, OSError, ValueError, ImportError) as exc:
        print(f"Could not start wake-word listening: {exc}", file=sys.stderr)
        return 2

    def on_wake() -> None:
        if speaker is not None:
            # Saying the wake word while it is talking interrupts it.
            speaker.stop()
            speaker.wait(1)
        print("\n[listening...]", end="", flush=True)
        if settings.wake_beep:
            _beep()

    log = None
    if settings.debug_log:
        # Contains what was heard, so it is only written when asked for.
        log_path = Path(settings.debug_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        def log(line: str) -> None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{time.strftime('%H:%M:%S')} {line}\n")

        log(f"--- started; microphone: {_default_microphone()}; wake threshold {settings.wake_threshold}")
        print(f"[diagnostic log on: {log_path}]")

    listener = WakeListener(
        model,
        transcriber.transcribe,
        ListenSettings(threshold=settings.wake_threshold),
        on_wake=on_wake,
        is_speaking=lambda: bool(speaker is not None and speaker.speaking),
        log=log,
    )
    name = settings.assistant_name
    try:
        with MicStream(FRAME_SAMPLES) as mic:
            print(f'{name} is listening ({status}). Say "Hey {name}" and then your request. Ctrl+C to quit.')
            for command in listener.commands(mic.frames()):
                if command is None:
                    # Something sounded like the name but was not; stay quiet.
                    continue
                if not command:
                    print("\r[heard my name, but no request]")
                    continue
                print(f"\rYou: {command}")
                _answer(conversation, command, name, speaker)
                # Sound recorded while the reply was being written is stale.
                mic.discard()
    except AudioError as exc:
        print(f"[{exc}]", file=sys.stderr)
        return 2
    return 0
