"""Text to speech with Piper, spoken in the background so the reply keeps streaming."""

from __future__ import annotations

import queue
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

VOICE_BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
_VOICE_NAME = re.compile(r"^(?P<lang>[a-z]{2})_(?P<region>[A-Z]{2})-(?P<name>[A-Za-z0-9_]+)-(?P<quality>x_low|low|medium|high)$")


class TTSError(RuntimeError):
    """Speech output could not be started."""


def voice_urls(voice: str) -> dict[str, str]:
    """Map the two files of a Piper voice to their download URLs."""
    match = _VOICE_NAME.match(voice)
    if not match:
        raise TTSError(
            f"{voice!r} is not a Piper voice name. Expected something like en_GB-alan-medium."
        )
    folder = "{lang}/{lang}_{region}/{name}/{quality}".format(**match.groupdict())
    return {
        f"{voice}.onnx": f"{VOICE_BASE_URL}/{folder}/{voice}.onnx",
        f"{voice}.onnx.json": f"{VOICE_BASE_URL}/{folder}/{voice}.onnx.json",
    }


def _download(url: str, target: Path, attempts: int = 6) -> None:
    """Download to a temporary file and rename, retrying on dropped connections."""
    partial = target.with_name(target.name + ".part")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
                while block := response.read(1 << 16):
                    out.write(block)
            partial.replace(target)
            return
        except (urllib.error.URLError, OSError) as exc:
            last_error = exc
            time.sleep(min(2 * (attempt + 1), 10))
    partial.unlink(missing_ok=True)
    raise TTSError(f"Could not download {url}: {last_error}")


def ensure_voice(voice: str, voices_dir: Path, download: Callable[[str, Path], None] = _download) -> Path:
    """Return the path to the voice model, downloading it on first use."""
    voices_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in voice_urls(voice).items():
        target = voices_dir / filename
        if not target.is_file():
            download(url, target)
    return voices_dir / f"{voice}.onnx"


class PiperSynthesizer:
    """Turns a sentence into audio samples."""

    def __init__(self, model_path: Path, speed: float = 1.0) -> None:
        from piper import PiperVoice, SynthesisConfig

        self._voice = PiperVoice.load(str(model_path))
        self.sample_rate = int(self._voice.config.sample_rate)
        # Piper's length_scale stretches the audio, so faster speech is a smaller scale.
        self._config = SynthesisConfig(length_scale=1.0 / speed) if speed != 1.0 else None

    def synthesize(self, text: str) -> Any:
        import numpy as np

        chunks = [chunk.audio_int16_array for chunk in self._voice.synthesize(text, syn_config=self._config)]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)


class SoundDevicePlayer:
    """Plays audio on the default speakers and can be cut off mid-sentence."""

    def __init__(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate

    def play(self, audio: Any, should_stop: Callable[[], bool]) -> None:
        import sounddevice as sd

        sd.play(audio, self._sample_rate)
        stream = sd.get_stream()
        while stream.active:
            if should_stop():
                sd.stop()
                return
            time.sleep(0.02)


class Speaker:
    """Speaks queued sentences in order on a background thread.

    say() returns immediately, so the caller can keep reading the model's
    reply while earlier sentences are being spoken. stop() cuts the current
    sentence short and drops everything still waiting.
    """

    def __init__(self, synthesizer: Any, player: Any) -> None:
        self._synthesizer = synthesizer
        self._player = player
        self._queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        # Each stop() starts a new generation; sentences from older ones are skipped.
        self._generation = 0
        self._lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        self._pending = 0
        self._thread = threading.Thread(target=self._run, name="jarvis-speaker", daemon=True)
        self._thread.start()

    def say(self, text: str) -> None:
        text = clean_for_speech(text)
        if not text:
            return
        with self._lock:
            self._pending += 1
            self._idle.clear()
            self._queue.put((self._generation, text))

    def stop(self) -> None:
        """Stop talking now and forget anything not yet spoken."""
        with self._lock:
            self._generation += 1

    def wait(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been spoken (or stopped)."""
        return self._idle.wait(timeout)

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=2)

    def _current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text = item
            try:
                if self._current(generation):
                    audio = self._synthesizer.synthesize(text)
                    if len(audio) and self._current(generation):
                        self._player.play(audio, lambda: not self._current(generation))
            except Exception as exc:  # keep the speaker alive if one sentence fails
                print(f"\n[speech failed: {exc}]")
            finally:
                with self._lock:
                    self._pending -= 1
                    if self._pending == 0:
                        self._idle.set()


_MARKUP = re.compile(r"[*`#>|~]+")
_SPACES = re.compile(r"\s+")


def clean_for_speech(text: str) -> str:
    """Remove markdown symbols that a speech engine would read out or stumble on."""
    return _SPACES.sub(" ", _MARKUP.sub("", text).replace("_", " ")).strip()
