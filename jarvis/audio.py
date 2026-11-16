"""Microphone recording."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from jarvis.stt import SAMPLE_RATE


class AudioError(RuntimeError):
    """The microphone could not be used."""


class Recorder:
    """Records from the default microphone between start() and stop()."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: Any = None

    def start(self) -> None:
        import sounddevice as sd

        with self._lock:
            self._chunks = []
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._on_audio,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            self._stream = None
            raise AudioError(f"Could not open the microphone: {exc}") from exc

    def stop(self) -> np.ndarray:
        """Stop recording and return the audio as mono float32 samples."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            chunks, self._chunks = self._chunks, []
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def _on_audio(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        # Called on the audio thread; copy because the buffer is reused.
        with self._lock:
            self._chunks.append(indata[:, 0].copy())


def duration_seconds(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    return len(audio) / sample_rate
