"""Microphone recording."""

from __future__ import annotations

import queue
import threading
from typing import Any, Iterator

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


class MicStream:
    """The microphone as an endless series of fixed-size 16-bit frames."""

    def __init__(self, frame_samples: int, sample_rate: int = SAMPLE_RATE) -> None:
        self._frame_samples = frame_samples
        self._sample_rate = sample_rate
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._stream: Any = None

    def __enter__(self) -> "MicStream":
        import sounddevice as sd

        try:
            self._stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self._frame_samples,
                callback=self._on_audio,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise AudioError(f"Could not open the microphone: {exc}") from exc
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def discard(self) -> None:
        """Drop audio captured so far, such as sound recorded while a reply was being written."""
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            try:
                # The timeout lets Ctrl+C through on Windows.
                yield self._frames.get(timeout=0.2)
            except queue.Empty:
                continue

    def _on_audio(self, indata: np.ndarray, frames: int, time: Any, status: Any) -> None:
        try:
            self._frames.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # Nothing has read the microphone for 40 seconds; this sound is
            # thrown away rather than letting memory grow.
            pass
