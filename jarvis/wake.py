"""Hands-free listening: wait for "Hey Jarvis", then record what follows.

Two stages keep false wake-ups down. openWakeWord listens continuously and
cheaply for something that sounds like "Hey Jarvis"; it also fires on
"Hey Travis" or "hey, jar of jam". When it fires, the words around it are
transcribed and the assistant only responds if "Jarvis" was really said.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, the frame size openWakeWord expects
FRAME_SECONDS = FRAME_SAMPLES / 16000

MODEL_BASE_URL = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/"
MODEL_FILES = ("melspectrogram.onnx", "embedding_model.onnx", "hey_jarvis_v0.1.onnx")


def ensure_wake_models(folder: Path, download: Callable[[str, Path], None] | None = None) -> Path:
    """Download the wake-word model files on first use; return their folder."""
    if download is None:
        from jarvis.tts import _download as download
    folder.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        if not (folder / name).is_file():
            download(MODEL_BASE_URL + name, folder / name)
    return folder


class WakeWordModel:
    """Scores each 80 ms frame for how much it sounds like "Hey Jarvis" (0 to 1)."""

    def __init__(self, folder: Path) -> None:
        from openwakeword.model import Model

        self._model = Model(
            wakeword_models=[str(folder / "hey_jarvis_v0.1.onnx")],
            inference_framework="onnx",
            melspec_model_path=str(folder / "melspectrogram.onnx"),
            embedding_model_path=str(folder / "embedding_model.onnx"),
        )

    def score(self, frame: np.ndarray) -> float:
        return float(next(iter(self._model.predict(frame).values())))

    def reset(self) -> None:
        self._model.reset()


def _edit_distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


_WORD = re.compile(r"[a-z']+")


def _sounds_like_jarvis(word: str) -> bool:
    # One letter off covers "Javis" and "Jarvas"; two would let "Travis" in.
    return _edit_distance(word, "jarvis") <= 1


# Words that can come before the name when someone is addressing it.
_GREETING_WORDS = frozenset(
    "hey hi hello ok okay yo so um uh oh and well alright right excuse me".split()
)


def command_after_wake_word(transcript: str, name_confirmed: bool = False) -> str | None:
    """Return what was said after "Jarvis", or None if "Jarvis" was not said to it.

    Only greeting words may come before the name ("Hey Jarvis", "OK so
    Jarvis"), so a sentence that merely mentions the name ("I think Jarvis
    is a great name") does not count. An empty string means the name was
    said with nothing after it.

    name_confirmed: the name was already recognised in an earlier look at
    the same audio, so the first word after the greeting is taken to be it
    however it is spelt this time (one transcription gave "Jarvis", the
    next "Javish").
    """
    for match in _WORD.finditer(transcript.lower()):
        word = match.group()
        if name_confirmed or _sounds_like_jarvis(word):
            if not name_confirmed or word not in _GREETING_WORDS:
                return transcript[match.end() :].lstrip(" ,.!?;:-").strip()
            continue
        if word not in _GREETING_WORDS:
            return None
    return None


# Audio kept after a detection before checking the name, so all of "Jarvis" is in it.
NAME_TAIL_SECONDS = 0.5
# After a check that did not find the name, wait this long before checking again.
RECHECK_PAUSE_SECONDS = 0.3


@dataclass
class ListenSettings:
    # Wake-word score that triggers the name check. Synthetic voices score
    # 1.0, but a real voice reached only 0.27; the name check behind it keeps
    # a low threshold from causing false wake-ups.
    threshold: float = 0.2
    pre_roll_seconds: float = 1.6  # audio kept from before the detection
    start_timeout: float = 6.0  # how long to wait for a command to begin
    end_silence: float = 1.0  # this much quiet after speech ends the command
    max_seconds: float = 15.0
    # Never treat quieter audio than this as speech. A laptop microphone in a
    # quiet room measured about 0.002, and speech from across a desk 0.005
    # to 0.04, so this sits just above the room.
    min_level: float = 0.004


def _level(frame: np.ndarray) -> float:
    samples = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0


class WakeListener:
    """Turns a stream of microphone frames into spoken commands."""

    def __init__(
        self,
        model: Any,
        transcribe: Callable[[np.ndarray], str],
        settings: ListenSettings | None = None,
        on_wake: Callable[[], None] | None = None,
        is_speaking: Callable[[], bool] = lambda: False,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._model = model
        self._transcribe = transcribe
        self._settings = settings or ListenSettings()
        self._on_wake = on_wake
        self._is_speaking = is_speaking
        # For diagnosing "it doesn't react": loudness and near-misses.
        self._log = log
        self._window_levels: list[float] = []
        self._window_best = 0.0
        self._cooldown = 0
        frames = round(self._settings.pre_roll_seconds / FRAME_SECONDS)
        self._recent: deque[np.ndarray] = deque(maxlen=frames)
        # Recent background levels, for telling speech from room noise.
        self._background: deque[float] = deque(maxlen=50)

    def commands(self, frames: Iterable[np.ndarray]) -> Iterator[str | None]:
        """Yield what is said after each wake-up.

        A string is the command ("" if only the wake word was said); None
        means the detector fired but "Jarvis" was not actually said.
        """
        frames = iter(frames)
        for frame in frames:
            self._recent.append(frame)
            level = _level(frame)
            if not self._is_speaking():
                # Its own voice from the speakers is not background noise.
                self._background.append(level)
            score = self._model.score(frame)
            self._note(level, score)
            if self._cooldown:
                self._cooldown -= 1
                continue
            if score < self._settings.threshold:
                continue

            speech_level = self._speech_level()
            # Let the end of the name finish sounding, then check that it was
            # really "Jarvis" before beeping. The detector is kept sensitive,
            # because real voices score far lower than the synthetic voices
            # it was trained on, so a false trigger must stay silent.
            lead = list(self._recent)
            tail = 0
            while tail < round(NAME_TAIL_SECONDS / FRAME_SECONDS):
                extra = next(frames, None)
                if extra is None:
                    break
                lead.append(extra)
                tail += 1
            first_look = self._transcribe(_as_float(lead))
            if command_after_wake_word(first_look) is None:
                if self._log is not None:
                    self._log(f"near miss (score {score:.2f}): heard {first_look!r}; ignored")
                # The detector may have fired before the name was finished, so
                # the audio and the detector's state are kept for another look;
                # a pause stops it from checking the same moment over and over.
                self._recent.extend(lead[len(lead) - tail :])
                self._cooldown = round(RECHECK_PAUSE_SECONDS / FRAME_SECONDS)
                yield None
                continue

            if self._on_wake is not None:
                self._on_wake()
            heard = self._record_command(frames, lead, waited=tail, speech_level=speech_level)
            self._model.reset()
            self._recent.clear()
            transcript = self._transcribe(heard)
            command = command_after_wake_word(transcript, name_confirmed=True)
            if command is None:
                command = command_after_wake_word(first_look) or ""
            if self._log is not None:
                self._log(
                    f"woke (score {score:.2f}); recorded {len(heard) / 16000:.1f}s with speech level "
                    f"{speech_level:.4f}; heard {transcript!r}; request: {command!r}"
                )
            yield command

    def _note(self, level: float, score: float) -> None:
        """Every five seconds, log how loud the microphone was and the best wake score."""
        if self._log is None:
            return
        self._window_levels.append(level)
        self._window_best = max(self._window_best, score)
        if len(self._window_levels) * FRAME_SECONDS < 5:
            return
        levels = np.array(self._window_levels)
        self._log(
            f"microphone: quiet {np.percentile(levels, 20):.4f}, loudest {levels.max():.4f}; "
            f"best wake score {self._window_best:.2f} (needs {self._settings.threshold:.2f})"
        )
        self._window_levels, self._window_best = [], 0.0

    def _speech_level(self) -> float:
        if not self._background:
            return self._settings.min_level
        quiet = float(np.percentile(np.array(self._background), 20))
        return max(quiet * 3, self._settings.min_level)

    def _record_command(
        self, frames: Iterator[np.ndarray], collected: list[np.ndarray], waited: int, speech_level: float
    ) -> np.ndarray:
        """Collect the wake word plus what follows, until the speaker stops.

        collected already holds the audio up to now, and waited the number
        of frames since the detection.
        """
        settings = self._settings
        collected = list(collected)
        started = False
        quiet_frames = 0
        end_quiet = round(settings.end_silence / FRAME_SECONDS)
        start_limit = round(settings.start_timeout / FRAME_SECONDS)
        total_limit = round(settings.max_seconds / FRAME_SECONDS)
        # The tail of "Jarvis" is still sounding when the detector fires, so
        # it does not count as the start of a command.
        grace = round(0.4 / FRAME_SECONDS)

        for frame in frames:
            collected.append(frame)
            waited += 1
            loud = _level(frame) > speech_level
            if waited > grace and loud:
                started = True
            if started:
                quiet_frames = 0 if loud else quiet_frames + 1
                if quiet_frames >= end_quiet:
                    break
            elif waited >= start_limit:
                break
            if len(collected) >= total_limit:
                break
        return _as_float(collected)


def _as_float(frames: list[np.ndarray]) -> np.ndarray:
    """Join 16-bit frames into the float audio that speech recognition takes."""
    return np.concatenate(frames).astype(np.float32) / 32768.0 if frames else np.zeros(0, np.float32)
