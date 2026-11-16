"""Speech to text with faster-whisper, on the GPU when one is usable."""

from __future__ import annotations

import os
import site
from pathlib import Path
from typing import Any, Callable

SAMPLE_RATE = 16000


class STTError(RuntimeError):
    """Speech recognition could not be started."""


def add_nvidia_dll_directories() -> list[str]:
    """Make the pip-installed CUDA libraries findable on Windows.

    The nvidia-cublas-cu12 / nvidia-cudnn-cu12 wheels put their DLLs under
    site-packages/nvidia/*/bin, which is not on the DLL search path.
    """
    added: list[str] = []
    if os.name != "nt":
        return added
    for packages in site.getsitepackages():
        for bin_dir in Path(packages).glob("nvidia/*/bin"):
            os.add_dll_directory(str(bin_dir))
            # CTranslate2 loads cuDNN lazily by name, which only consults PATH.
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            added.append(str(bin_dir))
    return added


def _load_whisper_model(model: str, device: str, compute_type: str) -> Any:
    from faster_whisper import WhisperModel
    from faster_whisper.utils import download_model

    # Use the copy already on disk when there is one, so starting up needs no
    # network. Only the first run downloads the model.
    try:
        model_path = download_model(model, local_files_only=True)
    except Exception:
        model_path = download_model(model)
    return WhisperModel(model_path, device=device, compute_type=compute_type)


class Transcriber:
    """Turns recorded audio into text."""

    # (device, compute_type) attempts for each requested device, fastest first.
    _ATTEMPTS = {
        "auto": [("cuda", "float16"), ("cpu", "int8")],
        "cuda": [("cuda", "float16")],
        "cpu": [("cpu", "int8")],
    }

    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        language: str | None = "en",
        loader: Callable[[str, str, str], Any] = _load_whisper_model,
    ) -> None:
        if device not in self._ATTEMPTS:
            raise ValueError(f"device must be one of {sorted(self._ATTEMPTS)}, got {device!r}")
        self.language = language or None
        self.device = ""
        self._model = None
        errors: list[str] = []
        for attempt_device, compute_type in self._ATTEMPTS[device]:
            try:
                self._model = loader(model, attempt_device, compute_type)
                self.device = attempt_device
                break
            except Exception as exc:  # a missing CUDA library surfaces as a plain RuntimeError
                errors.append(f"{attempt_device}: {exc}")
        if self._model is None:
            raise STTError("Could not load the speech model. " + " | ".join(errors))

    def warm_up(self) -> None:
        """Run one throwaway transcription so the first real one is fast."""
        import numpy as np

        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _info = self._model.transcribe(silence, language=self.language or "en", beam_size=1)
        for _segment in segments:
            pass

    def transcribe(self, audio: Any) -> str:
        """Return the words spoken in 16 kHz mono float32 audio ('' if none)."""
        if len(audio) == 0:
            return ""
        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=1,
            # Skip silence so a quiet recording returns nothing instead of a guess.
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
