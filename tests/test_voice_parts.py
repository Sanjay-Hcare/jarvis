import pytest

np = pytest.importorskip("numpy")

from jarvis.audio import duration_seconds
from jarvis.stt import STTError, Transcriber


class Segment:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self, texts):
        self.texts = texts
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return (Segment(t) for t in self.texts), None


def loader_for(model, failing=()):
    attempts = []

    def load(name, device, compute_type):
        attempts.append((device, compute_type))
        if device in failing:
            raise RuntimeError(f"{device} unavailable")
        return model

    return load, attempts


def test_segments_are_joined_and_trimmed():
    load, _ = loader_for(FakeModel([" Hello ", "there. "]))
    transcriber = Transcriber(loader=load)

    assert transcriber.transcribe(np.zeros(16000, dtype=np.float32)) == "Hello there."


def test_empty_audio_is_not_sent_to_the_model():
    model = FakeModel(["should not appear"])
    load, _ = loader_for(model)

    assert Transcriber(loader=load).transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert model.kwargs is None


def test_language_and_vad_are_passed_through():
    model = FakeModel(["hi"])
    load, _ = loader_for(model)
    Transcriber(language="", loader=load).transcribe(np.zeros(8000, dtype=np.float32))

    assert model.kwargs["language"] is None
    assert model.kwargs["vad_filter"] is True


def test_auto_falls_back_to_cpu_when_gpu_fails():
    load, attempts = loader_for(FakeModel([]), failing=("cuda",))
    transcriber = Transcriber(device="auto", loader=load)

    assert transcriber.device == "cpu"
    assert [device for device, _ in attempts] == ["cuda", "cpu"]


def test_explicit_cuda_does_not_fall_back():
    load, _ = loader_for(FakeModel([]), failing=("cuda",))

    with pytest.raises(STTError, match="cuda unavailable"):
        Transcriber(device="cuda", loader=load)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        Transcriber(device="gpu", loader=lambda *a: None)


def test_duration_seconds():
    assert duration_seconds(np.zeros(8000, dtype=np.float32)) == 0.5


def test_resolve_key_names():
    keyboard = pytest.importorskip("pynput.keyboard")
    from jarvis.ptt import resolve_key

    assert resolve_key("ctrl_r") == keyboard.Key.ctrl_r
    assert resolve_key(" F9 ") == keyboard.Key.f9
    assert resolve_key("a") == keyboard.KeyCode.from_char("a")
    with pytest.raises(ValueError):
        resolve_key("not_a_key")
    with pytest.raises(ValueError):
        resolve_key("")
