import numpy as np
import pytest

from jarvis.wake import FRAME_SAMPLES, ListenSettings, WakeListener, command_after_wake_word, ensure_wake_models


@pytest.mark.parametrize(
    "transcript, command",
    [
        ("Hey Jarvis, what time is it?", "what time is it?"),
        ("hey jarvis open youtube", "open youtube"),
        ("Jarvis. Turn the volume up.", "Turn the volume up."),
        ("OK so Jarvis, any new mail?", "any new mail?"),
        ("Hey Javis, lock my laptop", "lock my laptop"),
        ("Hey Jarvis.", ""),
    ],
)
def test_commands_are_taken_from_after_the_name(transcript, command):
    assert command_after_wake_word(transcript) == command


@pytest.mark.parametrize(
    "transcript",
    ["Hey Travis, what's up?", "Hey Charles", "Hey, jar of jam please", "I think Jarvis is a great name", ""],
)
def test_other_words_are_not_the_wake_word(transcript):
    assert command_after_wake_word(transcript) is None


class FakeModel:
    """Fires on frames marked with the value 9999."""

    def __init__(self):
        self.resets = 0

    def score(self, frame):
        return 1.0 if frame[0] == 9999 else 0.0

    def reset(self):
        self.resets += 1


def frame(level=0, wake=False):
    samples = np.full(FRAME_SAMPLES, level, dtype=np.int16)
    if wake:
        samples[0] = 9999
    return samples


def seconds(count, level=0):
    return [frame(level) for _ in range(round(count / 0.08))]


QUIET, SPEECH = 30, 3000


def listen(frames, transcript="Hey Jarvis, what time is it?", **settings):
    heard = []

    def transcribe(audio):
        heard.append(audio)
        return transcript

    model = FakeModel()
    woke = []
    listener = WakeListener(model, transcribe, ListenSettings(**settings), on_wake=lambda: woke.append(True))
    return list(listener.commands(frames)), heard, woke, model


def test_command_is_recorded_until_the_speaker_stops():
    frames = seconds(2, QUIET) + [frame(SPEECH, wake=True)] + seconds(1.5, SPEECH) + seconds(1.5, QUIET) + seconds(3, SPEECH)

    commands, heard, woke, model = listen(iter(frames))

    assert commands == ["what time is it?"]
    assert woke == [True] and model.resets == 1
    duration = len(heard[0]) / 16000
    # 1.6 s of audio from before the wake word, the 1.5 s command and the
    # 1 s of quiet that ended it; the later speech is not included.
    assert 3.9 <= duration <= 4.3


def test_a_short_pause_does_not_end_the_command():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(0.6, QUIET) + seconds(1, SPEECH) + seconds(2, QUIET)

    _commands, heard, _woke, _model = listen(iter(frames))

    # Both halves of the request are kept.
    assert len(heard[0]) / 16000 >= 1.6 + 1 + 0.6 + 1


def test_the_audio_passed_on_is_float_in_range():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1, QUIET)

    _commands, heard, _woke, _model = listen(iter(frames))

    assert heard[0].dtype == np.float32
    assert 0 < np.abs(heard[0]).max() <= 1


def test_waiting_for_a_command_gives_up():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(10, QUIET)

    commands, heard, _woke, _model = listen(iter(frames), transcript="Hey Jarvis", start_timeout=2)

    assert commands == [""]
    assert len(heard[0]) / 16000 <= 1.6 + 2.1


def test_a_long_command_is_cut_off():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(30, SPEECH)

    _commands, heard, _woke, _model = listen(iter(frames), max_seconds=5)

    assert len(heard[0]) / 16000 <= 5.1


def test_false_wake_ups_are_reported_as_none():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1, QUIET)

    commands, _heard, _woke, _model = listen(iter(frames), transcript="Hey Travis, what's up?")

    assert commands == [None]


def test_no_wake_word_means_nothing_happens():
    commands, heard, woke, _model = listen(iter(seconds(5, SPEECH)))

    assert commands == [] and heard == [] and woke == []


def test_models_are_downloaded_once(tmp_path):
    fetched = []

    def download(url, target):
        fetched.append(target.name)
        target.write_bytes(b"x")

    ensure_wake_models(tmp_path, download=download)
    ensure_wake_models(tmp_path, download=download)

    assert fetched == ["melspectrogram.onnx", "embedding_model.onnx", "hey_jarvis_v0.1.onnx"]
