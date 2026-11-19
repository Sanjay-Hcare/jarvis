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
    duration = len(heard[-1]) / 16000
    # 1.6 s of audio from before the wake word, the 1.5 s command and the
    # 1 s of quiet that ended it; the later speech is not included.
    assert 3.9 <= duration <= 4.3


def test_a_short_pause_does_not_end_the_command():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(0.6, QUIET) + seconds(1, SPEECH) + seconds(2, QUIET)

    _commands, heard, _woke, _model = listen(iter(frames))

    # Both halves of the request are kept.
    assert len(heard[-1]) / 16000 >= 1.6 + 1 + 0.6 + 1


def test_the_audio_passed_on_is_float_in_range():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1, QUIET)

    _commands, heard, _woke, _model = listen(iter(frames))

    assert all(audio.dtype == np.float32 for audio in heard)
    assert 0 < np.abs(heard[-1]).max() <= 1


def test_waiting_for_a_command_gives_up():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(10, QUIET)

    commands, heard, _woke, _model = listen(iter(frames), transcript="Hey Jarvis", start_timeout=2)

    assert commands == [""]
    assert len(heard[-1]) / 16000 <= 1.6 + 2.1


def test_a_long_command_is_cut_off():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(30, SPEECH)

    _commands, heard, _woke, _model = listen(iter(frames), max_seconds=5)

    assert len(heard[-1]) / 16000 <= 5.1


def test_false_wake_ups_are_reported_as_none():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1, QUIET)

    commands, heard, woke, model = listen(iter(frames), transcript="Hey Travis, what's up?")

    assert commands == [None]
    # Rejected on the first look, before any beep and without recording on.
    assert woke == [] and len(heard) == 1
    assert len(heard[0]) / 16000 <= 1.6 + 0.6


def test_the_name_is_checked_before_the_beep():
    events = []
    model = FakeModel()

    def transcribe(audio):
        events.append(("transcribe", round(len(audio) / 16000, 1)))
        return "Hey Jarvis, what time is it?"

    frames = seconds(2, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1.5, QUIET)
    listener = WakeListener(model, transcribe, on_wake=lambda: events.append(("beep",)))

    assert list(listener.commands(iter(frames))) == ["what time is it?"]
    (first, first_seconds), beep, (second, _) = events
    assert (first, beep, second) == ("transcribe", ("beep",), "transcribe")
    # The first look covers the moments before the detection plus the end of the name.
    assert 1.6 <= first_seconds <= 2.2


def test_the_full_request_is_kept_even_if_the_name_is_misheard_the_second_time():
    transcripts = iter(["Hey Jarvis, what", "Hey, Jobbys what time is it"])
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1.5, QUIET)
    listener = WakeListener(FakeModel(), lambda audio: next(transcripts))

    assert list(listener.commands(iter(frames))) == ["what time is it"]


def test_the_first_look_is_used_if_the_full_transcript_has_no_words_after_greetings():
    transcripts = iter(["Hey Jarvis, what", "Hey."])
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1.5, QUIET)
    listener = WakeListener(FakeModel(), lambda audio: next(transcripts))

    assert list(listener.commands(iter(frames))) == ["what"]


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



def test_a_sound_that_keeps_firing_is_not_checked_on_every_frame():
    class AlwaysFires(FakeModel):
        def score(self, frame):
            return 1.0

    checks = []
    listener = WakeListener(AlwaysFires(), lambda audio: checks.append(1) or "Hey Travis")

    results = list(listener.commands(iter(seconds(4, SPEECH))))

    # Each check uses 0.5 s of new audio and is followed by a 0.3 s pause.
    assert results == [None] * len(checks)
    assert 4 <= len(checks) <= 6


def test_a_second_look_after_an_early_detection_still_wakes():
    transcripts = iter(["Hey Jar", "Hey Jarvis, open YouTube", "Hey Jarvis, open YouTube"])
    model = FakeModel()
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(1.0, SPEECH) + [frame(SPEECH, wake=True)] + seconds(1, SPEECH) + seconds(1.5, QUIET)
    listener = WakeListener(model, lambda audio: next(transcripts))

    assert list(listener.commands(iter(frames))) == [None, "open YouTube"]


@pytest.mark.parametrize(
    "transcript, command",
    [
        ("Hey, Javish. What is the top 5 global news?", "What is the top 5 global news?"),
        ("Hey Jervis, what time is it?", "what time is it?"),
        ("OK, hey Garvis open YouTube", "open YouTube"),
        ("Hey, Jobbys, what is the weather?", "what is the weather?"),
        ("Hey.", None),
        # Real transcripts where the name was left out: the first word is kept.
        ("What time it is?", "What time it is?"),
        ("Hey, what time is it?", "what time is it?"),
    ],
)
def test_once_the_name_is_confirmed_any_spelling_of_it_is_accepted(transcript, command):
    assert command_after_wake_word(transcript, name_confirmed=True) == command


def test_a_respelt_name_in_the_full_transcript_keeps_the_request():
    transcripts = iter(["Hey Jarvis.", "Hey, Javish. What is the top 5 global news happening right now?"])
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(3, SPEECH) + seconds(1.5, QUIET)
    listener = WakeListener(FakeModel(), lambda audio: next(transcripts))

    assert list(listener.commands(iter(frames))) == ["What is the top 5 global news happening right now?"]


def test_a_pause_after_the_first_word_waits_for_the_rest():
    # "Hey Jarvis, what" ... 1.5 s waiting for the beep ... "time is it?"
    frames = (
        seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(0.4, SPEECH) + seconds(0.3, SPEECH)
        + seconds(1.5, QUIET) + seconds(1.0, SPEECH) + seconds(1.5, QUIET)
    )

    _commands, heard, _woke, _model = listen(iter(frames))

    # The words after the pause are part of the request.
    assert len(heard[-1]) / 16000 >= 1.6 + 0.4 + 0.3 + 1.5 + 1.0


def test_a_quiet_voice_is_not_kept_waiting_after_a_whole_request():
    # A quiet voice: only every other moment is loud enough to count as speech.
    quiet_voice = [frame(SPEECH if i % 2 else QUIET) for i in range(round(1.2 / 0.08))]
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(0.5, SPEECH) + quiet_voice + seconds(3, QUIET)

    _commands, heard, _woke, _model = listen(iter(frames))

    # Ends after the normal 1 s pause, not the longer early-pause wait.
    assert len(heard[-1]) / 16000 <= 1.6 + 0.5 + 1.2 + 1.0 + 0.2


def test_a_long_silence_after_a_short_start_still_ends_the_request():
    frames = seconds(1, QUIET) + [frame(SPEECH, wake=True)] + seconds(0.4, SPEECH) + seconds(0.3, SPEECH) + seconds(10, QUIET)

    _commands, heard, _woke, _model = listen(iter(frames))

    assert len(heard[-1]) / 16000 <= 1.6 + 0.7 + 2.6
