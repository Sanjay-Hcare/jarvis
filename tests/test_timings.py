import numpy as np

from jarvis.voice import _answer, reply_aloud
from jarvis.wake import FRAME_SAMPLES, WakeListener


class FakeConversation:
    def __init__(self, pieces):
        self.pieces = pieces

    def ask(self, text):
        yield from self.pieces


class FakeSpeaker:
    def __init__(self):
        self.said = []

    def say(self, text):
        self.said.append(text)


def test_reply_timings_are_recorded_in_order(capsys):
    timings = {}
    speaker = FakeSpeaker()

    pieces = ["It is nine in the morning. ", "Have a ", "good day."]
    reply_aloud(FakeConversation(pieces), "time?", "Jarvis", speaker, timings)

    assert timings["first_text"] <= timings["first_sentence"] <= timings["done"]
    assert speaker.said == ["It is nine in the morning.", "Have a good day."]
    assert capsys.readouterr().out == "Jarvis: It is nine in the morning. Have a good day.\n"


def test_answer_logs_reply_timing(capsys):
    lines = []

    _answer(FakeConversation(["Done."]), "lock it", "Jarvis", FakeSpeaker(), lines.append)

    assert len(lines) == 1
    assert lines[0].startswith("reply timing: first words of the reply ")
    assert "first sentence to the voice" in lines[0] and "reply finished" in lines[0]


def test_wake_log_includes_how_long_recognition_took():
    class Model:
        def __init__(self):
            self.fired = False

        def score(self, frame):
            if frame[0] == 9999 and not self.fired:
                self.fired = True
                return 1.0
            return 0.0

        def reset(self):
            pass

    def frames(level, count, wake=False):
        out = [np.full(FRAME_SAMPLES, level, dtype=np.int16) for _ in range(count)]
        if wake:
            out[0][0] = 9999
        return out

    lines = []
    audio = frames(30, 12) + frames(3000, 1, wake=True) + frames(3000, 25) + frames(30, 20)
    listener = WakeListener(Model(), lambda a: "Hey Jarvis, what time is it?", log=lines.append)

    assert list(listener.commands(iter(audio))) == ["what time is it?"]
    woke = [line for line in lines if line.startswith("woke")]
    assert len(woke) == 1 and "name check " in woke[0] and "transcription " in woke[0]
