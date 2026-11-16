import threading

import pytest

from jarvis.sentences import SentenceSplitter
from jarvis.tts import Speaker, TTSError, clean_for_speech, ensure_voice, voice_urls


def split(pieces, **kwargs):
    splitter = SentenceSplitter(**kwargs)
    sentences = []
    for piece in pieces:
        sentences.extend(splitter.feed(piece))
    rest = splitter.flush()
    return sentences + ([rest] if rest else [])


def test_sentences_are_released_as_soon_as_they_end():
    splitter = SentenceSplitter()

    assert splitter.feed("The capital of France ") == []
    assert splitter.feed("is Paris. It is") == ["The capital of France is Paris."]
    assert splitter.feed(" lovely!") == []
    assert splitter.flush() == "It is lovely!"


def test_pieces_can_split_anywhere():
    text = "I checked your calendar. You are free today! Want to go out?"

    assert split(list(text)) == [
        "I checked your calendar.",
        "You are free today!",
        "Want to go out?",
    ]


def test_numbers_and_addresses_are_not_cut():
    assert split(["It costs 3.5 dollars at example.com today. Nice."]) == [
        "It costs 3.5 dollars at example.com today.",
        "Nice.",
    ]


def test_short_fragments_join_the_next_sentence():
    assert split(["Sure. I can do that for you. Done."]) == ["Sure. I can do that for you.", "Done."]


def test_abbreviations_do_not_end_a_sentence():
    assert split(["Please ask Dr. Rao about it tomorrow. Thanks a lot."]) == [
        "Please ask Dr. Rao about it tomorrow.",
        "Thanks a lot.",
    ]


def test_flush_on_empty_buffer():
    assert SentenceSplitter().flush() == ""


def test_clean_for_speech_strips_markup():
    assert clean_for_speech("**Hello**, `world`!\n# Title") == "Hello, world! Title"
    assert clean_for_speech("use snake_case") == "use snake case"
    assert clean_for_speech("  *  ") == ""


def test_voice_urls():
    urls = voice_urls("en_GB-alan-medium")

    assert urls["en_GB-alan-medium.onnx"].endswith("/en/en_GB/alan/medium/en_GB-alan-medium.onnx")
    assert urls["en_GB-alan-medium.onnx.json"].endswith("/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json")
    with pytest.raises(TTSError):
        voice_urls("alan")


def test_ensure_voice_downloads_only_missing_files(tmp_path):
    (tmp_path / "en_GB-alan-medium.onnx").write_bytes(b"model")
    fetched = []

    def download(url, target):
        fetched.append(target.name)
        target.write_bytes(b"x")

    path = ensure_voice("en_GB-alan-medium", tmp_path, download=download)

    assert path == tmp_path / "en_GB-alan-medium.onnx"
    assert fetched == ["en_GB-alan-medium.onnx.json"]


class FakeSynthesizer:
    def synthesize(self, text):
        if text == "boom":
            raise RuntimeError("synthesis broke")
        return [text]


class FakePlayer:
    """Records what was played; can block until told to continue."""

    def __init__(self, block=False):
        self.played = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = block

    def play(self, audio, should_stop):
        self.started.set()
        if self.block:
            while not should_stop() and not self.release.wait(0.01):
                pass
            if should_stop():
                self.played.append(("stopped", audio[0]))
                return
        self.played.append(("done", audio[0]))


def test_speaker_plays_sentences_in_order():
    player = FakePlayer()
    speaker = Speaker(FakeSynthesizer(), player)
    for sentence in ("one", "two", "three"):
        speaker.say(sentence)

    assert speaker.wait(2)
    assert player.played == [("done", "one"), ("done", "two"), ("done", "three")]
    speaker.close()


def test_stop_interrupts_and_drops_the_queue():
    player = FakePlayer(block=True)
    speaker = Speaker(FakeSynthesizer(), player)
    speaker.say("first")
    speaker.say("second")
    assert player.started.wait(2)

    speaker.stop()

    assert speaker.wait(2)
    assert player.played == [("stopped", "first")]
    assert not speaker.speaking

    # It still works after being interrupted.
    player.block = False
    speaker.say("third")
    assert speaker.wait(2)
    assert player.played[-1] == ("done", "third")
    speaker.close()


def test_a_failed_sentence_does_not_kill_the_speaker(capsys):
    player = FakePlayer()
    speaker = Speaker(FakeSynthesizer(), player)
    speaker.say("boom")
    speaker.say("still here")

    assert speaker.wait(2)
    assert player.played == [("done", "still here")]
    assert "speech failed" in capsys.readouterr().out
    speaker.close()


def test_blank_text_is_not_queued():
    speaker = Speaker(FakeSynthesizer(), FakePlayer())
    speaker.say("  **  ")

    assert not speaker.speaking
    speaker.close()
