import pytest

from jarvis.chat import Conversation


class FakeClient:
    """Records what it was sent and replays a scripted reply."""

    def __init__(self, pieces=("Hi ", "there."), fail_after=None):
        self.pieces = pieces
        self.fail_after = fail_after
        self.calls = []

    def stream_chat(self, messages, temperature=0.7):
        self.calls.append((messages, temperature))
        for index, piece in enumerate(self.pieces):
            if self.fail_after is not None and index >= self.fail_after:
                raise RuntimeError("connection lost")
            yield piece


def test_reply_is_streamed_and_recorded():
    client = FakeClient()
    chat = Conversation(client, system_prompt="be nice")

    assert list(chat.ask("hello")) == ["Hi ", "there."]
    assert chat.history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there."},
    ]


def test_system_prompt_and_history_are_sent_in_order():
    client = FakeClient()
    chat = Conversation(client, system_prompt="be nice", temperature=0.2)
    list(chat.ask("first"))
    list(chat.ask("second"))

    messages, temperature = client.calls[-1]
    assert temperature == 0.2
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "be nice"
    assert messages[-1]["content"] == "second"


def test_only_recent_turns_are_sent():
    client = FakeClient()
    chat = Conversation(client, system_prompt="s", max_turns=2)
    for number in range(5):
        list(chat.ask(f"q{number}"))

    messages, _ = client.calls[-1]
    sent_questions = [m["content"] for m in messages if m["role"] == "user"]
    assert sent_questions == ["q2", "q3", "q4"]


def test_failed_reply_leaves_history_unchanged():
    chat = Conversation(FakeClient(fail_after=1), system_prompt="s")

    with pytest.raises(RuntimeError):
        list(chat.ask("hello"))
    assert chat.history == []


def test_blank_input_is_ignored():
    client = FakeClient()
    chat = Conversation(client, system_prompt="s")

    assert list(chat.ask("   ")) == []
    assert client.calls == []


def test_reset_clears_history():
    chat = Conversation(FakeClient(), system_prompt="s")
    list(chat.ask("hello"))
    chat.reset()

    assert chat.history == []


def test_max_turns_must_be_positive():
    with pytest.raises(ValueError):
        Conversation(FakeClient(), system_prompt="s", max_turns=0)
