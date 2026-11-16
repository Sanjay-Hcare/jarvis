import json

from jarvis.chat import Conversation
from jarvis.memory import FactExtractor, Memory, MemoryStore, is_grounded, statements_about_self
from jarvis.persona import build_system_prompt


def test_facts_survive_a_restart(tmp_path):
    path = tmp_path / "data" / "memory.json"
    store = MemoryStore(path)

    assert store.add("Sam lives in Pune.")
    assert MemoryStore(path).facts == ["Sam lives in Pune."]


def test_duplicates_and_blanks_are_ignored(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")

    assert store.add("Sam lives in Pune.")
    assert not store.add("  sam LIVES in pune  ")
    assert not store.add("   ")
    assert store.facts == ["Sam lives in Pune."]


def test_oldest_facts_are_dropped_when_full(tmp_path):
    store = MemoryStore(tmp_path / "memory.json", max_facts=2)
    for number in range(4):
        store.add(f"Sam fact {number}")

    assert store.facts == ["Sam fact 2", "Sam fact 3"]


def test_long_facts_are_shortened(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.add("Sam " + "x" * 500)

    assert len(store.facts[0]) == 200


def test_forget_removes_matching_facts(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.add("Sam likes coffee.")
    store.add("Sam likes tea.")

    assert store.forget("COFFEE") == ["Sam likes coffee."]
    assert store.forget("") == []
    assert store.forget("cocoa") == []
    assert MemoryStore(path).facts == ["Sam likes tea."]


def test_corrupt_file_is_set_aside_not_lost(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text("{not json", encoding="utf-8")

    store = MemoryStore(path)

    assert store.facts == []
    assert (tmp_path / "memory.json.corrupt").read_text(encoding="utf-8") == "{not json"
    store.add("Sam is back.")
    assert json.loads(path.read_text(encoding="utf-8"))["facts"][0]["text"] == "Sam is back."


class FakeJSONClient:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat_json(self, messages):
        self.calls.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def test_extractor_skips_messages_not_about_the_user():
    client = FakeJSONClient({"facts": ["Sam is made up."]})

    assert FactExtractor(client, "Sam").extract("What time is it?", []) == []
    assert client.calls == []


def test_extractor_keeps_only_facts_about_the_user():
    client = FakeJSONClient({"facts": ["Sam likes tea.", "The assistant is helpful.", 42, " sam runs daily. "]})

    facts = FactExtractor(client, "Sam").extract("I like tea and I run daily", ["Sam lives in Pune."])

    assert facts == ["Sam likes tea.", "sam runs daily."]
    assert "Sam lives in Pune." in client.calls[0][1]["content"]


def test_extractor_tolerates_odd_replies():
    for reply in ({}, {"facts": "Sam likes tea."}, {"facts": None}, []):
        assert FactExtractor(FakeJSONClient(reply), "Sam").extract("I like tea", []) == []


def test_memory_learns_in_the_background_and_reports_it(tmp_path):
    learned = []
    store = MemoryStore(tmp_path / "memory.json")
    memory = Memory(store, FactExtractor(FakeJSONClient({"facts": ["Sam likes tea."]}), "Sam"), learned.append)

    memory.observe("I like tea")
    memory.observe("I like tea")
    memory.wait()
    memory.close()

    assert store.facts == ["Sam likes tea."]
    assert learned == ["Sam likes tea."]


def test_extraction_failure_does_not_break_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    memory = Memory(store, FactExtractor(FakeJSONClient(RuntimeError("model down")), "Sam"))

    memory.observe("I like tea")
    memory.wait()
    memory.close()

    assert store.facts == []


def test_prompt_lists_facts_and_forbids_inventing():
    without = build_system_prompt("Jarvis", "Sam")
    with_facts = build_system_prompt("Jarvis", "Sam", ["Sam likes tea."])

    assert "What you remember" not in without
    assert "- Sam likes tea." in with_facts
    assert "Never claim to know" in without and "Never claim to know" in with_facts


class FakeChatClient:
    def __init__(self):
        self.system_prompts = []

    def stream_chat(self, messages, temperature=0.7):
        self.system_prompts.append(messages[0]["content"])
        yield "ok"


def test_conversation_rebuilds_prompt_and_reports_exchanges():
    prompts = iter(["first prompt", "second prompt"])
    exchanges = []
    client = FakeChatClient()
    chat = Conversation(client, system_prompt=lambda: next(prompts), on_exchange=lambda u, a: exchanges.append((u, a)))

    list(chat.ask("hello"))
    list(chat.ask("again"))

    assert client.system_prompts == ["first prompt", "second prompt"]
    assert exchanges == [("hello", "ok"), ("again", "ok")]


def test_questions_are_not_treated_as_statements():
    assert statements_about_self("What is the name of my dog?") == ""
    assert statements_about_self("Can you remind me about my exam") == ""
    assert statements_about_self("Tell them a joke.") == ""
    assert (
        statements_about_self("I'm 24. What should I do with my life? My sister is Priya!")
        == "I'm 24. My sister is Priya!"
    )


def test_extractor_never_sees_questions():
    client = FakeJSONClient({"facts": ["Sam's dog's name is unknown."]})

    assert FactExtractor(client, "Sam").extract("What is the name of my dog?", []) == []
    assert client.calls == []


def test_facts_must_use_words_the_user_said():
    said = "I love playing cricket and I usually wake up at six"

    assert is_grounded("Sam loves playing cricket.", said, "Sam")
    assert is_grounded("Sam wakes up at six.", said, "Sam")
    assert not is_grounded("Sam's favourite food is biryani.", said, "Sam")
    assert not is_grounded("Sam is.", said, "Sam")


def test_ungrounded_facts_from_the_model_are_dropped():
    client = FakeJSONClient({"facts": ["Sam likes tea.", "Sam's favourite food is biryani."]})

    assert FactExtractor(client, "Sam").extract("I like tea", []) == ["Sam likes tea."]
