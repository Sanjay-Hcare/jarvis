"""Long-term memory: facts about the user that survive between conversations."""

from __future__ import annotations

import json
import os
import queue
import re
import threading
from datetime import date
from pathlib import Path
from typing import Any, Callable, Protocol

MAX_FACT_CHARS = 200
_WORD = re.compile(r"[a-z0-9']+")


def _normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


class MemoryStore:
    """A small list of facts kept in a JSON file."""

    def __init__(self, path: Path, max_facts: int = 100) -> None:
        self._path = Path(path)
        self._max_facts = max_facts
        self._lock = threading.Lock()
        self._facts: list[dict[str, str]] = self._load()

    @property
    def facts(self) -> list[str]:
        with self._lock:
            return [fact["text"] for fact in self._facts]

    def add(self, text: str) -> bool:
        """Remember a fact. Returns False if it was empty or already known."""
        text = " ".join(str(text).split())[:MAX_FACT_CHARS].strip()
        key = _normalise(text)
        if not key:
            return False
        with self._lock:
            if any(_normalise(fact["text"]) == key for fact in self._facts):
                return False
            self._facts.append({"text": text, "added": date.today().isoformat()})
            # When full, the oldest facts make room for new ones.
            del self._facts[: max(0, len(self._facts) - self._max_facts)]
            self._save()
        return True

    def forget(self, query: str) -> list[str]:
        """Remove every fact containing all the words in query; return what was removed."""
        words = _normalise(query).split()
        if not words:
            return []
        with self._lock:
            removed = [f["text"] for f in self._facts if all(w in _normalise(f["text"]).split() for w in words)]
            if removed:
                self._facts = [f for f in self._facts if f["text"] not in removed]
                self._save()
        return removed

    def _load(self) -> list[dict[str, str]]:
        if not self._path.is_file():
            return []
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return [
                {"text": str(item["text"]), "added": str(item.get("added", ""))}
                for item in data.get("facts", [])
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
        except (json.JSONDecodeError, OSError, AttributeError, TypeError):
            # Keep the unreadable file for inspection instead of overwriting it.
            backup = self._path.with_suffix(self._path.suffix + ".corrupt")
            try:
                os.replace(self._path, backup)
            except OSError:
                pass
            return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"facts": self._facts}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # Rename is atomic, so a crash never leaves a half-written memory file.
        os.replace(temporary, self._path)


class JSONClient(Protocol):
    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


EXTRACTION_PROMPT = """\
You keep a notebook of lasting facts about a person named {user_name}. You are \
given one message that {user_name} said to an assistant. Reply with JSON of the \
form {{"facts": ["..."]}} listing facts worth remembering for months.

Record a fact ONLY when {user_name} states it directly about themselves: their \
name, age, home, work or studies, family and friends, likes and dislikes, \
goals, routines, or how they want the assistant to behave.

Do NOT record: questions, requests, commands, greetings, opinions about the \
assistant, things happening only right now (mood, weather, the current task), \
or anything you have to guess. Never record something already in the notebook.

Write each fact as one short sentence that starts with "{user_name}". If there \
is nothing to record, reply {{"facts": []}}. Most messages contain nothing to \
record.

Examples:
Message: "What's the weather like today?"
{{"facts": []}}
Message: "I'm a final year computer science student and I love playing cricket."
{{"facts": ["{user_name} is a final year computer science student.", "{user_name} loves playing cricket."]}}
Message: "Can you tell me a joke? I'm bored."
{{"facts": []}}
Message: "Remember that my sister's name is Priya."
{{"facts": ["{user_name}'s sister is named Priya."]}}
Message: "Thanks, that was helpful."
{{"facts": []}}"""

# Words that suggest the user is talking about themselves. Sentences without
# any of them are skipped, which saves a model call and avoids invented facts.
_ABOUT_SELF = re.compile(r"\b(i|i'm|im|i've|i'd|i'll|my|me|mine|myself|we|our|remember)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?]+[.!?]*")
_QUESTION_START = re.compile(
    r"^(who|what|when|where|why|how|which|can|could|do|does|did|is|are|am|was|were|"
    r"should|would|will|shall|may|have|has)\b",
    re.IGNORECASE,
)
# Words too common to show that a fact really came from the user's message.
_FILLER = frozenset(
    "a an the is are was were be been am and or but of to in on at for with by from as it its "
    "this that his her their him he she they them has have had does do did not no "
    "named name called year years old who which also very really".split()
)


def statements_about_self(text: str) -> str:
    """Keep only the sentences that state something about the speaker.

    Questions are dropped: "What is my dog's name?" tells us nothing, and a
    small model will happily turn it into a fact.
    """
    kept = []
    for sentence in _SENTENCE.findall(text):
        sentence = sentence.strip()
        if not sentence or sentence.endswith("?") or _QUESTION_START.match(sentence):
            continue
        if _ABOUT_SELF.search(sentence):
            kept.append(sentence)
    return " ".join(kept)


def is_grounded(fact: str, source: str, user_name: str) -> bool:
    """True when most of the fact's meaningful words appear in what the user said."""
    source_words = set(_WORD.findall(source.lower()))
    name_words = set(_WORD.findall(user_name.lower()))
    skip = _FILLER | name_words | {f"{word}'s" for word in name_words}
    words = [word for word in _WORD.findall(fact.lower()) if word not in skip]
    if not words:
        return False

    def present(word: str) -> bool:
        # Compare word stems loosely so "loves" matches "love" and "wakes" matches "wake".
        stem = word[:4]
        return any(
            w == word
            or (len(stem) == 4 and w.startswith(stem))
            or (len(w) >= 3 and word.startswith(w))
            for w in source_words
        )

    return sum(present(word) for word in words) / len(words) >= 0.6


class FactExtractor:
    """Asks the model which lasting facts, if any, a user message contains."""

    def __init__(self, client: JSONClient, user_name: str) -> None:
        self._client = client
        self._user_name = user_name

    def extract(self, user_text: str, known: list[str]) -> list[str]:
        statements = statements_about_self(user_text)
        if not statements:
            return []
        notebook = "\n".join(f"- {fact}" for fact in known) or "(empty)"
        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT.format(user_name=self._user_name)},
            {"role": "user", "content": f"Notebook so far:\n{notebook}\n\nMessage: {json.dumps(statements)}"},
        ]
        reply = self._client.chat_json(messages)
        facts = reply.get("facts", []) if isinstance(reply, dict) else []
        if not isinstance(facts, list):
            return []
        prefix = self._user_name.lower()
        return [
            fact.strip()
            for fact in facts
            # A fact that is not about the user, or that uses words the user never
            # said, is a sign the model went off-script.
            if isinstance(fact, str)
            and fact.strip().lower().startswith(prefix)
            and is_grounded(fact, statements, self._user_name)
        ]


class Memory:
    """Ties the store and the extractor together and learns in the background."""

    def __init__(
        self,
        store: MemoryStore,
        extractor: FactExtractor,
        on_learned: Callable[[str], None] | None = None,
    ) -> None:
        self.store = store
        self._extractor = extractor
        self._on_learned = on_learned
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="jarvis-memory", daemon=True)
        self._thread.start()

    def observe(self, user_text: str, _reply: str = "") -> None:
        """Look at what the user said, later, without delaying the conversation."""
        self._queue.put(user_text)

    def wait(self) -> None:
        """Block until everything observed so far has been processed."""
        self._queue.join()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            text = self._queue.get()
            try:
                if text is None:
                    return
                for fact in self._extractor.extract(text, self.store.facts):
                    if self.store.add(fact) and self._on_learned:
                        self._on_learned(fact)
            except Exception:
                # Memory is a nice-to-have: a failed extraction must never break the chat.
                pass
            finally:
                self._queue.task_done()
