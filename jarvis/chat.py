"""Conversation state: keeps the history and streams replies from the model."""

from __future__ import annotations

from typing import Iterator, Protocol

from jarvis.llm import Message


class ChatClient(Protocol):
    """Anything that can stream a reply for a list of messages."""

    def stream_chat(self, messages: list[Message], temperature: float = 0.7) -> Iterator[str]: ...


class Conversation:
    """One ongoing chat between the user and the assistant."""

    def __init__(
        self,
        client: ChatClient,
        system_prompt: str,
        max_turns: int = 12,
        temperature: float = 0.7,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._client = client
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._temperature = temperature
        self._history: list[Message] = []

    @property
    def history(self) -> list[Message]:
        """The user/assistant messages so far, oldest first (a copy)."""
        return list(self._history)

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._history.clear()

    def ask(self, text: str) -> Iterator[str]:
        """Send the user's message and yield the reply as it is generated.

        The exchange is added to the history only once the reply has finished,
        so a failed or interrupted reply leaves the history unchanged.
        """
        text = text.strip()
        if not text:
            return
        user_message: Message = {"role": "user", "content": text}
        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._recent_history(),
            user_message,
        ]
        pieces: list[str] = []
        for piece in self._client.stream_chat(messages, temperature=self._temperature):
            pieces.append(piece)
            yield piece
        reply = "".join(pieces).strip()
        if reply:
            self._history.append(user_message)
            self._history.append({"role": "assistant", "content": reply})

    def _recent_history(self) -> list[Message]:
        """The last max_turns exchanges, so the prompt stays small and fast."""
        return self._history[-2 * self._max_turns :]
