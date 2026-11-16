"""Conversation state: keeps the history, runs tools, and streams replies from the model."""

from __future__ import annotations

from typing import Any, Callable, Iterator, Protocol

from jarvis.llm import Message
from jarvis.tools import ToolBox, ToolCall


class ChatClient(Protocol):
    """Anything that can stream a reply for a list of messages."""

    def stream_chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str | ToolCall]: ...


class Conversation:
    """One ongoing chat between the user and the assistant."""

    def __init__(
        self,
        client: ChatClient,
        system_prompt: str | Callable[[], str],
        max_turns: int = 12,
        temperature: float = 0.7,
        on_exchange: Callable[[str, str], None] | None = None,
        toolbox: ToolBox | None = None,
        on_tool: Callable[[ToolCall, str], None] | None = None,
        no_action_note: str = "",
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        self._client = client
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._temperature = temperature
        self._on_exchange = on_exchange
        self._toolbox = toolbox
        self._on_tool = on_tool
        self._no_action_note = no_action_note
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
            {"role": "system", "content": self._current_system_prompt()},
            *self._recent_history(),
            user_message,
        ]
        pieces: list[str] = []
        for piece in self._reply(messages, text):
            pieces.append(piece)
            yield piece
        reply = "".join(pieces).strip()
        if reply:
            self._history.append(user_message)
            self._history.append({"role": "assistant", "content": reply})
            if self._on_exchange is not None:
                self._on_exchange(text, reply)

    def _reply(self, messages: list[Message], text: str) -> Iterator[str]:
        tools = self._toolbox.relevant(text) if self._toolbox else []
        if not tools:
            yield from self._stream_text(self._with_no_action_note(messages))
            return

        # With tools on offer the first answer is collected rather than
        # streamed: it is usually a tool call, which has no text to show.
        calls: list[ToolCall] = []
        said: list[str] = []
        for item in self._client.stream_chat(
            messages, temperature=self._temperature, tools=[tool.schema() for tool in tools]
        ):
            if isinstance(item, ToolCall):
                calls.append(item)
            else:
                said.append(item)

        if not calls:
            answer = "".join(said).strip()
            # Small models sometimes write a tool call out as text instead of
            # making it. That is not a reply; answer again without tools.
            if answer and not answer.startswith("{"):
                yield answer
            else:
                yield from self._stream_text(self._with_no_action_note(messages))
            return

        messages = [
            *messages,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": call.name, "arguments": call.arguments}} for call in calls
                ],
            },
        ]
        for call in calls:
            result = self._toolbox.call(call)
            if self._on_tool is not None:
                self._on_tool(call, result)
            messages.append({"role": "tool", "content": result})
        # The follow-up is asked without tools, so the model reports the
        # result instead of calling something else.
        yield from self._stream_text(messages)

    def _with_no_action_note(self, messages: list[Message]) -> list[Message]:
        """Remind the model, right before it answers, that nothing was done this turn.

        Without this a small model cheerfully reports "Email sent" for requests
        it has no way to carry out. A note next to the question works where an
        instruction in the system prompt does not.
        """
        if not self._no_action_note:
            return messages
        # Attached to the user's message itself: a separate system message
        # after it breaks the turn order the model was trained on.
        last = messages[-1]
        noted = {**last, "content": f"{last['content']}\n\n[{self._no_action_note}]"}
        return [*messages[:-1], noted]

    def _stream_text(self, messages: list[Message]) -> Iterator[str]:
        for item in self._client.stream_chat(messages, temperature=self._temperature):
            if isinstance(item, str):
                yield item

    def _current_system_prompt(self) -> str:
        # A callable prompt is rebuilt for every message, so newly learned
        # facts are used straight away.
        return self._system_prompt() if callable(self._system_prompt) else self._system_prompt

    def _recent_history(self) -> list[Message]:
        """The last max_turns exchanges, so the prompt stays small and fast."""
        return self._history[-2 * self._max_turns :]
