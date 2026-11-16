"""Streaming client for a local Ollama server. Uses only the standard library."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Iterable, Iterator

from jarvis.tools import ToolCall

Message = dict[str, Any]


class LLMError(RuntimeError):
    """The language model could not produce a reply."""


def iter_content(lines: Iterable[bytes]) -> Iterator[str | ToolCall]:
    """Yield text pieces and tool calls from Ollama's newline-delimited JSON chat stream."""
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        event = json.loads(raw)
        if "error" in event:
            raise LLMError(str(event["error"]))
        message = event.get("message", {})
        piece = message.get("content", "")
        if piece:
            yield piece
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            arguments = function.get("arguments")
            yield ToolCall(
                name=str(function.get("name", "")),
                arguments=arguments if isinstance(arguments, dict) else {},
            )
        if event.get("done"):
            return


class OllamaClient:
    """Talks to Ollama's /api/chat endpoint and streams the reply as it is generated."""

    def __init__(
        self,
        host: str,
        model: str,
        keep_alive: str = "30m",
        timeout: float = 120.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.keep_alive = keep_alive
        self.timeout = timeout

    def _request(self, payload: dict[str, Any]) -> urllib.request.Request:
        body = json.dumps({"model": self.model, "keep_alive": self.keep_alive, **payload})
        return urllib.request.Request(
            f"{self.host}/api/chat",
            data=body.encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    def chat_json(self, messages: list[Message]) -> dict[str, Any]:
        """Ask for a complete reply that is guaranteed to be a JSON object."""
        request = self._request(
            {"messages": messages, "stream": False, "format": "json", "options": {"temperature": 0}}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = json.load(response).get("message", {}).get("content", "")
            parsed = json.loads(content)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise LLMError(f"The model did not return usable JSON: {exc}") from exc
        return parsed if isinstance(parsed, dict) else {}

    def stream_chat(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[str | ToolCall]:
        """Send the conversation and yield the reply piece by piece.

        When tools are offered, the stream may contain ToolCall items instead
        of (or as well as) text.
        """
        payload: dict[str, Any] = {
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        request = self._request(payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                yield from iter_content(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace").strip()
            raise LLMError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.host}. Is it running? ({exc})"
            ) from exc
