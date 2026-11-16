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

    def problem(self, fetch: Any = None) -> str | None:
        """Return what is wrong with the setup, in words a user can act on, or None."""
        fetch = fetch or (lambda url: urllib.request.urlopen(url, timeout=5))
        try:
            with fetch(f"{self.host}/api/tags") as response:
                installed = [str(item.get("name", "")) for item in json.load(response).get("models", [])]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return (
                f"Ollama is not running at {self.host}. Start it (open the Ollama app, or run "
                "'ollama serve') and try again. Get it from https://ollama.com if it is not installed."
            )
        # "llama3.2" on its own means the "latest" tag.
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        if wanted not in installed:
            return f"The model {self.model} is not installed. Run:  ollama pull {self.model}"
        return None

    def preload(self) -> None:
        """Ask Ollama to load the model into memory in the background; never raises."""
        import threading

        def load() -> None:
            try:
                with urllib.request.urlopen(self._request({"messages": []}), timeout=self.timeout):
                    pass
            except (urllib.error.URLError, OSError):
                pass

        threading.Thread(target=load, name="jarvis-preload", daemon=True).start()

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
