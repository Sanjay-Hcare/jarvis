"""Streaming client for a local Ollama server. Uses only the standard library."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterable, Iterator

Message = dict[str, str]


class LLMError(RuntimeError):
    """The language model could not produce a reply."""


def iter_content(lines: Iterable[bytes]) -> Iterator[str]:
    """Yield text pieces from Ollama's newline-delimited JSON chat stream."""
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        event = json.loads(raw)
        if "error" in event:
            raise LLMError(str(event["error"]))
        piece = event.get("message", {}).get("content", "")
        if piece:
            yield piece
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

    def stream_chat(self, messages: list[Message], temperature: float = 0.7) -> Iterator[str]:
        """Send the conversation and yield the reply piece by piece."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "keep_alive": self.keep_alive,
                "options": {"temperature": temperature},
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
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
