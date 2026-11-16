"""Push-to-talk: wait for one key to be held down and released."""

from __future__ import annotations

import threading
from typing import Any


def resolve_key(name: str) -> Any:
    """Turn a key name such as 'ctrl_r', 'f9' or 'a' into a pynput key."""
    from pynput import keyboard

    name = name.strip().lower()
    if not name:
        raise ValueError("push_to_talk_key must not be empty")
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ValueError(
        f"Unknown push_to_talk_key {name!r}. Use a name like ctrl_r, alt_r, f9, or a single character."
    )


class PushToTalk:
    """Tracks whether the talk key is currently held, without blocking other keys."""

    def __init__(self, key_name: str = "ctrl_r") -> None:
        from pynput import keyboard

        self._key = resolve_key(key_name)
        self._pressed = threading.Event()
        self._released = threading.Event()
        self._released.set()
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def start(self) -> None:
        self._listener.start()
        self._listener.wait()

    def stop(self) -> None:
        self._listener.stop()

    def wait_for_press(self, timeout: float | None = None) -> bool:
        return self._pressed.wait(timeout)

    def wait_for_release(self, timeout: float | None = None) -> bool:
        return self._released.wait(timeout)

    def _on_press(self, key: Any) -> None:
        if key == self._key:
            self._released.clear()
            self._pressed.set()

    def _on_release(self, key: Any) -> None:
        if key == self._key:
            self._pressed.clear()
            self._released.set()
