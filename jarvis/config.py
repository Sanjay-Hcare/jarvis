"""Settings, loaded from an optional config.toml and overridable by environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class Settings:
    """Everything the assistant needs to know to start a conversation."""

    assistant_name: str = "Jarvis"
    user_name: str = "friend"
    model: str = "llama3.2:3b"
    ollama_host: str = "http://127.0.0.1:11434"
    temperature: float = 0.7
    # How long Ollama keeps the model in memory after a reply. A warm model
    # answers in well under a second; a cold one takes ~10s to load.
    keep_alive: str = "30m"
    # Number of past user/assistant exchanges sent with each request.
    max_turns: int = 12
    request_timeout: float = 120.0


def load_settings(path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Build settings from defaults, then config.toml, then JARVIS_* environment variables."""
    path = path or DEFAULT_CONFIG_PATH
    env = os.environ if env is None else env
    values: dict[str, object] = {}

    if path.is_file():
        with path.open("rb") as handle:
            file_values = tomllib.load(handle)
        known = {f.name for f in fields(Settings)}
        unknown = sorted(set(file_values) - known)
        if unknown:
            raise ValueError(f"Unknown setting(s) in {path}: {', '.join(unknown)}")
        values.update(file_values)

    for field in fields(Settings):
        raw = env.get(f"JARVIS_{field.name.upper()}")
        if raw is not None:
            values[field.name] = raw

    # Coerce to the declared types so "0.5" from the environment becomes 0.5.
    casts = {"temperature": float, "max_turns": int, "request_timeout": float}
    for name, cast in casts.items():
        if name in values:
            values[name] = cast(values[name])

    return Settings(**values)
