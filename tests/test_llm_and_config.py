import json

import pytest

from jarvis.config import Settings, load_settings
from jarvis.llm import LLMError, iter_content
from jarvis.persona import build_system_prompt


def line(content="", done=False, **extra):
    return json.dumps({"message": {"content": content}, "done": done, **extra}).encode()


def test_iter_content_yields_pieces_until_done():
    lines = [line("Hel"), b"", line("lo"), line("", done=True), line("ignored")]

    assert list(iter_content(lines)) == ["Hel", "lo"]


def test_iter_content_raises_on_server_error():
    with pytest.raises(LLMError, match="model not found"):
        list(iter_content([json.dumps({"error": "model not found"}).encode()]))


def test_defaults_when_no_config_file(tmp_path):
    assert load_settings(tmp_path / "missing.toml", env={}) == Settings()


def test_file_then_environment_override(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('user_name = "Sam"\nmodel = "from-file"\nmax_turns = 4\n', encoding="utf-8")

    settings = load_settings(path, env={"JARVIS_MODEL": "from-env", "JARVIS_TEMPERATURE": "0.2"})

    assert settings.user_name == "Sam"
    assert settings.model == "from-env"
    assert settings.max_turns == 4
    assert settings.temperature == 0.2


def test_unknown_setting_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('modle = "typo"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="modle"):
        load_settings(path, env={})


def test_system_prompt_uses_both_names():
    prompt = build_system_prompt("Jarvis", "Sam")

    assert "You are Jarvis" in prompt
    assert "Call the user Sam" in prompt


def test_boolean_settings_from_environment(tmp_path):
    assert load_settings(tmp_path / "none.toml", env={"JARVIS_SPEAK": "false"}).speak is False
    assert load_settings(tmp_path / "none.toml", env={"JARVIS_SPEAK": "On"}).speak is True
    with pytest.raises(ValueError):
        load_settings(tmp_path / "none.toml", env={"JARVIS_SPEAK": "maybe"})
