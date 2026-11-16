"""Tools: things the assistant can do, and the rules for when to offer them to the model."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


# Every tool reports failure as a result that starts with this.
ERROR_PREFIX = "Error:"


@dataclass(frozen=True)
class ToolCall:
    """A request from the model to run a tool."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    run: Callable[..., str]
    # JSON schema of the arguments, in the form the model expects.
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})
    # The tool is offered to the model only when the user's message matches.
    # A small local model calls a tool for almost anything if one is on
    # offer ("tell me a joke" -> get_time), so the decision of *whether* a
    # tool could be relevant is made here, and the model only decides *which*
    # tool and with what arguments.
    triggers: str = ""
    # For actions (as opposed to lookups) there is nothing to summarise, and a
    # small model asked to report "Opened notepad." will embellish it. When
    # set, this fixed sentence is the reply, with no second model call.
    confirmation: str = ""

    def matches(self, text: str) -> bool:
        return bool(self.triggers) and re.search(self.triggers, text, re.IGNORECASE) is not None

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolBox:
    """The set of tools available in a conversation."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def relevant(self, text: str) -> list[Tool]:
        """Tools worth offering for this message (often none)."""
        return [tool for tool in self._tools.values() if tool.matches(text)]

    def confirmation(self, name: str) -> str:
        tool = self._tools.get(name)
        return tool.confirmation if tool else ""

    def call(self, call: ToolCall) -> str:
        """Run a tool and describe the outcome in words the model can relay."""
        tool = self._tools.get(call.name)
        if tool is None:
            return f"Error: there is no tool called {call.name}."
        allowed = tool.parameters.get("properties", {})
        arguments = {key: value for key, value in call.arguments.items() if key in allowed}
        try:
            return str(tool.run(**arguments))
        except Exception as exc:  # a failing tool must not end the conversation
            return f"Error: {exc}"
