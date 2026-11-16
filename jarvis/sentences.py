"""Cut a stream of text pieces into sentences, so speech can start before the reply is finished."""

from __future__ import annotations

import re

# A sentence ends at . ! or ? (optionally followed by a closing quote or
# bracket) and then whitespace. Requiring the whitespace keeps "3.5" and
# "example.com" in one piece, and means we never cut before the next word
# has started arriving.
_SENTENCE_END = re.compile(r"""[.!?]+["')\]]*\s+""")

# Short words that end in a full stop without ending the sentence.
_ABBREVIATIONS = frozenset({"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"})


class SentenceSplitter:
    """Feed it text as it arrives; it hands back each sentence once it is complete."""

    def __init__(self, min_chars: int = 12) -> None:
        # Very short fragments ("Sure.") sound choppy on their own, so they are
        # held back and spoken together with the sentence that follows.
        self._min_chars = min_chars
        self._buffer = ""

    def feed(self, piece: str) -> list[str]:
        """Add text and return any sentences that are now complete."""
        self._buffer += piece
        sentences: list[str] = []
        start = 0
        for match in _SENTENCE_END.finditer(self._buffer):
            candidate = self._buffer[start : match.end()].strip()
            if len(candidate) < self._min_chars or self._ends_with_abbreviation(candidate):
                continue
            sentences.append(candidate)
            start = match.end()
        self._buffer = self._buffer[start:]
        return sentences

    def flush(self) -> str:
        """Return whatever is left once the reply has ended ('' if nothing)."""
        rest, self._buffer = self._buffer.strip(), ""
        return rest

    @staticmethod
    def _ends_with_abbreviation(text: str) -> bool:
        words = text.rstrip(".!?\"')] ").split()
        return bool(words) and words[-1].lower() in _ABBREVIATIONS and text.rstrip().endswith(".")
