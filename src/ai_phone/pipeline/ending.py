"""Validate a model's hangup request without treating a stray marker as permission to disconnect."""
from __future__ import annotations

import re

from ai_phone.pipeline.chunker import remove_end_marker
from ai_phone.pipeline.prompts import END_MARKER


_TERMINAL_MARKER = re.compile(re.escape(END_MARKER) + r"[\s.!…]*\Z", re.IGNORECASE)
_FAREWELL = re.compile(
    r"(?:good[ -]?bye|bye(?:[ -]bye)?(?: (?:now|for now))?|take care|"
    r"have a (?:good|great|nice) (?:day|evening|night|weekend))", re.IGNORECASE,
)
_COURTESY = re.compile(
    r"(?:ok(?:ay)?|all right|alright|well|"
    r"(?:thanks|thank you)(?: again| so much| for (?:your (?:time|help)|calling))?)", re.IGNORECASE,
)


def is_closing_reply(reply: str) -> bool:
    """Require one terminal marker and a complete spoken closing sentence, with no questions.

    This checks output structure, not whether the model correctly judged the task complete. Unknown
    or ambiguous closings keep the line open; the prompt asks for an explicit final "Goodbye."
    """
    spoken, markers = remove_end_marker(reply)
    if markers != 1 or not _TERMINAL_MARKER.search(reply) or "?" in reply or "？" in reply:
        return False
    spoken = spoken.rstrip(" \t\r\n.!…")
    if not spoken:
        return False
    closing = re.split(r"[.!;\n]+", spoken)[-1].strip()
    parts = re.split(r"\s*,\s*|\s+and\s+", closing, flags=re.IGNORECASE)
    # Match complete phrases: "Don't say goodbye" and "The password is goodbye" are not farewells.
    return (any(_FAREWELL.fullmatch(part) for part in parts)
            and all(_FAREWELL.fullmatch(part) or _COURTESY.fullmatch(part) for part in parts))
