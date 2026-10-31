"""Splits a streamed LLM reply into TTS units at clause boundaries so speech starts early."""
from __future__ import annotations

import re

from ai_phone.pipeline.prompts import END_MARKER

_SENT = set(".!?")
_CLAUSE = set(",;:")
# the model may write [end] or [End]; the match takes the spaces before the marker and a punctuation mark after it
_MARKER = re.compile(r"\s*" + re.escape(END_MARKER) + r"(\s*[.,!?;:])?", re.IGNORECASE)
_TRAILING_MARKER = re.compile(re.escape(END_MARKER) + r"\s*$", re.IGNORECASE)


def remove_end_marker(text: str) -> tuple[str, int]:
    """The text as spoken, and how many end markers it held. No space is left before punctuation, and a mark written
    after the marker stays only when the words before it end without one: "Thanks, bye [END]." gives "Thanks, bye.",
    "Goodbye. [END]." gives "Goodbye."."""
    def keep(m: re.Match) -> str:
        before = text[:m.start()].rstrip()
        return "" if before[-1:] in _SENT | _CLAUSE else (m.group(1) or "").strip()
    return _MARKER.subn(keep, text)


def _partial_marker_len(text: str) -> int:
    """Length of the start of the marker that ends `text` ("[", "[E", "[EN", "[END", any case), or 0."""
    low, marker = text.lower(), END_MARKER.lower()
    return next((i for i in range(len(marker) - 1, 0, -1) if low.endswith(marker[:i])), 0)


# A period inside a word is never a boundary. These common English forms also
# keep their following word when there is whitespace after the period. This is
# deliberately conservative: merging two sentences is preferable to making TTS
# pronounce a title, initial or abbreviation as a sentence of its own.
_ABBREVIATIONS = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "no", "fig", "dept", "etc"}
_INITIALS = re.compile(r"(?:[a-z]\.)+", re.IGNORECASE)
_CLOSERS = "\"'’”)]}"
_SPACE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return sum(any(ch.isalnum() for ch in word) for word in text.split())


def _sentence_end(text: str) -> bool:
    tail = text.rstrip(_CLOSERS)
    if tail[-1:] not in _SENT:
        return False
    if not tail.endswith("."):
        return True
    word = tail.split()[-1].lstrip("\"'‘“([{")
    if word[:-1].casefold() in _ABBREVIATIONS or _INITIALS.fullmatch(word):
        return False
    # Do not give TTS a list number as a separate sentence ("1. First ...").
    return not (word[:-1].isdigit() and _word_count(tail) == 1)


class ClauseChunker:
    """Incremental, word-safe TTS units independent of LLM token boundaries.

    Prefer full sentences, or clause punctuation after four words initially and
    six thereafter. Unpunctuated text falls back at 16/24 *complete words*, not
    eight/twelve arbitrary token fragments. A word without trailing whitespace
    stays pending until more text or flush(), so decimals, abbreviations and
    closing quotes can be resolved before synthesis starts.

    These bounds limit words per fallback unit, not elapsed time. Waiting for a
    meaningful phrase trades some first-audio latency for connected-speech
    prosody.
    """

    def __init__(self, min_words_for_clause: int = 6, max_words: int = 24,
                 first_min_words_for_clause: int = 4, first_max_words: int = 16):
        if min(min_words_for_clause, max_words, first_min_words_for_clause, first_max_words) < 1:
            raise ValueError("chunk sizes must be positive word counts")
        self.min_clause, self.max_words = min_words_for_clause, max_words
        self.first_min_clause, self.first_max_words = first_min_words_for_clause, first_max_words
        self._text = ""
        self._emitted = 0
        self.ended = False  # a complete END_MARKER was seen; never speak it

    def feed(self, token: str) -> list[str]:
        self._text += token
        if _partial_marker_len(self._text.rstrip()):
            return []
        cleaned, found = remove_end_marker(self._text)
        self.ended |= bool(found)
        if found and _TRAILING_MARKER.search(self._text):
            # Preserve the marker until its optional following punctuation is
            # known: "Goodbye. [END]" + "." must not become "Goodbye..".
            return []
        self._text = cleaned
        return self._drain()

    def _drain(self, *, final: bool = False) -> list[str]:
        units = []
        while self._text:
            min_clause = self.first_min_clause if self._emitted == 0 else self.min_clause
            max_words = self.first_max_words if self._emitted == 0 else self.max_words
            for space in _SPACE.finditer(self._text):
                # Look past whitespace so a marker in the next delta can still
                # attach its punctuation to the preceding words.
                if not final and space.end() == len(self._text):
                    return units
                text = self._text[:space.start()].strip()
                words = _word_count(text)
                if not words:
                    continue
                # Marker fragments are control text, not words or punctuation.
                # Keep them until completed, or discard them on final flush.
                if _partial_marker_len(text):
                    return units
                tail = text.rstrip(_CLOSERS)
                if ("\n" in space.group() or _sentence_end(text)
                        or (tail[-1:] in _CLAUSE and words >= min_clause)
                        or words >= max_words):
                    units.append(text)
                    self._emitted += 1
                    self._text = self._text[space.end():]
                    break
            else:
                break
        return units

    def flush(self) -> list[str]:
        self._text, found = remove_end_marker(self._text)
        self.ended |= bool(found)
        # A reply cut off inside a marker is not a request to end the call.
        self._text = self._text.rstrip()
        self._text = self._text[:len(self._text) - _partial_marker_len(self._text)]
        units = self._drain(final=True)
        text, self._text = self._text.strip(), ""
        if any(ch.isalnum() for ch in text):
            units.append(text)
            self._emitted += 1
        return units
