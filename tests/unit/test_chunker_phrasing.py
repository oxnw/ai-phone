"""Phrase boundaries must depend on text, never a backend's token-event sizes."""
import re

import pytest

from ai_phone.pipeline.chunker import ClauseChunker


def chunks(text, pieces, **kwargs):
    chunker = ClauseChunker(**kwargs)
    result = []
    for piece in pieces(text):
        result.extend(chunker.feed(piece))
    return result + chunker.flush()


FRAGMENTATIONS = [
    pytest.param(lambda text: [text], id="whole-delta"),
    pytest.param(list, id="characters"),
    pytest.param(lambda text: re.findall(r"\S+\s*", text), id="words"),
    pytest.param(lambda text: [text[i:i + 3] for i in range(0, len(text), 3)], id="subwords"),
]


@pytest.mark.parametrize("pieces", FRAGMENTATIONS)
@pytest.mark.parametrize("text,expected", [
    ("I can help you set up a new number today.", ["I can help you set up a new number today."]),
    ("I can help you set up the telephone.", ["I can help you set up the telephone."]),
    ("Please ask Dr. Lee about version 3.14 today.", ["Please ask Dr. Lee about version 3.14 today."]),
    ("I'll call at 10:30 a.m. tomorrow. Please stay nearby.",
     ["I'll call at 10:30 a.m. tomorrow.", "Please stay nearby."]),
    ("Talk to J. Smith in the U.S. office.", ["Talk to J. Smith in the U.S. office."]),
    ("Go to example.com or email a.b@example.com. I'll wait.",
     ["Go to example.com or email a.b@example.com.", "I'll wait."]),
    ('She said "Please wait." Then she left.', ['She said "Please wait."', "Then she left."]),
    ("Hello. How are you? I'm fine!", ["Hello.", "How are you?", "I'm fine!"]),
    ("Thanks, bye [END].", ["Thanks, bye."]),
])
def test_natural_units_do_not_depend_on_token_fragmentation(text, expected, pieces):
    assert chunks(text, pieces) == expected


def test_eight_subword_events_do_not_force_a_phrase_break():
    chunker = ClauseChunker()
    for piece in ["I", " can", " help", " you", " set", " up", " a", " new"]:
        assert chunker.feed(piece) == []
    assert chunker.feed(" number today.") == []
    assert chunker.flush() == ["I can help you set up a new number today."]


@pytest.mark.parametrize("pieces", FRAGMENTATIONS)
def test_meaningful_clause_punctuation_starts_speech_before_reply_finishes(pieces):
    text = "If you give me a moment, I can check the details for you; then we can decide."
    assert chunks(text, pieces) == [
        "If you give me a moment,", "I can check the details for you;", "then we can decide.",
    ]


def test_confirmed_clause_can_be_emitted_while_generation_continues():
    chunker = ClauseChunker()
    assert chunker.feed("If you give me a moment,") == []
    assert chunker.feed(" I") == ["If you give me a moment,"]
    assert chunker.flush() == ["I"]


@pytest.mark.parametrize("pieces", FRAGMENTATIONS)
def test_long_unpunctuated_text_has_word_bounded_units_and_loses_no_words(pieces):
    words = [f"word{i}" for i in range(50)]
    result = chunks(" ".join(words), pieces)
    assert [len(unit.split()) for unit in result] == [16, 24, 10]
    assert " ".join(result).split() == words


@pytest.mark.parametrize("pieces", FRAGMENTATIONS)
def test_fallback_does_not_split_a_word_in_a_later_token(pieces):
    text = " ".join([f"word{i}" for i in range(16)]) + " telephone service is ready"
    result = chunks(text, pieces)
    assert result[1] == "telephone service is ready"
    assert " ".join(result) == text


def test_even_a_long_single_word_is_not_cut_in_half():
    word = "antidisestablishmentarianism"
    chunker = ClauseChunker(max_words=1, first_max_words=1)
    for char in word:
        assert chunker.feed(char) == []
    assert chunker.feed(" ") == []
    assert chunker.feed("a") == [word]
    assert chunker.flush() == ["a"]


@pytest.mark.parametrize("pieces", FRAGMENTATIONS)
def test_paragraph_boundary_and_flush_preserve_remaining_text(pieces):
    assert chunks("First point\nSecond point without punctuation", pieces) == [
        "First point", "Second point without punctuation",
    ]


def test_sentence_number_is_not_spoken_as_its_own_unit():
    assert chunks("1. Please stay on the line.", list) == ["1. Please stay on the line."]


@pytest.mark.parametrize("name", ["min_words_for_clause", "max_words", "first_min_words_for_clause", "first_max_words"])
def test_word_bounds_must_be_positive(name):
    with pytest.raises(ValueError, match="positive word counts"):
        ClauseChunker(**{name: 0})


@pytest.mark.parametrize("partial", ["[", "[E", "[EN", "[END"])
@pytest.mark.parametrize("space", [" ", "\n", "\t"])
def test_flush_drops_incomplete_control_marker_before_trailing_whitespace(partial, space):
    chunker = ClauseChunker()
    units = chunker.feed("I can help " + partial + space) + chunker.flush()
    assert units == ["I can help"]
    assert chunker.ended is False
