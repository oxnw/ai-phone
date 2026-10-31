import pytest

from ai_phone.pipeline.chunker import ClauseChunker


def run(toks, **kw):
    c = ClauseChunker(**kw)
    out = []
    for t in toks:
        out += c.feed(t)
    return out + c.flush()


def test_sentence_punctuation_always_emits():
    assert run(["Hi", "."]) == ["Hi."]


def test_clause_punctuation_needs_min_words():
    # first unit: clause break allowed from 4 words; later units need 6
    assert run(["Yes", ",", " I", " can", " do", " that", ",", " sure", "."]) == ["Yes, I can do that,", "sure."]
    assert run(["Well", ",", " yes", ",", " I", " can", " do", " that", "."]) == ["Well, yes, I can do that."]


def test_max_words_emits_without_punctuation():
    toks = [f" w{i}" for i in range(24)]
    out = run(toks, max_words=12, first_max_words=8)
    assert out[0].split() == [f"w{i}" for i in range(8)]           # first unit: 8 complete words
    assert out[1].split() == [f"w{i}" for i in range(8, 20)]       # later units: 12 complete words
    assert len(out) == 3


def test_flush_returns_remainder_and_empties():
    c = ClauseChunker()
    c.feed("Hello")
    c.feed(" there")
    assert c.flush() == ["Hello there"] and c.flush() == []


def test_whitespace_only_is_dropped():
    assert run([" ", "\n"]) == []


def test_end_marker_split_over_tokens_is_removed_and_flagged():
    c = ClauseChunker()
    out = []
    for t in ["Goodbye", ",", " thanks", ".", " [", "END", "]"]:
        out += c.feed(t)
    out += c.flush()
    assert out == ["Goodbye, thanks."] and c.ended


def test_end_marker_glued_to_the_sentence():
    c = ClauseChunker()
    out = c.feed("Bye now.[END]") + c.flush()
    assert out == ["Bye now."] and c.ended


def test_partial_marker_is_held_back_until_complete():
    c = ClauseChunker(max_words=2, first_max_words=2)
    assert c.feed("Okay") == []
    assert c.feed(" [EN") == []          # a partial marker must never enter synthesis
    assert c.feed("D]") == []
    assert c.ended and c.flush() == ["Okay"]


def test_no_marker_means_not_ended():
    c = ClauseChunker()
    assert c.feed("Hello.") == []  # wait for a boundary or end of stream
    assert not c.ended and c.flush() == ["Hello."]


def test_marker_alone_emits_nothing():
    c = ClauseChunker()
    assert c.feed("[END]") == [] and c.flush() == [] and c.ended


@pytest.mark.parametrize("tail", [" [", " [E", " [EN", " [END", " [en"])
def test_reply_cut_inside_a_marker_speaks_the_words_before_it_and_does_not_end(tail):
    c = ClauseChunker()
    out = []
    for t in ["I", " can", " help", tail]:
        out += c.feed(t)
    out += c.flush()
    assert out == ["I can help"] and not c.ended


def test_lowercase_marker_is_removed_and_flagged():
    c = ClauseChunker()
    out = c.feed("Bye.[end]") + c.flush()
    assert out == ["Bye."] and c.ended


def test_mixed_case_partial_marker_is_held_then_removed():
    c = ClauseChunker(max_words=1, first_max_words=1)
    assert c.feed("Bye [En") == []       # a partial marker must never enter synthesis
    assert c.feed("d]") == []
    assert c.ended and c.flush() == ["Bye"]


def feed_all(toks):
    c = ClauseChunker()
    out = []
    for t in toks:
        out += c.feed(t)
    return out + c.flush(), c


@pytest.mark.parametrize("toks", [["Goodbye", ".", " [END]", "."], ["Goodbye", ".", " [", "END", "]", "."],
                                  ["Goodbye. [END]."]])
def test_punctuation_after_the_marker_is_never_a_unit_of_its_own(toks):
    out, c = feed_all(toks)
    assert out == ["Goodbye."] and c.ended


@pytest.mark.parametrize("toks", [["Thanks", ",", " bye", " [END]", "."], ["Thanks, bye [END]."],
                                  ["Thanks", ",", " bye", " [", "END", "]", " ."]])
def test_no_space_is_left_before_punctuation_where_the_marker_was(toks):
    out, c = feed_all(toks)
    assert out == ["Thanks, bye."] and c.ended


def test_a_unit_without_letters_or_digits_is_never_emitted():
    assert run(["Okay", ".", " ..."]) == ["Okay."]
    assert run(["-", " !"]) == []
