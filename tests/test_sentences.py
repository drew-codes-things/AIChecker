import pytest

from aichecker.sentences import Span, split_sentences


def pieces(text):
    return [span.slice_of(text) for span in split_sentences(text)]


def test_plain_sentences_split_on_terminal_punctuation():
    assert pieces("One thing. Two things! Three things? Four.") == ["One thing.", "Two things!", "Three things?", "Four."]


def test_abbreviations_and_initials_do_not_split():
    text = "Dr. Smith met J. K. Rowling in Washington D.C. on Jan. 5. They talked."
    assert pieces(text) == ["Dr. Smith met J. K. Rowling in Washington D.C. on Jan. 5.", "They talked."]


def test_decimals_and_versions_stay_inside_a_sentence():
    assert pieces("It cost $3.50 and ran version 2.1.4 fine. Done.") == ["It cost $3.50 and ran version 2.1.4 fine.", "Done."]


def test_lowercase_after_full_stop_is_not_a_boundary():
    assert pieces("We use e.g. apples. Then pears.") == ["We use e.g. apples.", "Then pears."]


def test_ellipsis_followed_by_capital_splits_and_by_lowercase_does_not():
    assert pieces("Wait... Then he left. He said so... and left.") == ["Wait...", "Then he left.", "He said so... and left."]


def test_closing_quotes_stay_with_their_sentence():
    assert pieces('"Yes," he said. "No," she said.') == ['"Yes," he said.', '"No," she said.']


def test_single_newline_is_a_soft_wrap_but_blank_line_splits():
    assert pieces("The line was\nwrapped here. Next one.\n\nNew paragraph without stop") == [
        "The line was\nwrapped here.", "Next one.", "New paragraph without stop"]


def test_offsets_point_into_the_original_text():
    text = "  First one.   Second one.  "
    spans = split_sentences(text)
    assert spans == [Span(2, 12), Span(15, 26)]
    assert [span.slice_of(text) for span in spans] == ["First one.", "Second one."]


@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\t \n"])
def test_blank_text_has_no_sentences(text):
    assert split_sentences(text) == []


def test_text_without_terminal_punctuation_is_one_sentence():
    assert pieces("no stop at all") == ["no stop at all"]


def test_unicode_text_keeps_offsets_in_characters():
    text = "Café au lait \u2014 nice. Next über line."
    assert [span.slice_of(text) for span in split_sentences(text)] == ["Café au lait \u2014 nice.", "Next über line."]
