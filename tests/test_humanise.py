import pytest

from aichecker.worker.humanise import (MAX_SEGMENT_CHARS, Rewriter, build_messages, clean_reply, length_is_reasonable,
                                       meaning_warnings)
from fakes import FakeLink

ORIGINAL = ('In 2024 the team of 12 shipped version 3.1.4 for Acme Corp. "We are very proud," said Maria Lopez. '
            "The release cost $4,500 and took six weeks, which was longer than planned.")
SAMPLES = ["I write plain notes. They are short.", "Second sample, a bit longer than the first one."]


def test_the_prompt_carries_samples_notes_and_the_passage_and_keeps_the_meaning_rule():
    messages = build_messages(ORIGINAL, SAMPLES, {"words": 100, "sentences": 10, "average_sentence_words": 10.0,
                                                  "short_sentence_share": 0.4, "long_sentence_share": 0.1,
                                                  "contractions_per_100_words": 3.0, "commas_per_100_words": 5.0,
                                                  "semicolons_per_100_words": 0.0, "dashes_per_100_words": 0.0,
                                                  "questions_per_100_sentences": 10.0, "exclamations_per_100_sentences": 0.0,
                                                  "first_person_per_100_words": 2.0, "average_word_letters": 4.5,
                                                  "average_paragraph_words": 60})
    system, user = messages[0]["content"], messages[1]["content"]
    assert "Keep the meaning exactly" in system
    assert "--- sample 1 ---" in user and "--- sample 2 ---" in user
    assert "Style notes" in user and "Contractions appear" in user
    assert user.rstrip().endswith(">>>") and ORIGINAL in user


def test_the_prompt_never_mentions_detectors_or_scores():
    messages = build_messages(ORIGINAL, SAMPLES, None)
    everything = (messages[0]["content"] + messages[1]["content"]).lower()
    for forbidden in ("detector", "detection", "gptzero", "turnitin", "undetect", "score", "bypass", "evade", "human-like", "ai-written"):
        assert forbidden not in everything


@pytest.mark.parametrize("raw,expected", [
    ("<think>planning</think>\nThe text.", "The text."),
    ("```\nThe text.\n```", "The text."),
    ("Here is the rewritten text:\nThe text.", "The text."),
    ('"The text."', "The text."),
    ('"One" and "two"', '"One" and "two"'),
    ("  The text.  ", "The text."),
])
def test_clean_reply_removes_wrappers_only(raw, expected):
    assert clean_reply(raw) == expected


def test_length_guard_rejects_empty_and_runaway_rewrites():
    assert length_is_reasonable("a" * 100, "b" * 100)
    assert not length_is_reasonable("a" * 100, "")
    assert not length_is_reasonable("a" * 100, "b" * 30)
    assert not length_is_reasonable("a" * 100, "b" * 300)


def test_meaning_warnings_flag_lost_numbers_names_and_quotes():
    dropped = "The team shipped the release for the customer and said they were proud, taking longer than planned."
    warnings = " ".join(meaning_warnings(ORIGINAL, dropped))
    assert "2024" in warnings and "4,500" in warnings
    assert "Acme" in warnings or "Maria" in warnings or "Lopez" in warnings
    assert "quoted" in warnings


def test_meaning_warnings_are_empty_when_everything_survives():
    kept = ('Version 3.1.4 shipped in 2024 from a team of 12 at Acme Corp. "We are very proud," Maria Lopez said. '
            "It cost $4,500 and took six weeks, longer than planned.")
    assert meaning_warnings(ORIGINAL, kept) == []


def test_rewrite_returns_the_cleaned_reply_with_warnings():
    link = FakeLink()
    link.reply = lambda messages: "<think>x</think>" + "Version 3.1.4 shipped in 2024. " * 3 + "Acme Corp"
    result = Rewriter(link, "qwen3:14b").rewrite("Version 3.1.4 shipped in 2024 at Acme Corp today and it went well for all.", SAMPLES, None)
    assert not result.kept_original and "<think>" not in result.text
    assert [call[0] for call in link.calls] == ["chat"]


def test_a_bad_first_reply_is_retried_once_then_the_original_is_kept():
    link = FakeLink()
    link.reply = lambda messages: ""
    result = Rewriter(link, "qwen3:14b").rewrite("Some perfectly ordinary sentence for the test.", SAMPLES, None)
    assert result.kept_original and result.text == "Some perfectly ordinary sentence for the test."
    assert [call[0] for call in link.calls] == ["chat", "chat"]
    assert "original was kept" in result.warnings[0]


def test_a_bad_first_reply_followed_by_a_good_one_is_accepted():
    replies = iter(["", "A perfectly fine rewritten sentence for the test."])
    link = FakeLink()
    link.reply = lambda messages: next(replies)
    result = Rewriter(link, "m").rewrite("Some perfectly ordinary sentence for the test.", SAMPLES, None)
    assert not result.kept_original and result.text.startswith("A perfectly fine")


def test_rewrite_needs_samples_and_refuses_oversized_segments():
    rewriter = Rewriter(FakeLink(), "m")
    with pytest.raises(ValueError, match="writing sample"):
        rewriter.rewrite("text", [], None)
    with pytest.raises(ValueError, match=str(MAX_SEGMENT_CHARS)):
        rewriter.rewrite("x" * (MAX_SEGMENT_CHARS + 1), SAMPLES, None)
