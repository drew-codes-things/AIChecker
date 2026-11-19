import pytest

from aichecker.web.humanise import (MAX_HUMANISE_CHARACTERS, SEGMENT_LIMIT_CHARS, Cancelled, humanise, join_segments,
                                    plan_segments)
from aichecker.web.upstream import UpstreamError
from aichecker.web.voice import VoiceError, VoiceLibrary, index_text

VOICE = ("I keep my notes short. I don't like fuss, and I say what I mean. We went to the shop on Friday, and I bought bread. "
         "It's fine. I'm not going to dress it up. Honestly, plain words work best for me, and I've always thought so. ")
VOICE = "\n\n".join([VOICE] * 8)


class FakeWorker:
    def __init__(self, fail_score=False, kept=False):
        self.calls = []
        self.fail_score = fail_score
        self.kept = kept

    def embed(self, texts):
        self.calls.append(("embed", len(texts)))
        return [[float(len(text) % 5 + 1), 1.0, 2.0] for text in texts]

    def rewrite(self, text, samples, profile):
        self.calls.append(("rewrite", len(samples), profile is not None))
        return {"text": text.upper(), "warnings": ["check 5"] if text.startswith("Two") else [], "kept_original": self.kept}

    def score(self, text):
        self.calls.append(("score", text))
        if self.fail_score:
            raise UpstreamError("the scoring worker is busy", 503, True)
        return {"assessment": {"band": "little_sign"}}


@pytest.fixture
def library(tmp_path):
    library = VoiceLibrary(tmp_path / "voice.db")
    index_text(library, "me.txt", VOICE, lambda texts: [[float(len(t) % 5 + 1), 1.0, 2.0] for t in texts], 300)
    return library


def run(text, library, worker, cancelled=lambda: False):
    steps = []
    result = humanise(text, library, worker, lambda step, total, message: steps.append((step, total, message)), cancelled)
    return result, steps


def test_paragraphs_become_segments_joined_back_exactly():
    text = "One paragraph here.\n\nTwo paragraphs there.\n\nThree."
    segments = plan_segments(text)
    assert [segment.text for segment in segments] == ["One paragraph here.", "Two paragraphs there.", "Three."]
    assert join_segments(segments, [segment.text for segment in segments]) == text


def test_a_huge_paragraph_is_split_into_pieces_under_the_limit_and_rejoined_with_spaces():
    paragraph = " ".join(f"This is sentence {n} of a very long paragraph." for n in range(120))
    segments = plan_segments(paragraph)
    assert len(segments) > 1 and all(len(segment.text) <= SEGMENT_LIMIT_CHARS for segment in segments)
    assert join_segments(segments, [segment.text for segment in segments]) == paragraph


def test_the_pipeline_rewrites_each_part_with_samples_then_scores_before_and_after(library):
    worker = FakeWorker()
    result, steps = run("One paragraph here.\n\nTwo paragraphs there.", library, worker)
    assert result["rewritten"] == "ONE PARAGRAPH HERE.\n\nTWO PARAGRAPHS THERE."
    assert result["original"] == "One paragraph here.\n\nTwo paragraphs there."
    assert [call[0] for call in worker.calls] == ["embed", "rewrite", "rewrite", "score", "score"]
    assert worker.calls[1][1] >= 1 and worker.calls[1][2] is True
    assert result["before"] and result["after"] and result["score_error"] is None
    assert result["warnings"] == ["Part 2: check 5"]
    assert steps[0][2].startswith("Finding") and steps[-1] == (5, 5, "Done")


def test_scoring_trouble_does_not_lose_the_rewrite(library):
    result, _ = run("One paragraph here.", library, FakeWorker(fail_score=True))
    assert result["rewritten"] == "ONE PARAGRAPH HERE."
    assert result["before"] is None and result["after"] is None and "busy" in result["score_error"]


def test_kept_original_parts_are_counted(library):
    result, _ = run("One paragraph here.\n\nTwo.", library, FakeWorker(kept=True))
    assert result["kept_original_parts"] == 2


def test_without_enough_of_the_users_own_writing_nothing_is_rewritten(tmp_path):
    empty = VoiceLibrary(tmp_path / "empty.db")
    worker = FakeWorker()
    with pytest.raises(VoiceError, match="add some of your own writing") as error:
        run("Some text to rewrite.", empty, worker)
    assert error.value.status == 409 and worker.calls == []


def test_oversized_and_empty_input_are_refused(library):
    with pytest.raises(VoiceError) as oversized:
        run("x" * (MAX_HUMANISE_CHARACTERS + 1), library, FakeWorker())
    assert oversized.value.status == 413
    with pytest.raises(VoiceError, match="no text"):
        run("  \n\n  ", library, FakeWorker())


def test_cancelling_between_parts_stops_before_the_next_rewrite(library):
    worker = FakeWorker()
    calls = {"count": 0}

    def cancelled():
        calls["count"] += 1
        return calls["count"] > 1

    with pytest.raises(Cancelled):
        run("One.\n\nTwo.\n\nThree.", library, worker, cancelled)
    assert [call[0] for call in worker.calls].count("rewrite") == 1
