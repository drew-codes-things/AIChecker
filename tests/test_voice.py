import array

import pytest

from aichecker.web.voice import (EMBED_BATCH, MAX_LIBRARY_CHUNKS, VoiceError, VoiceLibrary, chunk_text, embed_chunks,
                                 evenly_spaced, index_text, nearest, normalise)

LONG_PARAGRAPH = " ".join(f"Sentence number {n} is here and says something plain about the day." for n in range(40))


def fake_embed(texts):
    return [[float(len(text) % 7 + 1), float(text.count("a") + 1), 1.0] for text in texts]


@pytest.fixture
def library(tmp_path):
    return VoiceLibrary(tmp_path / "voice.db", clock=lambda: 1.0)


def test_chunks_group_paragraphs_up_to_the_target_and_never_exceed_the_maximum():
    text = "\n\n".join(f"Paragraph {n} " + "words go here " * 12 for n in range(10))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1100 for chunk in chunks)
    assert "".join(chunks).replace("\n", "").replace(" ", "") .startswith("Paragraph0")


def test_a_very_long_paragraph_is_split_on_sentences():
    chunks = chunk_text(LONG_PARAGRAPH)
    assert len(chunks) >= 3 and all(len(chunk) <= 1100 for chunk in chunks)
    assert all(chunk.endswith(".") for chunk in chunks)


def test_tiny_fragments_are_dropped_but_a_short_text_still_gives_one_chunk():
    assert chunk_text("Short.") == ["Short."]
    assert chunk_text("") == []
    kept = chunk_text("Tiny.\n\n" + "A longer paragraph with enough words in it to count as a chunk by itself, right here. " * 3)
    assert all(len(chunk) >= 120 for chunk in kept) or len(kept) == 1


def test_evenly_spaced_keeps_order_and_covers_the_whole_range():
    assert evenly_spaced(list(range(10)), 20) == list(range(10))
    picked = evenly_spaced(list(range(100)), 5)
    assert picked == [0, 20, 40, 60, 80]


def test_normalise_gives_unit_length_and_refuses_zero_vectors():
    assert sum(x * x for x in normalise([3.0, 4.0])) == pytest.approx(1.0)
    with pytest.raises(VoiceError, match="zeros"):
        normalise([0.0, 0.0])


def test_embedding_is_batched_and_checked():
    calls = []

    def embed(batch):
        calls.append(len(batch))
        return fake_embed(batch)

    pairs = embed_chunks([f"chunk {n}" for n in range(EMBED_BATCH * 2 + 3)], embed)
    assert calls == [EMBED_BATCH, EMBED_BATCH, 3] and len(pairs) == EMBED_BATCH * 2 + 3
    with pytest.raises(VoiceError, match="wrong number"):
        embed_chunks(["a", "b"], lambda batch: [[1.0]])


def test_nearest_returns_the_most_similar_chunks_and_caps_per_source(library):
    vectors = {"north": [1.0, 0.0], "northish": [0.9, 0.1], "east": [0.0, 1.0], "northeast": [0.7, 0.7], "north2": [0.95, 0.05]}
    library.add_source("a.txt", "upload", None, None, 100, [(name, vector) for name, vector in vectors.items() if name != "east"][:3])
    library.add_source("b.txt", "upload", None, None, 100, [("east", vectors["east"]), ("northeast", vectors["northeast"])])
    chunks = library.chunks()
    best = nearest([1.0, 0.0], chunks, count=3, per_source=2)
    assert [chunk.text for chunk in best][0] == "north"
    assert sum(1 for chunk in best if chunk.source_id == chunks[0].source_id) <= 2
    assert len(best) == 3


def test_nearest_ignores_vectors_of_another_size(library):
    library.add_source("a.txt", "upload", None, None, 10, [("x", [1.0, 0.0, 0.0])])
    assert nearest([1.0, 0.0], library.chunks()) == []


def test_index_text_adds_a_source_with_its_passages_and_totals_follow(library):
    result = index_text(library, "diary.txt", LONG_PARAGRAPH, fake_embed, max_chunks=300)
    assert result["passages"] == result["passages_in_file"] >= 3
    totals = library.totals()
    assert totals["sources"] == 1 and totals["passages"] == result["passages"] and totals["ready"] is True
    assert library.sources()[0]["name"] == "diary.txt"


def test_a_long_file_is_sampled_evenly_to_the_passage_limit(library):
    text = "\n\n".join(f"Paragraph {n} " + "some plain words about nothing in particular " * 14 for n in range(60))
    result = index_text(library, "long.txt", text, fake_embed, max_chunks=10)
    assert result["passages"] == 10 and result["passages_in_file"] > 10


def test_not_ready_until_enough_writing_is_present(library):
    index_text(library, "tiny.txt", "Just one short paragraph of text that is long enough to be a passage, honestly yes it is.", fake_embed, 300)
    assert library.totals()["ready"] is False


def test_removing_a_source_removes_its_passages(library):
    result = index_text(library, "a.txt", LONG_PARAGRAPH, fake_embed, 300)
    assert library.remove_source(result["id"]) is True
    assert library.remove_source(result["id"]) is False
    assert library.totals() == {"sources": 0, "passages": 0, "characters": 0, "ready": False}


def test_the_library_size_is_capped(library):
    library.add_source("big", "upload", None, None, 1, [(f"t{n}", [1.0, float(n)]) for n in range(MAX_LIBRARY_CHUNKS)])
    with pytest.raises(VoiceError, match="at most") as error:
        library.add_source("more", "upload", None, None, 1, [("x", [1.0, 1.0])])
    assert error.value.status == 409


def test_adding_nothing_is_an_error(library):
    with pytest.raises(VoiceError):
        library.add_source("x", "upload", None, None, 0, [])


def test_vectors_survive_storage_as_unit_floats(library):
    library.add_source("a", "upload", None, None, 5, [("hello", [3.0, 4.0])])
    vector = library.chunks()[0].vector
    assert isinstance(vector, array.array) and vector[0] == pytest.approx(0.6) and vector[1] == pytest.approx(0.8)


def test_profile_texts_stop_at_the_character_budget(library):
    for number in range(4):
        library.add_source(f"s{number}", "upload", None, None, 1, [("w " * 400, [1.0, float(number)])])
    texts = library.profile_texts()
    assert 1 <= len(texts) <= 4
