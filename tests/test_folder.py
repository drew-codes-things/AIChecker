import os

import pytest

from aichecker.web.folder import MAX_FOLDER_DEPTH, list_voice_files, rescan_folder
from aichecker.web.voice import VoiceError, VoiceLibrary
from support import make_docx

PARAGRAPH = "I like to write plainly about ordinary days, and I never use a long word where a short one will do. " * 3


def fake_embed(texts):
    return [[float(len(text) % 5 + 1), 1.0, float(text.count("w") + 1)] for text in texts]


@pytest.fixture
def library(tmp_path):
    return VoiceLibrary(tmp_path / "data" / "voice.db", clock=lambda: 1.0)


def test_only_supported_files_inside_the_folder_are_listed(tmp_path):
    folder = tmp_path / "writing"
    (folder / "sub").mkdir(parents=True)
    (folder / "a.txt").write_text(PARAGRAPH)
    (folder / "b.png").write_bytes(b"x")
    (folder / "sub" / "c.md").write_text(PARAGRAPH)
    names = sorted(path.name for path in list_voice_files(folder))
    assert names == ["a.txt", "c.md"]


def test_symlinks_that_leave_the_folder_are_not_followed(tmp_path):
    folder = tmp_path / "writing"
    folder.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text(PARAGRAPH)
    os.symlink(outside, folder / "link.txt")
    os.symlink(tmp_path, folder / "linkdir")
    assert list_voice_files(folder) == []


def test_files_deeper_than_the_depth_limit_are_ignored(tmp_path):
    folder = tmp_path / "writing"
    deep = folder
    for level in range(MAX_FOLDER_DEPTH + 1):
        deep = deep / f"d{level}"
    deep.mkdir(parents=True)
    (deep / "deep.txt").write_text(PARAGRAPH)
    (folder / "top.txt").write_text(PARAGRAPH)
    assert [path.name for path in list_voice_files(folder)] == ["top.txt"]


def test_a_missing_folder_is_a_404_style_error(tmp_path):
    with pytest.raises(VoiceError) as error:
        list_voice_files(tmp_path / "nope")
    assert error.value.status == 404


def test_rescan_adds_new_files_skips_unchanged_and_removes_deleted(tmp_path, library):
    folder = tmp_path / "writing"
    folder.mkdir()
    (folder / "one.txt").write_text(PARAGRAPH)
    (folder / "two.docx").write_bytes(make_docx([PARAGRAPH, PARAGRAPH]))
    first = rescan_folder(library, folder, fake_embed, 300)
    assert first["added"] == 2 and first["unchanged"] == 0 and first["skipped"] == []
    again = rescan_folder(library, folder, fake_embed, 300)
    assert again["added"] == 0 and again["unchanged"] == 2
    (folder / "one.txt").unlink()
    changed = rescan_folder(library, folder, fake_embed, 300)
    assert changed["removed"] == 1 and library.totals()["sources"] == 1


def test_a_changed_file_replaces_its_old_passages(tmp_path, library):
    folder = tmp_path / "writing"
    folder.mkdir()
    target = folder / "one.txt"
    target.write_text(PARAGRAPH)
    rescan_folder(library, folder, fake_embed, 300)
    target.write_text(PARAGRAPH + "\n\nA second paragraph added later on, written in the same plain way as the first one.")
    stat = target.stat()
    os.utime(target, (stat.st_atime, stat.st_mtime + 10))
    summary = rescan_folder(library, folder, fake_embed, 300)
    assert summary["added"] == 1 and library.totals()["sources"] == 1


def test_unreadable_files_are_reported_not_fatal(tmp_path, library):
    folder = tmp_path / "writing"
    folder.mkdir()
    (folder / "good.txt").write_text(PARAGRAPH)
    (folder / "bad.pdf").write_bytes(b"not a pdf at all")
    summary = rescan_folder(library, folder, fake_embed, 300)
    assert summary["added"] == 1
    assert summary["skipped"][0]["name"] == "bad.pdf" and "PDF" in summary["skipped"][0]["reason"]
