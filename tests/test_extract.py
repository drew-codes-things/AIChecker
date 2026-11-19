import pytest

from aichecker.web.extract import MAX_UPLOAD_BYTES, ExtractError, extract_text
from support import make_docx, make_odt, make_pdf, make_zip

LIMIT = 10_000


def test_plain_text_and_markdown_are_read_as_utf8_with_bom_and_crlf_normalised():
    result = extract_text("note.txt", b"\xef\xbb\xbfHello\r\nworld\r\n\r\n\r\n\r\nNext", LIMIT)
    assert result.text == "Hello\nworld\n\nNext" and not result.truncated
    assert extract_text("notes.MD", "# Title\n\nBody \u2014 here".encode(), LIMIT).text.startswith("# Title")


def test_text_that_is_not_utf8_falls_back_to_windows_1252():
    legacy_bytes = b"caf\xe9 \x93quoted\x94"
    assert extract_text("old.txt", legacy_bytes, LIMIT).text == "caf\u00e9 \u201cquoted\u201d"


def test_binary_data_posing_as_text_is_refused():
    with pytest.raises(ExtractError, match="binary"):
        extract_text("a.txt", b"abc\x00\x01\x02def", LIMIT)


def test_docx_paragraphs_become_blank_line_separated_text():
    data = make_docx(["First paragraph.", "", "Second one &amp; more."])
    assert extract_text("a.docx", data, LIMIT).text == "First paragraph.\n\nSecond one & more."


def test_docx_tabs_and_line_breaks_survive():
    body = '<w:p><w:r><w:t>one</w:t><w:tab/><w:t>two</w:t><w:br/><w:t>three</w:t></w:r></w:p>'
    data = make_zip({"word/document.xml": '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + body + "</w:body></w:document>"})
    assert extract_text("a.docx", data, LIMIT).text == "one\ttwo\nthree"


def test_odt_paragraphs_headings_spans_and_spaces_are_read():
    members = {"content.xml": ('<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
                               'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
                               '<text:h>Title</text:h><text:p>Some <text:span>bold</text:span> text<text:s text:c="2"/>here'
                               '<text:line-break/>next</text:p></office:text></office:body></office:document-content>')}
    assert extract_text("a.odt", make_zip(members), LIMIT).text == "Title\n\nSome bold text  here\nnext"
    assert "Hello" in extract_text("b.odt", make_odt(["Hello there"]), LIMIT).text


def test_pdf_text_is_extracted():
    result = extract_text("a.pdf", make_pdf(["Hello from a PDF file", "second line here"]), LIMIT)
    assert "Hello from a PDF file" in result.text and "second line here" in result.text


def test_a_pdf_without_text_is_reported_as_unreadable_not_empty_success():
    with pytest.raises(ExtractError, match="no readable text"):
        extract_text("a.pdf", make_pdf([""]), LIMIT)


def test_garbage_named_pdf_docx_or_odt_is_refused_cleanly():
    for name in ("a.pdf", "a.docx", "a.odt"):
        with pytest.raises(ExtractError):
            extract_text(name, b"this is not that kind of file at all", LIMIT)


def test_a_zip_without_the_document_part_is_refused():
    with pytest.raises(ExtractError, match="word/document.xml"):
        extract_text("a.docx", make_zip({"other.txt": "x"}), LIMIT)


def test_xml_entity_tricks_are_refused_before_parsing():
    bomb = ('<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaaaaaaaa"><!ENTITY b "&a;&a;&a;&a;">]>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>&b;</w:t></w:r></w:p></w:body></w:document>')
    with pytest.raises(ExtractError, match="entities"):
        extract_text("a.docx", make_zip({"word/document.xml": bomb}), LIMIT)


def test_a_zip_that_unpacks_to_far_too_much_xml_is_refused():
    huge = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>' + "<w:p/>" * 5_000_000 + "</w:body></w:document>"
    data = make_zip({"word/document.xml": huge})
    assert len(data) < 1_000_000
    with pytest.raises(ExtractError, match="unpacks to more than"):
        extract_text("a.docx", data, LIMIT)


def test_unsupported_empty_and_oversized_uploads_are_refused_with_the_right_status():
    with pytest.raises(ExtractError) as unsupported:
        extract_text("photo.png", b"data", LIMIT)
    assert unsupported.value.status == 415
    with pytest.raises(ExtractError) as no_extension:
        extract_text("README", b"data", LIMIT)
    assert no_extension.value.status == 415
    with pytest.raises(ExtractError) as empty:
        extract_text("a.txt", b"", LIMIT)
    assert empty.value.status == 400
    with pytest.raises(ExtractError) as oversized:
        extract_text("a.txt", b"x" * (MAX_UPLOAD_BYTES + 1), LIMIT)
    assert oversized.value.status == 413


def test_whitespace_only_documents_have_no_text():
    with pytest.raises(ExtractError, match="no text"):
        extract_text("a.txt", b"  \n\n \t ", LIMIT)


def test_long_text_is_cut_at_the_limit_and_says_so():
    result = extract_text("a.txt", ("word " * 5000).encode(), 1000)
    assert result.truncated and len(result.text) == 1000 and result.characters > 1000


def test_a_path_in_the_file_name_does_not_matter():
    assert extract_text("../../etc/passwd.txt", b"hello there", LIMIT).text == "hello there"
