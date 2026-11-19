import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import PurePath
from xml.etree import ElementTree

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_XML_BYTES = 24 * 1024 * 1024
MAX_PDF_PAGES = 300
MIN_PDF_CHARACTERS = 20
BINARY_SNIFF_BYTES = 4096
TEXT_EXTENSIONS = (".txt", ".md", ".markdown")
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS + (".docx", ".odt", ".pdf")
ZIP_MAGIC = b"PK\x03\x04"
PDF_MAGIC = b"%PDF-"
WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TEXT_NS = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}"
FORBIDDEN_XML = (b"<!DOCTYPE", b"<!ENTITY")
BLANK_LINES = re.compile(r"\n{3,}")


class ExtractError(ValueError):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Extracted:
    text: str
    characters: int
    truncated: bool


def _extension(filename: str) -> str:
    return PurePath(filename or "").suffix.lower()


def _tidy(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    return BLANK_LINES.sub("\n\n", text).strip()


def _decode_plain(data: bytes) -> str:
    if b"\x00" in data[:BINARY_SNIFF_BYTES]:
        raise ExtractError("that file looks like binary data, not text")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _read_xml_member(archive: zipfile.ZipFile, member: str) -> ElementTree.Element:
    try:
        info = archive.getinfo(member)
    except KeyError:
        raise ExtractError(f"that file has no {member}; it is not a valid document") from None
    if info.file_size > MAX_XML_BYTES:
        raise ExtractError(f"{member} unpacks to more than {MAX_XML_BYTES // (1024 * 1024)} MB, which is not accepted")
    with archive.open(info) as stream:
        raw = stream.read(MAX_XML_BYTES + 1)
    if len(raw) > MAX_XML_BYTES:
        raise ExtractError(f"{member} unpacks to more than {MAX_XML_BYTES // (1024 * 1024)} MB, which is not accepted")
    if any(marker in raw.upper() for marker in FORBIDDEN_XML):
        raise ExtractError("that document declares XML entities, which are not accepted")
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as error:
        raise ExtractError(f"{member} is not valid XML: {error}") from error


def _open_archive(data: bytes) -> zipfile.ZipFile:
    if not data.startswith(ZIP_MAGIC):
        raise ExtractError("that file is not a valid document (no zip header)")
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as error:
        raise ExtractError(f"that file is a damaged zip: {error}") from error


def _docx_paragraph(paragraph: ElementTree.Element) -> str:
    pieces = []
    for node in paragraph.iter():
        if node.tag == WORD_NS + "t" and node.text:
            pieces.append(node.text)
        elif node.tag == WORD_NS + "tab":
            pieces.append("\t")
        elif node.tag in (WORD_NS + "br", WORD_NS + "cr"):
            pieces.append("\n")
    return "".join(pieces)


def _extract_docx(data: bytes) -> str:
    with _open_archive(data) as archive:
        root = _read_xml_member(archive, "word/document.xml")
    paragraphs = [_docx_paragraph(node) for node in root.iter(WORD_NS + "p")]
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())


def _odt_text(node: ElementTree.Element) -> str:
    pieces = [node.text or ""]
    for child in node:
        if child.tag == TEXT_NS + "s":
            pieces.append(" " * int(child.get(TEXT_NS + "c", "1") or 1))
        elif child.tag == TEXT_NS + "tab":
            pieces.append("\t")
        elif child.tag == TEXT_NS + "line-break":
            pieces.append("\n")
        else:
            pieces.append(_odt_text(child))
        pieces.append(child.tail or "")
    return "".join(pieces)


def _extract_odt(data: bytes) -> str:
    with _open_archive(data) as archive:
        root = _read_xml_member(archive, "content.xml")
    blocks = [_odt_text(node) for node in root.iter() if node.tag in (TEXT_NS + "p", TEXT_NS + "h")]
    return "\n\n".join(block for block in blocks if block.strip())


def _extract_pdf(data: bytes) -> str:
    if not data.startswith(PDF_MAGIC):
        raise ExtractError("that file is not a PDF (no %PDF header)")
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractError("that PDF is password protected")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ExtractError(f"that PDF has {len(reader.pages)} pages; the limit is {MAX_PDF_PAGES}")
        pages = [page.extract_text() or "" for page in reader.pages]
    except ExtractError:
        raise
    except (PyPdfError, ValueError, KeyError, OSError, RecursionError) as error:
        raise ExtractError(f"that PDF could not be read: {error}") from error
    text = "\n\n".join(pages)
    if len(re.sub(r"\s", "", text)) < MIN_PDF_CHARACTERS:
        raise ExtractError("that PDF has no readable text (a scan without a text layer?)")
    return text


def extract_text(filename: str, data: bytes, max_characters: int) -> Extracted:
    extension = _extension(filename)
    if extension not in SUPPORTED_EXTENSIONS:
        raise ExtractError(f"{extension or 'that file type'} is not supported; use one of {', '.join(SUPPORTED_EXTENSIONS)}", 415)
    if not data:
        raise ExtractError("that file is empty", 400)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtractError(f"that file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)
    if extension in TEXT_EXTENSIONS:
        raw = _decode_plain(data)
    elif extension == ".docx":
        raw = _extract_docx(data)
    elif extension == ".odt":
        raw = _extract_odt(data)
    else:
        raw = _extract_pdf(data)
    text = _tidy(raw)
    if not text:
        raise ExtractError("no text was found in that file")
    total = len(text)
    if total > max_characters:
        return Extracted(text[:max_characters], total, True)
    return Extracted(text, total, False)
