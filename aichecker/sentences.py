import re
from dataclasses import dataclass

TERMINAL_RUN = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)")
PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
OPENERS = "\"'“‘(["
NON_TERMINAL_WORDS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "cf", "fig", "no",
    "inc", "ltd", "co", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
})
MIN_SENTENCE_CHARACTERS = 1


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def slice_of(self, text: str) -> str:
        return text[self.start:self.end]


def _word_before(text: str, position: int) -> str:
    match = re.search(r"([A-Za-z][A-Za-z.]*)$", text[max(0, position - 12):position])
    return match.group(1).lower().rstrip(".") if match else ""


def _is_initial(text: str, position: int) -> bool:
    match = re.search(r"(?:^|[\s(])([A-Z])$", text[max(0, position - 3):position])
    return match is not None


def _next_visible(text: str, position: int) -> str:
    match = re.compile(r"\S").search(text, position)
    return match.group(0) if match else ""


def _starts_new_sentence(character: str) -> bool:
    if not character:
        return True
    return character.isupper() or character.isdigit() or character in OPENERS


def _is_real_boundary(text: str, run_start: int, run_end: int) -> bool:
    following = _next_visible(text, run_end)
    is_plain_full_stop = text[run_start:run_end].rstrip("\"'\u201d\u2019)]") == "."
    if is_plain_full_stop:
        word = _word_before(text, run_start)
        if word in NON_TERMINAL_WORDS or _is_initial(text, run_start):
            return False
    return _starts_new_sentence(following)


def _trim(text: str, start: int, end: int) -> Span | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end - start < MIN_SENTENCE_CHARACTERS:
        return None
    return Span(start, end)


def _boundaries_inside(text: str, start: int, end: int) -> list[int]:
    cuts = []
    for match in TERMINAL_RUN.finditer(text, start, end):
        if _is_real_boundary(text, match.start(), match.end()):
            cuts.append(match.end())
    return cuts


def split_sentences(text: str) -> list[Span]:
    spans: list[Span] = []
    paragraph_start = 0
    paragraph_ends = [(match.start(), match.end()) for match in PARAGRAPH_BREAK.finditer(text)]
    paragraph_ends.append((len(text), len(text)))
    for paragraph_end, next_start in paragraph_ends:
        cursor = paragraph_start
        for cut in _boundaries_inside(text, paragraph_start, paragraph_end) + [paragraph_end]:
            span = _trim(text, cursor, cut)
            if span:
                spans.append(span)
            cursor = cut
        paragraph_start = next_start
    return spans
