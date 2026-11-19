from collections.abc import Callable
from dataclasses import dataclass

from ..sentences import split_sentences
from ..voiceprofile import build_profile
from .upstream import UpstreamError
from .voice import MIN_READY_CHARS, MIN_READY_CHUNKS, VoiceError, VoiceLibrary, nearest

SEGMENT_LIMIT_CHARS = 1800
MAX_HUMANISE_CHARACTERS = 12_000
PARAGRAPH_JOINER = "\n\n"
SENTENCE_JOINER = " "
EMBED_BATCH = 32


class Cancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class Segment:
    text: str
    after: str


def _split_paragraph(paragraph: str) -> list[str]:
    if len(paragraph) <= SEGMENT_LIMIT_CHARS:
        return [paragraph]
    pieces, current = [], ""
    for span in split_sentences(paragraph):
        sentence = span.slice_of(paragraph)
        if current and len(current) + 1 + len(sentence) > SEGMENT_LIMIT_CHARS:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [piece[:SEGMENT_LIMIT_CHARS] for piece in pieces]


def plan_segments(text: str) -> list[Segment]:
    paragraphs = [part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip()]
    segments: list[Segment] = []
    for paragraph_index, paragraph in enumerate(paragraphs):
        pieces = _split_paragraph(paragraph)
        last_paragraph = paragraph_index == len(paragraphs) - 1
        for piece_index, piece in enumerate(pieces):
            last_piece = piece_index == len(pieces) - 1
            after = "" if last_piece and last_paragraph else PARAGRAPH_JOINER if last_piece else SENTENCE_JOINER
            segments.append(Segment(piece, after))
    return segments


def join_segments(segments: list[Segment], texts: list[str]) -> str:
    return "".join(text + segment.after for segment, text in zip(segments, texts))


def humanise(text: str, library: VoiceLibrary, worker, progress: Callable[[int, int, str], None],
             cancelled: Callable[[], bool]) -> dict:
    if len(text) > MAX_HUMANISE_CHARACTERS:
        raise VoiceError(f"the Humaniser takes at most {MAX_HUMANISE_CHARACTERS} characters at a time (about 2,000 words)", 413)
    segments = plan_segments(text)
    if not segments:
        raise VoiceError("there is no text to rewrite")
    totals = library.totals()
    if not totals["ready"]:
        raise VoiceError(f"add some of your own writing first: the Humaniser needs at least {MIN_READY_CHUNKS} passages and "
                         f"{MIN_READY_CHARS} characters in the Voice tab, you have {totals['passages']} passages and "
                         f"{totals['characters']} characters", 409)
    chunks = library.chunks()
    profile = build_profile(library.profile_texts())
    steps = len(segments) + 3
    progress(0, steps, "Finding your closest writing")
    vectors: list[list[float]] = []
    for start in range(0, len(segments), EMBED_BATCH):
        vectors.extend(worker.embed([segment.text for segment in segments[start:start + EMBED_BATCH]]))
    rewritten, warnings, kept_original = [], [], 0
    for number, (segment, vector) in enumerate(zip(segments, vectors), start=1):
        if cancelled():
            raise Cancelled("stopped")
        progress(number, steps, f"Rewriting part {number} of {len(segments)}")
        samples = [chunk.text for chunk in nearest(vector, chunks)]
        answer = worker.rewrite(segment.text, samples, profile)
        rewritten.append(answer["text"])
        kept_original += 1 if answer.get("kept_original") else 0
        warnings.extend(f"Part {number}: {warning}" for warning in answer.get("warnings", []))
    after_text = join_segments(segments, rewritten)
    before_text = join_segments(segments, [segment.text for segment in segments])
    result = {"original": before_text, "rewritten": after_text, "warnings": warnings, "parts": len(segments),
              "kept_original_parts": kept_original, "before": None, "after": None, "score_error": None}
    for label, step_number, scored in (("before", len(segments) + 1, before_text), ("after", len(segments) + 2, after_text)):
        if cancelled():
            raise Cancelled("stopped")
        progress(step_number, steps, f"Checking the {label} text")
        try:
            result[label] = worker.score(scored)
        except UpstreamError as error:
            result["score_error"] = str(error)
            break
    progress(steps, steps, "Done")
    return result
