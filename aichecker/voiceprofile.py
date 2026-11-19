import re
from collections.abc import Iterable

from .sentences import split_sentences

WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
CONTRACTION = re.compile(r"\b\w+(?:n't|'(?:s|re|ll|ve|d|m))\b", re.IGNORECASE)
FIRST_PERSON = frozenset({"i", "me", "my", "mine", "we", "our", "us", "i'm", "i've", "i'll", "i'd"})
SHORT_SENTENCE_WORDS = 8
LONG_SENTENCE_WORDS = 25
MIN_WORDS_FOR_PROFILE = 80
PER_HUNDRED = 100.0
LONG_DASH = "\u2014"
SHORT_DASH = "–"
STOCK_PHRASES = (
    "delve", "tapestry", "in today's fast-paced world", "it is important to note", "it's worth noting",
    "furthermore", "moreover", "in conclusion", "navigate the complexities", "a testament to",
    "plays a crucial role", "in the realm of", "ever-evolving", "unlock the potential", "game-changer",
)


class ProfileError(ValueError):
    pass


def _normalise_apostrophes(text: str) -> str:
    return text.replace("’", "'")


def build_profile(texts: Iterable[str]) -> dict:
    joined = "\n\n".join(text for text in texts if text and text.strip())
    cleaned = _normalise_apostrophes(joined)
    words = WORD.findall(cleaned)
    if len(words) < MIN_WORDS_FOR_PROFILE:
        raise ProfileError(f"need at least {MIN_WORDS_FOR_PROFILE} words of writing to measure a voice, got {len(words)}")
    sentences = [span.slice_of(cleaned) for span in split_sentences(cleaned)]
    sentence_lengths = [len(WORD.findall(sentence)) for sentence in sentences] or [len(words)]
    word_count = len(words)
    sentence_count = len(sentence_lengths)
    paragraphs = [part for part in re.split(r"\n\s*\n", cleaned) if part.strip()]
    return {
        "words": word_count,
        "sentences": sentence_count,
        "average_sentence_words": round(sum(sentence_lengths) / sentence_count, 1),
        "short_sentence_share": round(sum(1 for n in sentence_lengths if n <= SHORT_SENTENCE_WORDS) / sentence_count, 2),
        "long_sentence_share": round(sum(1 for n in sentence_lengths if n >= LONG_SENTENCE_WORDS) / sentence_count, 2),
        "contractions_per_100_words": round(len(CONTRACTION.findall(cleaned)) / word_count * PER_HUNDRED, 1),
        "commas_per_100_words": round(cleaned.count(",") / word_count * PER_HUNDRED, 1),
        "semicolons_per_100_words": round(cleaned.count(";") / word_count * PER_HUNDRED, 2),
        "dashes_per_100_words": round((cleaned.count(LONG_DASH) + cleaned.count(SHORT_DASH) + cleaned.count(" - ")) / word_count * PER_HUNDRED, 2),
        "questions_per_100_sentences": round(cleaned.count("?") / sentence_count * PER_HUNDRED, 1),
        "exclamations_per_100_sentences": round(cleaned.count("!") / sentence_count * PER_HUNDRED, 1),
        "first_person_per_100_words": round(sum(1 for word in words if word.lower() in FIRST_PERSON) / word_count * PER_HUNDRED, 1),
        "average_word_letters": round(sum(len(word) for word in words) / word_count, 1),
        "average_paragraph_words": round(word_count / len(paragraphs), 0),
    }


def guidance_lines(profile: dict) -> list[str]:
    lines = [
        f"Sentences average about {profile['average_sentence_words']:g} words; "
        f"{round(profile['short_sentence_share'] * 100)}% are short (8 words or fewer) and "
        f"{round(profile['long_sentence_share'] * 100)}% are long (25 words or more).",
        f"Contractions appear about {profile['contractions_per_100_words']:g} times per 100 words"
        + (" (use them freely)." if profile["contractions_per_100_words"] >= 1.5 else " (use few or none)."),
        f"Commas about {profile['commas_per_100_words']:g} per 100 words.",
    ]
    if profile["first_person_per_100_words"] >= 1.0:
        lines.append("Writes in the first person (I, we, my).")
    if profile["questions_per_100_sentences"] >= 5:
        lines.append("Asks the occasional question.")
    if profile["semicolons_per_100_words"] < 0.05:
        lines.append("Does not use semicolons.")
    if profile["dashes_per_100_words"] < 0.05:
        lines.append("Does not use dashes; uses commas or full stops instead.")
    lines.append(f"Average word length {profile['average_word_letters']:g} letters.")
    return lines
