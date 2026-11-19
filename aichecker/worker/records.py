from dataclasses import dataclass

WINDOW_TOKENS = 512
MAX_TOKENS = 4096


class ScoringInterrupted(RuntimeError):
    pass


@dataclass(frozen=True)
class SentenceStat:
    start: int
    end: int
    tokens: int
    nll: float
    cross_entropy: float

    @property
    def binoculars(self) -> float:
        return self.nll / self.cross_entropy if self.cross_entropy > 0 else 1.0


@dataclass(frozen=True)
class DocumentScore:
    tokens: int
    scored_tokens: int
    truncated: bool
    characters_scored: int
    features: dict
    sentences: list[SentenceStat]
