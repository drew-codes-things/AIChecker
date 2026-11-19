from dataclasses import dataclass

import torch

from .records import SentenceStat

POSITION_CHUNK = 128
GLTR_TOP1 = 1
GLTR_TOP10 = 10
GLTR_TAIL_FROM = 1000


@dataclass(frozen=True)
class TokenSignals:
    nll: torch.Tensor
    cross_entropy: torch.Tensor
    variance: torch.Tensor
    rank: torch.Tensor

    def __len__(self) -> int:
        return int(self.nll.shape[0])


def token_signals(observer_logits: torch.Tensor, performer_logits: torch.Tensor, labels: torch.Tensor) -> TokenSignals:
    if observer_logits.shape != performer_logits.shape:
        raise ValueError(f"observer logits {tuple(observer_logits.shape)} and performer logits "
                         f"{tuple(performer_logits.shape)} must have the same shape")
    if observer_logits.dim() != 2 or labels.shape != observer_logits.shape[:1]:
        raise ValueError("expected logits of shape [tokens, vocabulary] and one label per token")
    if labels.shape[0] == 0:
        raise ValueError("no tokens to score")
    parts = {"nll": [], "cross_entropy": [], "variance": [], "rank": []}
    for start in range(0, labels.shape[0], POSITION_CHUNK):
        window = slice(start, start + POSITION_CHUNK)
        observer = observer_logits[window].float()
        performer_log_probs = torch.log_softmax(performer_logits[window].float(), dim=-1)
        observer_probs = torch.softmax(observer, dim=-1)
        chunk_labels = labels[window].unsqueeze(-1)
        expected_log_prob = (observer_probs * performer_log_probs).sum(-1)
        second_moment = (observer_probs * performer_log_probs.square()).sum(-1)
        parts["nll"].append(-performer_log_probs.gather(-1, chunk_labels).squeeze(-1))
        parts["cross_entropy"].append(-expected_log_prob)
        parts["variance"].append(second_moment - expected_log_prob.square())
        actual = observer.gather(-1, chunk_labels)
        parts["rank"].append((observer > actual).sum(-1))
    return TokenSignals(**{name: torch.cat(values) for name, values in parts.items()})


def document_features(signals: TokenSignals) -> dict[str, float]:
    count = len(signals)
    if count == 0:
        raise ValueError("no tokens to summarise")
    nll_total = signals.nll.double().sum().item()
    cross_entropy_total = signals.cross_entropy.double().sum().item()
    variance_total = max(signals.variance.double().sum().item(), 1e-12)
    rank = signals.rank
    return {
        "binoculars": nll_total / cross_entropy_total,
        "fast_detect": (cross_entropy_total - nll_total) / variance_total ** 0.5,
        "gltr_top1": (rank < GLTR_TOP1).double().mean().item(),
        "gltr_top10": (rank < GLTR_TOP10).double().mean().item(),
        "gltr_tail": (rank >= GLTR_TAIL_FROM).double().mean().item(),
    }


def concat_signals(parts: list[TokenSignals]) -> TokenSignals:
    return TokenSignals(*(torch.cat([getattr(part, name) for part in parts])
                          for name in ("nll", "cross_entropy", "variance", "rank")))


def sentence_stats(signals: TokenSignals, token_ends: list[int], spans) -> list[SentenceStat]:
    if len(token_ends) != len(signals):
        raise ValueError("one token end offset is needed per scored token")
    nll = signals.nll.double().tolist()
    cross_entropy = signals.cross_entropy.double().tolist()
    totals = [[0, 0.0, 0.0] for _ in spans]
    span_index = 0
    for position, last_character in enumerate(token_ends):
        while span_index < len(spans) and spans[span_index].end <= last_character:
            span_index += 1
        if span_index == len(spans):
            break
        if spans[span_index].start <= last_character:
            totals[span_index][0] += 1
            totals[span_index][1] += nll[position]
            totals[span_index][2] += cross_entropy[position]
    return [SentenceStat(span.start, span.end, count, nll_sum, ce_sum)
            for span, (count, nll_sum, ce_sum) in zip(spans, totals)]
