from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ..sentences import split_sentences
from .records import MAX_TOKENS, WINDOW_TOKENS, DocumentScore, ScoringInterrupted
from .signals import TokenSignals, concat_signals, document_features, sentence_stats, token_signals

BATCH_TOKEN_BUDGET = 2048
START_TOKEN = "<|endoftext|>"


@dataclass(frozen=True)
class _Window:
    document: int
    order: int
    ids: list[int]


def plan_windows(token_count: int, window_tokens: int = WINDOW_TOKENS, max_tokens: int = MAX_TOKENS) -> list[tuple[int, int]]:
    content_per_window = window_tokens - 1
    limit = min(token_count, max_tokens)
    return [(start, min(start + content_per_window, limit)) for start in range(0, limit, content_per_window)]


def batch_by_budget(lengths: Sequence[int], budget: int = BATCH_TOKEN_BUDGET) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda index: -lengths[index])
    batches: list[list[int]] = []
    current: list[int] = []
    for index in order:
        longest = lengths[current[0]] if current else lengths[index]
        if current and longest * (len(current) + 1) > budget:
            batches.append(current)
            current = []
        current.append(index)
    if current:
        batches.append(current)
    return batches


class PairScorer:
    def __init__(self, observer_dir: Path, performer_dir: Path, device: str = "cuda"):
        self.observer_dir = Path(observer_dir)
        self.performer_dir = Path(performer_dir)
        self.device = device
        self.observer = None
        self.performer = None
        self.tokenizer = None
        self.start_id = None

    @property
    def loaded(self) -> bool:
        return self.observer is not None

    def load(self) -> None:
        if self.loaded:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.observer_dir)
        other = AutoTokenizer.from_pretrained(self.performer_dir)
        if tokenizer.get_vocab() != other.get_vocab():
            raise RuntimeError("the observer and performer tokenizers differ; Binoculars needs a shared vocabulary")
        start_id = tokenizer.convert_tokens_to_ids(START_TOKEN)
        if start_id is None or start_id == tokenizer.unk_token_id:
            raise RuntimeError(f"the tokenizer has no {START_TOKEN} token to start each window with")
        observer = self._load_model(AutoModelForCausalLM, self.observer_dir)
        try:
            performer = self._load_model(AutoModelForCausalLM, self.performer_dir)
        except Exception:
            observer = None
            torch.cuda.empty_cache()
            raise
        self.tokenizer, self.start_id = tokenizer, start_id
        self.observer, self.performer = observer, performer

    def _load_model(self, model_class, directory: Path):
        model = model_class.from_pretrained(directory, dtype=torch.bfloat16)
        return model.to(self.device).eval()

    def unload(self) -> None:
        self.observer = None
        self.performer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _run_batch(self, windows: list[_Window]) -> list[TokenSignals]:
        longest = max(len(window.ids) for window in windows) + 1
        input_ids = torch.full((len(windows), longest), self.start_id, dtype=torch.long)
        attention = torch.zeros((len(windows), longest), dtype=torch.long)
        for row, window in enumerate(windows):
            row_ids = [self.start_id] + window.ids
            input_ids[row, :len(row_ids)] = torch.tensor(row_ids)
            attention[row, :len(row_ids)] = 1
        input_ids, attention = input_ids.to(self.device), attention.to(self.device)
        observer_logits = self.observer(input_ids=input_ids, attention_mask=attention).logits
        performer_logits = self.performer(input_ids=input_ids, attention_mask=attention).logits
        results = []
        for row, window in enumerate(windows):
            count = len(window.ids)
            labels = input_ids[row, 1:count + 1]
            results.append(token_signals(observer_logits[row, :count], performer_logits[row, :count], labels))
        return results

    def score_documents(self, texts: Sequence[str], should_yield: Callable[[], bool] | None = None) -> list[DocumentScore]:
        if not self.loaded:
            raise RuntimeError("the model pair is not loaded")
        encodings = [self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True) for text in texts]
        windows: list[_Window] = []
        for document, encoding in enumerate(encodings):
            for order, (start, end) in enumerate(plan_windows(len(encoding["input_ids"]))):
                windows.append(_Window(document, order, encoding["input_ids"][start:end]))
        results: dict[tuple[int, int], TokenSignals] = {}
        for batch in batch_by_budget([len(window.ids) + 1 for window in windows]):
            if should_yield is not None and should_yield():
                raise ScoringInterrupted("the model engine started answering, so the check was stopped to give the GPU back")
            chosen = [windows[index] for index in batch]
            for window, signals in zip(chosen, self._run_batch(chosen)):
                results[(window.document, window.order)] = signals
        return [self._assemble(texts[index], encodings[index], results, index) for index in range(len(texts))]

    def score(self, text: str, should_yield: Callable[[], bool] | None = None) -> DocumentScore:
        return self.score_documents([text], should_yield)[0]

    def _assemble(self, text: str, encoding, results, document: int) -> DocumentScore:
        total_tokens = len(encoding["input_ids"])
        order = 0
        parts = []
        while (document, order) in results:
            parts.append(results[(document, order)])
            order += 1
        if not parts:
            raise ValueError("there is no text to score")
        signals = concat_signals(parts)
        scored = len(signals)
        offsets = encoding["offset_mapping"][:scored]
        token_ends = [max(end - 1, start) for start, end in offsets]
        spans = [span for span in split_sentences(text) if span.start <= token_ends[-1]]
        return DocumentScore(
            tokens=total_tokens, scored_tokens=scored, truncated=scored < total_tokens,
            characters_scored=token_ends[-1] + 1, features=document_features(signals),
            sentences=sentence_stats(signals, token_ends, spans))
