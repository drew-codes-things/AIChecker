import math

import pytest

torch = pytest.importorskip("torch")

from aichecker.sentences import Span
from aichecker.worker import signals
from aichecker.worker.scorer import batch_by_budget, plan_windows
from aichecker.worker.signals import concat_signals, document_features, sentence_stats, token_signals


def random_logits(tokens, vocabulary, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, vocabulary, generator=generator) * 3.0


def naive_signals(observer, performer, labels):
    nll, cross_entropy, variance, rank = [], [], [], []
    for position in range(labels.shape[0]):
        p = [math.exp(x) for x in torch.log_softmax(observer[position].double(), -1).tolist()]
        log_q = torch.log_softmax(performer[position].double(), -1).tolist()
        label = int(labels[position])
        mean = sum(pi * lq for pi, lq in zip(p, log_q))
        nll.append(-log_q[label])
        cross_entropy.append(-mean)
        variance.append(sum(pi * lq * lq for pi, lq in zip(p, log_q)) - mean * mean)
        rank.append(sum(1 for value in observer[position].tolist() if value > observer[position, label].item()))
    return nll, cross_entropy, variance, rank


def test_per_token_signals_match_a_plain_loop_implementation():
    observer, performer = random_logits(9, 40, 1), random_logits(9, 40, 2)
    labels = torch.randint(0, 40, (9,), generator=torch.Generator().manual_seed(3))
    result = token_signals(observer, performer, labels)
    nll, cross_entropy, variance, rank = naive_signals(observer, performer, labels)
    assert result.nll.tolist() == pytest.approx(nll, rel=1e-4)
    assert result.cross_entropy.tolist() == pytest.approx(cross_entropy, rel=1e-4)
    assert result.variance.tolist() == pytest.approx(variance, rel=1e-3, abs=1e-4)
    assert result.rank.tolist() == rank


def test_chunking_over_positions_does_not_change_the_numbers(monkeypatch):
    observer, performer = random_logits(50, 30, 4), random_logits(50, 30, 5)
    labels = torch.randint(0, 30, (50,), generator=torch.Generator().manual_seed(6))
    whole = token_signals(observer, performer, labels)
    monkeypatch.setattr(signals, "POSITION_CHUNK", 7)
    chunked = token_signals(observer, performer, labels)
    for name in ("nll", "cross_entropy", "variance", "rank"):
        assert torch.allclose(getattr(whole, name).double(), getattr(chunked, name).double())


def test_half_precision_logits_are_accepted():
    observer, performer = random_logits(6, 20, 7).bfloat16(), random_logits(6, 20, 8).bfloat16()
    labels = torch.zeros(6, dtype=torch.long)
    result = token_signals(observer, performer, labels)
    assert result.nll.dtype == torch.float32 and len(result) == 6


def test_a_confident_correct_performer_scores_as_machine_like():
    labels = torch.tensor([3, 1, 4, 1, 5, 9, 2, 6])
    sharp = torch.full((8, 20), -10.0)
    sharp[torch.arange(8), labels] = 10.0
    features = document_features(token_signals(sharp, sharp, labels))
    assert features["binoculars"] < 0.01
    assert features["gltr_top1"] == 1.0 and features["gltr_top10"] == 1.0 and features["gltr_tail"] == 0.0


def test_surprising_text_scores_as_human_like():
    labels = torch.tensor([0] * 40)
    wrong = torch.zeros(40, 20)
    wrong[:, 1] = 8.0
    features = document_features(token_signals(wrong, wrong, labels))
    assert features["binoculars"] > 1.0
    assert features["gltr_top1"] == 0.0
    assert features["fast_detect"] < 0


def test_binoculars_is_the_ratio_of_summed_nll_to_summed_cross_entropy():
    observer, performer = random_logits(30, 25, 9), random_logits(30, 25, 10)
    labels = torch.randint(0, 25, (30,), generator=torch.Generator().manual_seed(11))
    result = token_signals(observer, performer, labels)
    features = document_features(result)
    assert features["binoculars"] == pytest.approx(result.nll.sum().item() / result.cross_entropy.sum().item(), rel=1e-6)
    expected_fast = (result.cross_entropy.sum().item() - result.nll.sum().item()) / math.sqrt(result.variance.sum().item())
    assert features["fast_detect"] == pytest.approx(expected_fast, rel=1e-4)


def test_gltr_buckets_use_the_observers_rank_of_the_actual_token():
    vocabulary = 2000
    observer = torch.arange(vocabulary, dtype=torch.float32).flip(0).repeat(5, 1)
    labels = torch.tensor([0, 9, 10, 999, 1500])
    result = token_signals(observer, observer, labels)
    assert result.rank.tolist() == [0, 9, 10, 999, 1500]
    features = document_features(result)
    assert features["gltr_top1"] == pytest.approx(1 / 5)
    assert features["gltr_top10"] == pytest.approx(2 / 5)
    assert features["gltr_tail"] == pytest.approx(1 / 5)


def test_shape_problems_are_reported_not_guessed():
    with pytest.raises(ValueError, match="same shape"):
        token_signals(torch.zeros(3, 5), torch.zeros(3, 6), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="one label per token"):
        token_signals(torch.zeros(3, 5), torch.zeros(3, 5), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="no tokens"):
        document_features(token_signals(torch.zeros(0, 5), torch.zeros(0, 5), torch.zeros(0, dtype=torch.long)))


def test_concatenated_windows_equal_one_pass():
    observer, performer = random_logits(20, 15, 12), random_logits(20, 15, 13)
    labels = torch.randint(0, 15, (20,), generator=torch.Generator().manual_seed(14))
    whole = token_signals(observer, performer, labels)
    joined = concat_signals([token_signals(observer[:8], performer[:8], labels[:8]),
                             token_signals(observer[8:], performer[8:], labels[8:])])
    assert document_features(joined) == pytest.approx(document_features(whole))


def test_sentence_stats_assign_each_token_to_the_sentence_that_contains_its_last_character():
    observer = random_logits(6, 12, 15)
    labels = torch.randint(0, 12, (6,), generator=torch.Generator().manual_seed(16))
    result = token_signals(observer, observer, labels)
    spans = [Span(0, 11), Span(13, 20)]
    token_ends = [2, 5, 10, 11, 15, 19]
    stats = sentence_stats(result, token_ends, spans)
    assert [stat.tokens for stat in stats] == [3, 2]
    assert stats[0].nll == pytest.approx(result.nll[:3].sum().item(), rel=1e-5)
    assert stats[1].nll == pytest.approx(result.nll[4:].sum().item(), rel=1e-5)
    assert stats[0].binoculars == pytest.approx(stats[0].nll / stats[0].cross_entropy)


def test_sentence_stats_need_one_end_per_token():
    result = token_signals(torch.zeros(3, 5), torch.zeros(3, 5), torch.zeros(3, dtype=torch.long))
    with pytest.raises(ValueError, match="one token end"):
        sentence_stats(result, [1, 2], [Span(0, 5)])


def test_windows_cover_the_text_without_overlap_and_stop_at_the_limit():
    assert plan_windows(0) == []
    assert plan_windows(100) == [(0, 100)]
    assert plan_windows(511) == [(0, 511)]
    assert plan_windows(512) == [(0, 511), (511, 512)]
    windows = plan_windows(5000, max_tokens=4096)
    assert windows[0] == (0, 511) and windows[-1][1] == 4096
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))


def test_batches_respect_the_token_budget_and_keep_every_window_once():
    lengths = [512, 512, 100, 60, 300, 512, 40]
    batches = batch_by_budget(lengths, budget=1100)
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    for batch in batches:
        assert max(lengths[index] for index in batch) * len(batch) <= 1100 or len(batch) == 1
