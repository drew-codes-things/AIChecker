import pytest

from aichecker.metrics import (MetricError, auroc, percentile_rank, quantiles, rates_at_threshold,
                               threshold_at_fpr, tpr_at_fpr)


def test_auroc_is_one_for_perfect_separation_and_zero_when_reversed():
    labels = [0, 0, 0, 1, 1, 1]
    assert auroc(labels, [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) == 1.0
    assert auroc(labels, [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) == 0.0


def test_auroc_is_half_when_every_score_ties():
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auroc_matches_pairwise_definition_with_ties():
    labels = [0, 1, 0, 1, 1, 0]
    scores = [0.2, 0.2, 0.5, 0.9, 0.5, 0.1]
    pairs = [(human, ai) for human, hl in zip(scores, labels) if hl == 0 for ai, al in zip(scores, labels) if al == 1]
    expected = sum(1.0 if ai > human else 0.5 if ai == human else 0.0 for human, ai in pairs) / len(pairs)
    assert auroc(labels, scores) == pytest.approx(expected)


def test_auroc_needs_both_classes_and_equal_lengths():
    with pytest.raises(MetricError):
        auroc([1, 1], [0.1, 0.2])
    with pytest.raises(MetricError):
        auroc([0, 1], [0.1])
    with pytest.raises(MetricError):
        auroc([0, 2], [0.1, 0.2])


def test_threshold_at_fpr_lets_through_the_stated_share_of_humans():
    human = [float(index) for index in range(100)]
    threshold = threshold_at_fpr(human, 0.05)
    assert sum(1 for score in human if score > threshold) == 5


def test_threshold_at_fpr_rejects_bad_input():
    with pytest.raises(MetricError):
        threshold_at_fpr([], 0.01)
    with pytest.raises(MetricError):
        threshold_at_fpr([1.0], 0.0)
    with pytest.raises(MetricError):
        threshold_at_fpr([1.0], 1.0)


def test_tpr_at_fpr_counts_ai_above_the_human_threshold():
    labels = [0] * 100 + [1] * 10
    scores = [float(index) for index in range(100)] + [97.5, 98.5, 99.5, 50.0, 60.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    assert tpr_at_fpr(labels, scores, 0.05) == pytest.approx(0.8)


def test_rates_at_threshold_reports_true_and_false_positive_rates():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.6, 0.3, 0.7, 0.8, 0.4, 0.9]
    assert rates_at_threshold(labels, scores, 0.5) == (0.75, 0.25)


def test_quantiles_include_the_extremes_and_are_sorted():
    points = quantiles([5.0, 1.0, 3.0, 2.0, 4.0], 5)
    assert points == [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(MetricError):
        quantiles([], 5)
    with pytest.raises(MetricError):
        quantiles([1.0], 1)


def test_percentile_rank_is_the_share_of_points_below_the_value():
    points = [0.0, 1.0, 2.0, 3.0]
    assert percentile_rank(points, -1.0) == 0.0
    assert percentile_rank(points, 1.5) == 0.5
    assert percentile_rank(points, 9.0) == 1.0
