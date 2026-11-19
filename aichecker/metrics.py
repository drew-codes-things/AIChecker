from collections.abc import Sequence


class MetricError(ValueError):
    pass


def _require_both_classes(labels: Sequence[int]) -> None:
    if not any(labels) or all(labels):
        raise MetricError("need at least one human (0) and one AI (1) example")
    if any(label not in (0, 1) for label in labels):
        raise MetricError("labels must be 0 for human or 1 for AI")


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores):
        raise MetricError("labels and scores differ in length")
    _require_both_classes(labels)
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum_ai = 0.0
    position = 0
    while position < len(order):
        tie_end = position
        while tie_end + 1 < len(order) and scores[order[tie_end + 1]] == scores[order[position]]:
            tie_end += 1
        average_rank = (position + tie_end) / 2 + 1
        rank_sum_ai += average_rank * sum(labels[order[index]] for index in range(position, tie_end + 1))
        position = tie_end + 1
    ai_count = sum(labels)
    human_count = len(labels) - ai_count
    return (rank_sum_ai - ai_count * (ai_count + 1) / 2) / (ai_count * human_count)


def threshold_at_fpr(human_scores: Sequence[float], false_positive_rate: float) -> float:
    if not human_scores:
        raise MetricError("no human scores to set a threshold from")
    if not 0 < false_positive_rate < 1:
        raise MetricError("false positive rate must be between 0 and 1")
    ordered = sorted(human_scores, reverse=True)
    allowed = int(false_positive_rate * len(ordered))
    return ordered[allowed] if allowed < len(ordered) else ordered[-1]


def rates_at_threshold(labels: Sequence[int], scores: Sequence[float], threshold: float) -> tuple[float, float]:
    _require_both_classes(labels)
    flagged_ai = sum(1 for label, score in zip(labels, scores) if label == 1 and score > threshold)
    flagged_human = sum(1 for label, score in zip(labels, scores) if label == 0 and score > threshold)
    ai_total = sum(labels)
    human_total = len(labels) - ai_total
    return flagged_ai / ai_total, flagged_human / human_total


def tpr_at_fpr(labels: Sequence[int], scores: Sequence[float], false_positive_rate: float) -> float:
    _require_both_classes(labels)
    human_scores = [score for label, score in zip(labels, scores) if label == 0]
    threshold = threshold_at_fpr(human_scores, false_positive_rate)
    return rates_at_threshold(labels, scores, threshold)[0]


def quantiles(values: Sequence[float], points: int) -> list[float]:
    if not values:
        raise MetricError("no values to take quantiles of")
    if points < 2:
        raise MetricError("need at least two quantile points")
    ordered = sorted(values)
    last = len(ordered) - 1
    return [ordered[round(index / (points - 1) * last)] for index in range(points)]


def percentile_rank(sorted_quantile_points: Sequence[float], value: float) -> float:
    if not sorted_quantile_points:
        raise MetricError("no quantile points")
    below = sum(1 for point in sorted_quantile_points if point < value)
    return below / len(sorted_quantile_points)
