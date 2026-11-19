import hashlib
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import date

from .calibration import FEATURE_NAMES, Calibration, Scorer
from .metrics import auroc, quantiles, rates_at_threshold, threshold_at_fpr, tpr_at_fpr

TRAIN, VALIDATION, TEST = "train", "validation", "test"
SPLIT_SHARES = ((TRAIN, 60), (VALIDATION, 15), (TEST, 25))
CANDIDATES = (("binoculars",), ("fast_detect",), ("binoculars", "fast_detect"), FEATURE_NAMES)
QUANTILE_POINTS = 41
SENTENCE_MIN_TOKENS = 8
SENTENCE_STRONG_QUANTILE = 0.05
SENTENCE_SOME_QUANTILE = 0.20
MIN_TOKENS_FOR_READING = 48
LENGTH_BUCKETS = ((0, 128), (128, 256), (256, 400), (400, 10**9))
RIDGE = 1.0
NEWTON_STEPS = 60
NEWTON_TOLERANCE = 1e-7
MIN_GROUP_SIZE = 20


class FitError(ValueError):
    pass


def assign_split(source_id: str, seed: str) -> str:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    upper = 0
    for name, share in SPLIT_SHARES:
        upper += share
        if bucket < upper:
            return name
    return TEST


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    rows = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise FitError("the feature matrix is singular; a feature may be constant")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for row in range(size):
            if row != column:
                factor = rows[row][column] / rows[column][column]
                rows[row] = [value - factor * other for value, other in zip(rows[row], rows[column])]
    return [rows[index][size] / rows[index][index] for index in range(size)]


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, value))))


def fit_logistic(features: Sequence[Sequence[float]], labels: Sequence[int], names: Sequence[str]) -> Scorer:
    if len(features) != len(labels) or not features:
        raise FitError("need one label per feature row and at least one row")
    if len(set(labels)) != 2:
        raise FitError("need both human and AI rows to fit")
    count, width = len(features), len(names)
    means = [sum(row[column] for row in features) / count for column in range(width)]
    stds = [math.sqrt(sum((row[column] - means[column]) ** 2 for row in features) / count) or 1.0 for column in range(width)]
    design = [[1.0] + [(row[column] - means[column]) / stds[column] for column in range(width)] for row in features]
    positives = sum(labels)
    weights_by_class = {1: count / (2 * positives), 0: count / (2 * (count - positives))}
    sample_weights = [weights_by_class[label] for label in labels]
    beta = [0.0] * (width + 1)
    for _ in range(NEWTON_STEPS):
        gradient = [0.0] * (width + 1)
        hessian = [[0.0] * (width + 1) for _ in range(width + 1)]
        for row, label, weight in zip(design, labels, sample_weights):
            probability = _sigmoid(sum(b * x for b, x in zip(beta, row)))
            residual = weight * (label - probability)
            curvature = weight * probability * (1 - probability)
            for left in range(width + 1):
                gradient[left] += residual * row[left]
                for right in range(left, width + 1):
                    hessian[left][right] += curvature * row[left] * row[right]
        for left in range(width + 1):
            for right in range(left):
                hessian[left][right] = hessian[right][left]
        for index in range(1, width + 1):
            gradient[index] -= RIDGE * beta[index]
            hessian[index][index] += RIDGE
        step = _solve(hessian, gradient)
        beta = [b + s for b, s in zip(beta, step)]
        if max(abs(s) for s in step) < NEWTON_TOLERANCE:
            break
    return Scorer(tuple(names), tuple(means), tuple(stds), tuple(beta[1:]), beta[0])


def logits_for(scorer: Scorer, records: Sequence[Mapping]) -> list[float]:
    return [scorer.logit(record["features"]) for record in records]


def _auroc_or_none(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(set(labels)) < 2 or min(labels.count(0), labels.count(1)) < MIN_GROUP_SIZE // 4:
        return None
    return round(auroc(labels, scores), 4)


def summarise(records: Sequence[Mapping], logits: Sequence[float], thresholds: Mapping[str, float]) -> dict:
    labels = [record["label"] for record in records]
    humans, ais = labels.count(0), labels.count(1)
    summary: dict = {"n_human": humans, "n_ai": ais}
    if not humans or not ais:
        return summary
    summary["auroc"] = _auroc_or_none(labels, logits)
    if humans >= MIN_GROUP_SIZE:
        summary["tpr_at_1pct_fpr"] = round(tpr_at_fpr(labels, logits, 0.01), 4)
        summary["tpr_at_5pct_fpr"] = round(tpr_at_fpr(labels, logits, 0.05), 4)
    for key, threshold in thresholds.items():
        true_rate, false_rate = rates_at_threshold(labels, logits, threshold)
        summary[f"tpr_at_{key}_threshold"] = round(true_rate, 4)
        summary[f"fpr_at_{key}_threshold"] = round(false_rate, 4)
    return summary


def _group(records: Sequence[Mapping], key: Callable[[Mapping], str | None]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        name = key(record)
        if name:
            groups[name].append(index)
    return groups


def _bucket_name(tokens: int) -> str:
    for low, high in LENGTH_BUCKETS:
        if low <= tokens < high:
            return f"{low}-{high}" if high < 10**9 else f"{low}+"
    return "other"


def breakdowns(records: Sequence[Mapping], logits: Sequence[float], thresholds: Mapping[str, float]) -> dict:
    humans = [index for index, record in enumerate(records) if record["label"] == 0]
    result: dict = {"per_domain": {}, "per_generator": {}, "per_decoding": {}, "per_length": {}}

    def pick(indices):
        return [records[i] for i in indices], [logits[i] for i in indices]

    for domain, indices in sorted(_group(records, lambda r: r["domain"]).items()):
        chosen, scores = pick(indices)
        result["per_domain"][domain] = summarise(chosen, scores, thresholds)
    generators = _group(records, lambda r: r["model"] if r["label"] == 1 else None)
    for model, indices in sorted(generators.items()):
        chosen, scores = pick(humans + indices)
        entry = summarise(chosen, scores, thresholds)
        for field in ("n_human", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "fpr_at_fpr1_threshold", "fpr_at_fpr5_threshold"):
            entry.pop(field, None)
        result["per_generator"][model] = entry
    decodings = _group(records, lambda r: f"{r['decoding']}, repetition penalty {r['repetition_penalty']}" if r["label"] == 1 else None)
    for name, indices in sorted(decodings.items()):
        chosen, scores = pick(humans + indices)
        entry = summarise(chosen, scores, thresholds)
        for field in ("n_human", "tpr_at_1pct_fpr", "tpr_at_5pct_fpr", "fpr_at_fpr1_threshold", "fpr_at_fpr5_threshold"):
            entry.pop(field, None)
        result["per_decoding"][name] = entry
    for bucket, indices in sorted(_group(records, lambda r: _bucket_name(r["tokens"])).items(), key=lambda pair: int(pair[0].split("-")[0].rstrip("+"))):
        chosen, scores = pick(indices)
        result["per_length"][bucket] = summarise(chosen, scores, thresholds)
    return result


def sentence_pairs(records: Sequence[Mapping]) -> list[tuple[int, float]]:
    pairs = []
    for record in records:
        for tokens, nll, cross_entropy in record["sentences"]:
            if tokens >= SENTENCE_MIN_TOKENS and cross_entropy > 0:
                pairs.append((record["label"], nll / cross_entropy))
    return pairs


def _quantile_value(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def sentence_thresholds(train: Sequence[Mapping]) -> dict:
    human_scores = [score for label, score in sentence_pairs(train) if label == 0]
    if len(human_scores) < MIN_GROUP_SIZE:
        raise FitError("too few human sentences to set sentence thresholds")
    return {"min_tokens": SENTENCE_MIN_TOKENS, "strong": round(_quantile_value(human_scores, SENTENCE_STRONG_QUANTILE), 4),
            "some": round(_quantile_value(human_scores, SENTENCE_SOME_QUANTILE), 4)}


def sentence_report(test: Sequence[Mapping], thresholds: Mapping) -> dict:
    pairs = sentence_pairs(test)
    labels = [label for label, _ in pairs]
    scores = [-score for _, score in pairs]
    report = {"n_human_sentences": labels.count(0), "n_ai_sentences": labels.count(1), "auroc": round(auroc(labels, scores), 4)}
    for level in ("strong", "some"):
        flagged = [(label, score) for label, score in pairs if score < thresholds[level]]
        report[f"ai_sentences_marked_{level}"] = round(sum(1 for label, _ in flagged if label == 1) / max(labels.count(1), 1), 4)
        report[f"human_sentences_marked_{level}"] = round(sum(1 for label, _ in flagged if label == 0) / max(labels.count(0), 1), 4)
    return report


def feature_quantile_table(train: Sequence[Mapping]) -> dict:
    table = {}
    for name in FEATURE_NAMES:
        table[name] = {
            "human": [round(v, 5) for v in quantiles([r["features"][name] for r in train if r["label"] == 0], QUANTILE_POINTS)],
            "ai": [round(v, 5) for v in quantiles([r["features"][name] for r in train if r["label"] == 1], QUANTILE_POINTS)],
        }
    return table


def build_calibration(records: Sequence[Mapping], seed: str, pair: Mapping, data_notes: Mapping,
                      window_tokens: int, max_tokens: int, created: str | None = None) -> Calibration:
    split: dict[str, list[Mapping]] = {TRAIN: [], VALIDATION: [], TEST: []}
    for record in records:
        split[assign_split(record["source_id"], seed)].append(record)
    for name, part in split.items():
        if not any(r["label"] == 0 for r in part) or not any(r["label"] == 1 for r in part):
            raise FitError(f"the {name} split needs both human and AI documents")
    train_labels = [r["label"] for r in split[TRAIN]]
    validation_labels = [r["label"] for r in split[VALIDATION]]
    candidates, best = {}, None
    for names in CANDIDATES:
        scorer = fit_logistic([[r["features"][n] for n in names] for r in split[TRAIN]], train_labels, names)
        validation_auroc = auroc(validation_labels, logits_for(scorer, split[VALIDATION]))
        candidates["+".join(names)] = round(validation_auroc, 4)
        if best is None or validation_auroc > best[0] + 0.002:
            best = (validation_auroc, scorer)
    scorer = best[1]
    train_logits = logits_for(scorer, split[TRAIN])
    train_human = [score for r, score in zip(split[TRAIN], train_logits) if r["label"] == 0]
    thresholds = {"fpr1": threshold_at_fpr(train_human, 0.01), "fpr5": threshold_at_fpr(train_human, 0.05)}
    test_logits = logits_for(scorer, split[TEST])
    sentence = sentence_thresholds(split[TRAIN])
    accuracy = {
        "scorer_choice": {"candidates_validation_auroc": candidates, "chosen": list(scorer.features)},
        "overall": summarise(split[TEST], test_logits, thresholds),
        **breakdowns(split[TEST], test_logits, thresholds),
        "sentence_level": sentence_report(split[TEST], sentence),
        "split": {name: {"human": sum(1 for r in part if r["label"] == 0), "ai": sum(1 for r in part if r["label"] == 1)}
                  for name, part in split.items()},
        "data": dict(data_notes),
    }
    return Calibration(scorer, thresholds, MIN_TOKENS_FOR_READING, window_tokens, max_tokens, sentence,
                       feature_quantile_table(split[TRAIN]), accuracy, dict(pair), created or date.today().isoformat())
