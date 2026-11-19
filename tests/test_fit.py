import random

import pytest

from aichecker.calibration import FEATURE_NAMES, Calibration
from aichecker.fit import (FitError, TEST, TRAIN, VALIDATION, assign_split, build_calibration, fit_logistic,
                           sentence_thresholds)
from aichecker.metrics import auroc

DOMAINS = ("news", "poetry")
MODELS = ("gpt4", "mistral")


def make_features(rng, ai):
    shift = 1.0 if ai else 0.0
    return {"binoculars": 0.95 - 0.25 * shift + rng.gauss(0, 0.06), "fast_detect": -1.0 + 3.0 * shift + rng.gauss(0, 1.0),
            "gltr_top1": 0.30 + 0.15 * shift + rng.gauss(0, 0.04), "gltr_top10": 0.60 + 0.15 * shift + rng.gauss(0, 0.04),
            "gltr_tail": 0.03 - 0.02 * shift + abs(rng.gauss(0, 0.01))}


def make_records(count=600, seed=1):
    rng = random.Random(seed)
    records = []
    for number in range(count):
        domain = DOMAINS[number % 2]
        for ai in (False, True):
            sentences = [[rng.randint(5, 30), rng.uniform(20, 60), rng.uniform(25, 60) * (0.85 if ai else 1.0)] for _ in range(6)]
            records.append({"id": f"{number}-{int(ai)}", "source_id": f"src{number}", "label": int(ai), "domain": domain,
                            "model": MODELS[number % 2] if ai else "human", "decoding": "greedy" if ai else "",
                            "repetition_penalty": "no" if ai else "", "tokens": 100 + (number * 7) % 500,
                            "features": make_features(rng, ai), "sentences": sentences})
    return records


def test_split_is_stable_proportional_and_keeps_a_source_together():
    assert assign_split("abc", "seed") == assign_split("abc", "seed")
    names = [assign_split(f"s{n}", "seed") for n in range(4000)]
    shares = {name: names.count(name) / 4000 for name in (TRAIN, VALIDATION, TEST)}
    assert shares[TRAIN] == pytest.approx(0.60, abs=0.04)
    assert shares[VALIDATION] == pytest.approx(0.15, abs=0.04)
    assert shares[TEST] == pytest.approx(0.25, abs=0.04)


def test_logistic_fit_separates_two_classes_and_orients_binoculars_negatively():
    rng = random.Random(2)
    rows, labels = [], []
    for ai in (0, 1):
        for _ in range(400):
            features = make_features(rng, bool(ai))
            rows.append([features["binoculars"], features["fast_detect"]])
            labels.append(ai)
    scorer = fit_logistic(rows, labels, ("binoculars", "fast_detect"))
    assert scorer.weights[0] < 0 < scorer.weights[1]
    logits = [scorer.logit(dict(zip(("binoculars", "fast_detect"), row))) for row in rows]
    assert auroc(labels, logits) > 0.95


def test_class_balancing_keeps_the_probability_centred_when_classes_are_unequal():
    rng = random.Random(3)
    rows, labels = [], []
    for ai, count in ((0, 100), (1, 900)):
        for _ in range(count):
            rows.append([rng.gauss(ai, 1.0)])
            labels.append(ai)
    scorer = fit_logistic(rows, labels, ("fast_detect",))
    assert scorer.probability({"fast_detect": 0.5}) == pytest.approx(0.5, abs=0.12)


def test_fit_refuses_one_class_empty_or_constant_input():
    with pytest.raises(FitError, match="both"):
        fit_logistic([[1.0], [2.0]], [1, 1], ("fast_detect",))
    with pytest.raises(FitError):
        fit_logistic([], [], ("fast_detect",))
    with pytest.raises(FitError, match="mismatch|one label"):
        fit_logistic([[1.0]], [0, 1], ("fast_detect",))


def test_build_calibration_produces_a_valid_calibration_with_honest_held_out_numbers():
    records = make_records()
    calibration = build_calibration(records, "seed", {"observer": "a", "performer": "b"}, {"source": "synthetic"}, 512, 4096, "2026-11-18")
    assert Calibration.from_dict(calibration.to_dict()).to_dict() == calibration.to_dict()
    overall = calibration.accuracy["overall"]
    assert overall["auroc"] > 0.95
    assert 0.0 <= overall["fpr_at_fpr1_threshold"] <= 0.08
    assert overall["tpr_at_fpr5_threshold"] >= overall["tpr_at_fpr1_threshold"]
    assert calibration.thresholds["fpr1"] > calibration.thresholds["fpr5"]
    assert set(calibration.accuracy["per_domain"]) == set(DOMAINS)
    assert set(calibration.accuracy["per_generator"]) == set(MODELS)
    assert calibration.accuracy["split"]["test"]["human"] > 0
    assert calibration.accuracy["scorer_choice"]["chosen"]


def test_the_test_split_is_never_used_to_fit_anything():
    records = make_records()
    first = build_calibration(records, "seed", {}, {}, 512, 4096, "d")
    for record in records:
        if assign_split(record["source_id"], "seed") == TEST:
            record["features"] = {name: 1000.0 for name in FEATURE_NAMES}
    second = build_calibration(records, "seed", {}, {}, 512, 4096, "d")
    assert first.scorer.to_dict() == second.scorer.to_dict()
    assert first.thresholds == second.thresholds
    assert first.accuracy["overall"] != second.accuracy["overall"]


def test_validation_split_is_the_only_thing_that_picks_the_feature_set():
    records = make_records()
    calibration = build_calibration(records, "seed", {}, {}, 512, 4096, "d")
    choice = calibration.accuracy["scorer_choice"]
    assert choice["chosen"] and all(0 <= value <= 1 for value in choice["candidates_validation_auroc"].values())


def test_sentence_thresholds_come_from_human_sentences_with_enough_tokens():
    records = make_records(200)
    thresholds = sentence_thresholds([r for r in records if assign_split(r["source_id"], "seed") == TRAIN])
    assert thresholds["min_tokens"] == 8 and thresholds["strong"] < thresholds["some"]
    with pytest.raises(FitError):
        sentence_thresholds([])


def test_a_split_without_both_classes_is_refused():
    only_human = [r for r in make_records(50) if r["label"] == 0]
    with pytest.raises(FitError, match="needs both"):
        build_calibration(only_human, "seed", {}, {}, 512, 4096)
