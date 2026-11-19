import json
import math

import pytest

from aichecker.calibration import Calibration, CalibrationError, Scorer
from support import make_calibration, make_calibration_data


def test_logit_standardises_then_weights_each_feature():
    scorer = Scorer(("binoculars", "fast_detect"), (1.0, 0.0), (0.5, 2.0), (-2.0, 1.0), 0.5)
    assert scorer.logit({"binoculars": 0.5, "fast_detect": 4.0}) == pytest.approx(0.5 + 2.0 + 2.0)


def test_probability_is_a_sigmoid_of_the_logit_and_never_overflows():
    scorer = Scorer(("binoculars",), (0.0,), (1.0,), (1.0,), 0.0)
    assert scorer.probability({"binoculars": 0.0}) == 0.5
    assert scorer.probability({"binoculars": 1e9}) == pytest.approx(1.0)
    assert scorer.probability({"binoculars": -1e9}) == pytest.approx(0.0, abs=1e-12)
    assert scorer.probability({"binoculars": 1.0}) == pytest.approx(1 / (1 + math.exp(-1.0)))


def test_missing_feature_is_an_error_not_a_silent_zero():
    scorer = Scorer(("binoculars",), (0.0,), (1.0,), (1.0,), 0.0)
    with pytest.raises(CalibrationError, match="binoculars"):
        scorer.logit({})


def test_round_trip_through_json_keeps_everything(tmp_path):
    original = make_calibration()
    path = tmp_path / "calibration.json"
    original.save(path)
    loaded = Calibration.load(path)
    assert loaded.to_dict() == original.to_dict()


def test_missing_file_and_bad_json_say_what_is_wrong(tmp_path):
    with pytest.raises(CalibrationError, match="does not exist"):
        Calibration.load(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(CalibrationError, match="not valid JSON"):
        Calibration.load(broken)


@pytest.mark.parametrize("mutate,message", [
    (lambda d: d.pop("thresholds"), "missing"),
    (lambda d: d.update(schema=99), "schema"),
    (lambda d: d["thresholds"].pop("fpr1"), "thresholds"),
    (lambda d: d["sentence"].pop("strong"), "sentence"),
    (lambda d: d["scorer"].update(weights=[1.0]), "one number per feature"),
    (lambda d: d["scorer"].update(features=["nonsense"] * 5), "unknown features"),
    (lambda d: d["feature_quantiles"].pop("gltr_top1"), "gltr_top1"),
])
def test_incomplete_calibration_is_refused(mutate, message):
    data = make_calibration_data()
    mutate(data)
    with pytest.raises(CalibrationError, match=message):
        Calibration.from_dict(json.loads(json.dumps(data)))


def test_non_object_calibration_is_refused():
    with pytest.raises(CalibrationError):
        Calibration.from_dict([1, 2, 3])
