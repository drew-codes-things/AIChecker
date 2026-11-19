import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
FEATURE_NAMES = ("binoculars", "fast_detect", "gltr_top1", "gltr_top10", "gltr_tail")
REQUIRED_KEYS = ("schema", "scorer", "thresholds", "min_tokens", "window_tokens", "max_tokens", "sentence",
                 "feature_quantiles", "accuracy", "pair", "created")
THRESHOLD_KEYS = ("fpr1", "fpr5")
SENTENCE_KEYS = ("min_tokens", "strong", "some")
LOGIT_CLAMP = 30.0


class CalibrationError(ValueError):
    pass


@dataclass(frozen=True)
class Scorer:
    features: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float

    def logit(self, values: Mapping[str, float]) -> float:
        total = self.bias
        for name, mean, std, weight in zip(self.features, self.mean, self.std, self.weights):
            if name not in values:
                raise CalibrationError(f"feature {name!r} is missing from the scored values")
            total += weight * (values[name] - mean) / (std or 1.0)
        return total

    def probability(self, values: Mapping[str, float]) -> float:
        clamped = max(-LOGIT_CLAMP, min(LOGIT_CLAMP, self.logit(values)))
        return 1.0 / (1.0 + math.exp(-clamped))

    def to_dict(self) -> dict:
        return {"features": list(self.features), "mean": list(self.mean), "std": list(self.std),
                "weights": list(self.weights), "bias": self.bias}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Scorer":
        try:
            features = tuple(data["features"])
            parts = (tuple(float(x) for x in data["mean"]), tuple(float(x) for x in data["std"]),
                     tuple(float(x) for x in data["weights"]))
            bias = float(data["bias"])
        except (KeyError, TypeError, ValueError) as error:
            raise CalibrationError(f"scorer section is incomplete or not numeric: {error}") from error
        if any(len(part) != len(features) for part in parts):
            raise CalibrationError("scorer mean, std and weights must each have one number per feature")
        unknown = [name for name in features if name not in FEATURE_NAMES]
        if unknown:
            raise CalibrationError(f"scorer uses unknown features {unknown}")
        return cls(features, parts[0], parts[1], parts[2], bias)


@dataclass(frozen=True)
class Calibration:
    scorer: Scorer
    thresholds: Mapping[str, float]
    min_tokens: int
    window_tokens: int
    max_tokens: int
    sentence: Mapping[str, float]
    feature_quantiles: Mapping[str, Mapping[str, list]]
    accuracy: Mapping
    pair: Mapping
    created: str

    def to_dict(self) -> dict:
        return {"schema": SCHEMA_VERSION, "scorer": self.scorer.to_dict(), "thresholds": dict(self.thresholds),
                "min_tokens": self.min_tokens, "window_tokens": self.window_tokens, "max_tokens": self.max_tokens,
                "sentence": dict(self.sentence), "feature_quantiles": self.feature_quantiles,
                "accuracy": self.accuracy, "pair": self.pair, "created": self.created}

    @classmethod
    def from_dict(cls, data: Mapping) -> "Calibration":
        if not isinstance(data, Mapping):
            raise CalibrationError("calibration must be a JSON object")
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise CalibrationError(f"calibration is missing {missing}")
        if data["schema"] != SCHEMA_VERSION:
            raise CalibrationError(f"calibration schema {data['schema']} is not the supported {SCHEMA_VERSION}")
        if any(key not in data["thresholds"] for key in THRESHOLD_KEYS):
            raise CalibrationError(f"thresholds need {list(THRESHOLD_KEYS)}")
        if any(key not in data["sentence"] for key in SENTENCE_KEYS):
            raise CalibrationError(f"sentence section needs {list(SENTENCE_KEYS)}")
        scorer = Scorer.from_dict(data["scorer"])
        quantiles = data["feature_quantiles"]
        for name in FEATURE_NAMES:
            if name not in quantiles or not {"human", "ai"} <= set(quantiles[name]):
                raise CalibrationError(f"feature_quantiles needs human and ai points for {name}")
        return cls(scorer, dict(data["thresholds"]), int(data["min_tokens"]), int(data["window_tokens"]),
                   int(data["max_tokens"]), dict(data["sentence"]), quantiles, data["accuracy"], data["pair"],
                   str(data["created"]))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise CalibrationError(f"{path} does not exist: run the calibration first") from error
        except json.JSONDecodeError as error:
            raise CalibrationError(f"{path} is not valid JSON: {error}") from error
        return cls.from_dict(raw)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
