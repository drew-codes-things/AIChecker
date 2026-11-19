from collections.abc import Mapping

from .calibration import Calibration
from .metrics import percentile_rank

NOT_ENOUGH_TEXT = "not_enough_text"
LITTLE_SIGN = "little_sign"
SOME_SIGN = "some_sign"
STRONG_SIGN = "strong_sign"
LOWER_MEANS_AI = frozenset({"binoculars"})
STRONG_LEAN = 0.95
MILD_LEAN = 0.80
HUMAN_LEAN = 0.35
SIGNAL_TITLES = {
    "binoculars": "Predictability (Binoculars score)",
    "fast_detect": "Probability curvature (Fast-DetectGPT style)",
    "gltr_top1": "Words that were the model's first guess",
    "gltr_top10": "Words inside the model's top ten guesses",
    "gltr_tail": "Words the model found very surprising",
}
SIGNAL_FORMAT = {
    "binoculars": "{:.2f}",
    "fast_detect": "{:+.2f}",
    "gltr_top1": "{:.0%}",
    "gltr_top10": "{:.0%}",
    "gltr_tail": "{:.1%}",
}
REASON_FEATURES = ("binoculars", "fast_detect", "gltr_top10", "gltr_tail")
BAND_HEADLINES = {
    NOT_ENOUGH_TEXT: "Too short to judge",
    LITTLE_SIGN: "Little sign of AI writing",
    SOME_SIGN: "Some signs of AI writing",
    STRONG_SIGN: "Strong signs of AI writing",
}
PRIOR_NOTE = ("The percentage assumes AI and human writing are equally likely beforehand. "
              "If most of what you check is human, treat it as higher than it really is.")


def ai_likeness(calibration: Calibration, feature: str, value: float) -> float:
    human_points = calibration.feature_quantiles[feature]["human"]
    rank = percentile_rank(human_points, value)
    return 1.0 - rank if feature in LOWER_MEANS_AI else rank


def _lean_for(likeness: float) -> str:
    if likeness >= MILD_LEAN:
        return "ai"
    if likeness <= HUMAN_LEAN:
        return "human"
    return "neutral"


def _reason_text(feature: str, value: float, likeness: float, lean: str) -> str:
    shown = SIGNAL_FORMAT[feature].format(value)
    share = round(likeness * 100)
    if lean == "ai":
        comparison = f"more machine-like than {share}% of the human writing we tested"
    elif lean == "human":
        comparison = f"more human-like than {100 - share}% of the human writing we tested"
    else:
        comparison = "in the ordinary range for human writing"
    return f"{SIGNAL_TITLES[feature]}: {shown}, {comparison}."


def build_reasons(features: Mapping[str, float], calibration: Calibration) -> list[dict]:
    reasons = []
    for feature in REASON_FEATURES:
        likeness = ai_likeness(calibration, feature, features[feature])
        lean = _lean_for(likeness)
        reasons.append({"signal": feature, "leans": lean, "likeness": round(likeness, 3),
                        "text": _reason_text(feature, features[feature], likeness, lean)})
    return sorted(reasons, key=lambda reason: -reason["likeness"])


def band_for(logit: float, calibration: Calibration) -> str:
    if logit > calibration.thresholds["fpr1"]:
        return STRONG_SIGN
    if logit > calibration.thresholds["fpr5"]:
        return SOME_SIGN
    return LITTLE_SIGN


def assess(features: Mapping[str, float], token_count: int, calibration: Calibration) -> dict:
    if token_count < calibration.min_tokens:
        return {"band": NOT_ENOUGH_TEXT, "headline": BAND_HEADLINES[NOT_ENOUGH_TEXT], "likelihood": None,
                "reasons": [], "tokens": token_count,
                "notes": [f"Need at least {calibration.min_tokens} tokens (about {round(calibration.min_tokens * 0.75)} words) "
                          "for a reading that means anything."]}
    logit = calibration.scorer.logit(features)
    band = band_for(logit, calibration)
    return {"band": band, "headline": BAND_HEADLINES[band],
            "likelihood": round(calibration.scorer.probability(features), 3),
            "reasons": build_reasons(features, calibration), "tokens": token_count, "notes": [PRIOR_NOTE]}


def sentence_level(score: float, token_count: int, calibration: Calibration) -> str:
    if token_count < calibration.sentence["min_tokens"]:
        return "unscored"
    if score < calibration.sentence["strong"]:
        return "strong"
    if score < calibration.sentence["some"]:
        return "some"
    return "none"
