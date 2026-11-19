from aichecker.report import (LITTLE_SIGN, NOT_ENOUGH_TEXT, SOME_SIGN, STRONG_SIGN, ai_likeness, assess, band_for,
                              sentence_level)
from support import AI_LIKE, HUMAN_LIKE, make_calibration


def test_ai_likeness_flips_direction_for_binoculars_only():
    calibration = make_calibration()
    assert ai_likeness(calibration, "binoculars", 0.5) == 1.0
    assert ai_likeness(calibration, "binoculars", 1.5) == 0.0
    assert ai_likeness(calibration, "fast_detect", 9.0) == 1.0
    assert ai_likeness(calibration, "fast_detect", -9.0) == 0.0


def test_human_like_features_give_little_sign_and_a_low_likelihood():
    result = assess(HUMAN_LIKE, 300, make_calibration())
    assert result["band"] == LITTLE_SIGN
    assert result["likelihood"] < 0.5
    assert all(reason["leans"] != "ai" for reason in result["reasons"])


def test_ai_like_features_give_strong_sign_with_reasons_naming_the_comparison():
    result = assess(AI_LIKE, 300, make_calibration())
    assert result["band"] == STRONG_SIGN
    assert result["likelihood"] > 0.9
    assert result["reasons"][0]["leans"] == "ai"
    assert "more machine-like than" in result["reasons"][0]["text"]
    assert result["notes"]


def test_band_edges_follow_the_two_false_positive_thresholds():
    calibration = make_calibration()
    assert band_for(2.01, calibration) == STRONG_SIGN
    assert band_for(2.0, calibration) == SOME_SIGN
    assert band_for(1.01, calibration) == SOME_SIGN
    assert band_for(1.0, calibration) == LITTLE_SIGN


def test_short_text_gets_no_likelihood_and_says_why():
    result = assess(AI_LIKE, 10, make_calibration())
    assert result["band"] == NOT_ENOUGH_TEXT
    assert result["likelihood"] is None
    assert "48 tokens" in result["notes"][0]


def test_no_band_text_claims_certainty():
    calibration = make_calibration()
    for features in (HUMAN_LIKE, AI_LIKE):
        result = assess(features, 300, calibration)
        spoken = (result["headline"] + " " + " ".join(reason["text"] for reason in result["reasons"])).lower()
        assert "is ai" not in spoken and "was written by ai" not in spoken and "verdict" not in spoken


def test_sentence_levels_use_lower_is_more_ai_and_skip_short_sentences():
    calibration = make_calibration()
    assert sentence_level(0.6, 20, calibration) == "strong"
    assert sentence_level(0.8, 20, calibration) == "some"
    assert sentence_level(0.95, 20, calibration) == "none"
    assert sentence_level(0.2, 3, calibration) == "unscored"
