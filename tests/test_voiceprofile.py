import pytest

from aichecker.voiceprofile import ProfileError, build_profile, guidance_lines

CASUAL = ("I don't really like long meetings. We always run over, and I end up checking my phone. "
          "Why do we do this? It's just habit, I think. Anyway, I'd rather send a short note and get on with it. "
          "My team's fine with that, mostly; they've said so. Honestly, it's easier. We tried a new rota last week "
          "and it didn't help much, so I stopped bothering. You can't fix everything with a calendar invite, can you? "
          "I'm happy to be wrong about this, but I doubt I am. That's my view and I'm sticking to it for now.")
FORMAL = ("The committee reviewed the proposal in detail. It concluded that the budget was adequate. "
          "Further consultation with stakeholders was recommended by the chair. The timetable remains under review "
          "pending the outcome of that consultation. Members noted the risks identified in the earlier assessment "
          "and agreed that mitigation should be documented before approval is granted by the board of trustees. "
          "The secretary will circulate the minutes within a fortnight, and the next meeting is scheduled for spring. "
          "No further business was raised, and the session closed shortly afterwards without any dissent recorded.")


def test_a_casual_voice_scores_high_on_contractions_and_first_person():
    profile = build_profile([CASUAL])
    assert profile["contractions_per_100_words"] > 5
    assert profile["first_person_per_100_words"] > 5
    assert profile["questions_per_100_sentences"] > 0


def test_a_formal_voice_scores_low_on_contractions_and_first_person():
    profile = build_profile([FORMAL])
    assert profile["contractions_per_100_words"] == 0
    assert profile["first_person_per_100_words"] == 0
    assert profile["average_sentence_words"] > build_profile([CASUAL])["average_sentence_words"]


def test_guidance_describes_the_measurements_in_words():
    casual = " ".join(guidance_lines(build_profile([CASUAL])))
    formal = " ".join(guidance_lines(build_profile([FORMAL])))
    assert "use them freely" in casual and "first person" in casual
    assert "use few or none" in formal and "Does not use dashes" in formal


def test_too_little_writing_is_refused_with_the_numbers():
    with pytest.raises(ProfileError, match="at least 80 words"):
        build_profile(["Just a few words here."])


def test_blank_samples_are_ignored_and_curly_apostrophes_count_as_contractions():
    profile = build_profile(["", "   ", CASUAL.replace("'", "’")])
    assert profile["contractions_per_100_words"] > 5
