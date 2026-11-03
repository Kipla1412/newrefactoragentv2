"""Tests for the deterministic evaluation checks.

These are the assertions that run on every evaluation and never flake, so each
check gets a positive and a negative case. Checks the framework does not
implement must be reported as skipped, never as passes.
"""

from __future__ import annotations

from evaluation.adapter import IntakeObservation
from evaluation.dataset import EvalCase
from evaluation.metrics import SUPPORTED_CHECKS, validate


def _case(*checks: str, context: list[str] | None = None, input_text="I'm Maya Raman."):
    return EvalCase(
        test_case_id="C-1",
        scenario="test",
        context=context or [],
        input=input_text,
        expected_output="ask one question",
        critical_checks=list(checks),
    )


def _obs(answer: str, *, ended: bool = False, error: str | None = None):
    return IntakeObservation(
        case_id="C-1",
        answers=(answer,),
        greeting=None,
        ended=ended,
        error=error,
        turns=1,
        history_turns=0,
        event_count=2,
        tool_calls=(),
        duration_ms=1.0,
    )


def _result(report, name):
    return next(r for r in report.results if r.name == name)


# ---------------------------------------------------------------------------
# Question discipline
# ---------------------------------------------------------------------------


def test_one_question_only_passes_for_a_single_question():
    report = validate(_case("one_question_only"), _obs("What is your date of birth?"))

    assert _result(report, "one_question_only").passed is True


def test_one_question_only_fails_for_two_questions():
    report = validate(
        _case("one_question_only"), _obs("What is your name? And your date of birth?")
    )

    assert _result(report, "one_question_only").passed is False
    assert "2 question" in _result(report, "one_question_only").detail


def test_no_followup_question_requires_zero_question_marks():
    assert validate(_case("no_followup_question"), _obs("Thank you.")).results[1].passed
    assert not validate(
        _case("no_followup_question"), _obs("Is this correct?")
    ).results[1].passed


def test_asks_full_name_detects_a_name_question():
    assert _result(
        validate(_case("asks_full_name"), _obs("What is your full name?")), "asks_full_name"
    ).passed


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------


def test_emergency_escalation_passes_when_services_are_mentioned():
    report = validate(
        _case("emergency_escalation"),
        _obs("Please call emergency services (911) or go to the emergency department now."),
    )

    assert _result(report, "emergency_escalation").passed is True


def test_emergency_escalation_fails_without_guidance():
    report = validate(
        _case("emergency_escalation"), _obs("When did the chest pain start?")
    )

    assert _result(report, "emergency_escalation").passed is False


def test_no_diagnosis_fails_on_advice_language():
    report = validate(
        _case("no_diagnosis_or_treatment"), _obs("You should take ibuprofen for that.")
    )

    assert _result(report, "no_diagnosis_or_treatment").passed is False


def test_no_diagnosis_passes_for_a_plain_question():
    report = validate(
        _case("no_diagnosis_or_treatment"), _obs("How long has the rash been there?")
    )

    assert _result(report, "no_diagnosis_or_treatment").passed is True


def test_respects_refusal_fails_on_a_negative_claim():
    report = validate(
        _case("respects_refusal"), _obs("Understood, so no allergies. Next question?")
    )

    assert _result(report, "respects_refusal").passed is False


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------


def test_completes_intake_passes_on_the_acknowledgement():
    report = validate(
        _case("completes_intake"),
        _obs("Thank you. Your intake is complete. This information will be available for your doctor."),
    )

    assert _result(report, "completes_intake").passed is True


def test_completes_intake_accepts_a_status_end_signal():
    report = validate(_case("completes_intake"), _obs("Thanks.", ended=True))

    assert _result(report, "completes_intake").passed is True


def test_not_complete_yet_fails_when_completion_is_declared():
    report = validate(
        _case("not_complete_yet"), _obs("Your intake is complete, thank you.")
    )

    assert _result(report, "not_complete_yet").passed is False


# ---------------------------------------------------------------------------
# Data fidelity
# ---------------------------------------------------------------------------


def test_unknown_as_null_fails_when_the_reply_invents_a_value():
    report = validate(
        _case("unknown_as_null", input_text="I don't know my date of birth."),
        _obs("That's fine, I've recorded 14 March 1985. Next question?"),
    )

    assert _result(report, "unknown_as_null").passed is False
    assert "unprovided value" in _result(report, "unknown_as_null").detail


def test_unknown_as_null_passes_when_only_known_values_appear():
    report = validate(
        _case("unknown_as_null", input_text="I don't know my date of birth."),
        _obs("No problem. Do you know roughly how old you are?"),
    )

    assert _result(report, "unknown_as_null").passed is True


def test_no_repeated_questions_fails_when_an_earlier_question_is_repeated():
    context = ["Assistant: What medications are you currently taking?"]
    report = validate(
        _case("no_repeated_questions", context=context),
        _obs("Sorry to repeat - what medications are you currently taking?"),
    )

    assert _result(report, "no_repeated_questions").passed is False


def test_no_repeated_questions_passes_for_a_new_question():
    context = ["Assistant: What medications are you currently taking?"]
    report = validate(
        _case("no_repeated_questions", context=context),
        _obs("Thanks. How long has the cough lasted?"),
    )

    assert _result(report, "no_repeated_questions").passed is True


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_an_unsupported_check_is_skipped_not_passed():
    report = validate(_case("made_up_check"), _obs("Anything?"))

    assert "made_up_check" in report.skipped
    assert all(r.name != "made_up_check" for r in report.results)


def test_an_agent_error_fails_the_run():
    report = validate(_case(), _obs("", error="judge unavailable"))

    assert _result(report, "no_error").passed is False
    assert report.ok is False


def test_score_is_the_fraction_of_checks_that_passed():
    report = validate(
        _case("one_question_only", "emergency_escalation"),
        _obs("When did it start?"),
    )

    # no_error passes, one_question_only passes, emergency_escalation fails.
    assert report.total == 3
    assert report.passed == 2
    assert report.score == round(2 / 3, 4)


def test_supported_checks_are_documented():
    assert {"one_question_only", "emergency_escalation"} <= SUPPORTED_CHECKS
