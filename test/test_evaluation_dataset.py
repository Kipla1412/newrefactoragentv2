"""Tests for the evaluation dataset model and loader.

The dataset expresses cases as "given this conversation, what is the next
reply?", so these tests cover the two things that makes load-bearing: parsing
``context`` back into turns, and failing loudly on malformed input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    EvalCase,
    EvalDataset,
    load_dataset,
)

#: Checks the bundled dataset must keep exercising.
KEY_CHECKS = {
    "one_question_only",
    "no_hallucination",
    "emergency_escalation",
    "unknown_as_null",
    "respects_refusal",
    "no_diagnosis_or_treatment",
}


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _case(**overrides) -> dict:
    base = {
        "test_case_id": "CASE_1",
        "scenario": "Normal opening",
        "context": [],
        "input": "Hello.",
        "expected_output": "Ask one question.",
        "expected_behaviour": ["Ask only one question"],
        "critical_checks": ["one_question_only"],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The bundled dataset
# ---------------------------------------------------------------------------


def test_bundled_dataset_loads():
    dataset = load_dataset()

    assert dataset.agent
    assert dataset.version


def test_bundled_dataset_has_at_least_ten_cases():
    assert len(load_dataset().test_cases) >= 10


def test_bundled_dataset_exercises_the_critical_checks():
    present = load_dataset().critical_checks()

    assert KEY_CHECKS.issubset(present), KEY_CHECKS - present


def test_every_case_has_a_scenario_input_and_expectation():
    for case in load_dataset().test_cases:
        assert case.scenario.strip(), f"{case.id} has no scenario"
        assert case.input.strip(), f"{case.id} has no input"
        assert case.expected_output.strip(), f"{case.id} has no expected output"


def test_every_case_expects_at_least_one_critical_check():
    for case in load_dataset().test_cases:
        assert case.critical_checks, f"{case.id} declares no critical checks"


def test_every_case_produces_exactly_one_prompt():
    for case in load_dataset().test_cases:
        assert case.prompts == [case.input]


def test_default_dataset_path_is_bundled():
    assert DEFAULT_DATASET_PATH.name == "intake_dataset.json"
    assert DEFAULT_DATASET_PATH.is_file()


def test_by_id_returns_the_case():
    dataset = load_dataset()
    first = dataset.test_cases[0]

    assert dataset.by_id(first.id) is first


def test_by_id_raises_for_an_unknown_case():
    with pytest.raises(DatasetError) as exc:
        load_dataset().by_id("NOPE_999")

    assert "NOPE_999" in str(exc.value)


def test_bundled_dataset_reports_unreplayable_contexts():
    """Prose state descriptions cannot be replayed as turns; they are listed."""
    dataset = load_dataset()

    for case_id in dataset.not_replayable():
        assert dataset.by_id(case_id).context


# ---------------------------------------------------------------------------
# Context parsing
# ---------------------------------------------------------------------------


def test_conversation_parses_assistant_and_patient_lines():
    case = EvalCase(**_case(context=[
        "Assistant: What's your full name?",
        "Patient: Maya Raman.",
    ]))

    # "Patient" maps to the ContextManager's user role.
    assert case.conversation == [
        ("assistant", "What's your full name?"),
        ("user", "Maya Raman."),
    ]


def test_conversation_accepts_agent_and_user_prefixes():
    case = EvalCase(**_case(context=["Agent: Hi there.", "User: Hello."]))

    assert case.conversation == [("assistant", "Hi there."), ("user", "Hello.")]


def test_prose_context_is_skipped_and_marked_not_replayable():
    case = EvalCase(**_case(context=[
        "The agent has collected all required categories.",
    ]))

    assert case.conversation == []
    assert case.is_replayable is False


def test_empty_context_is_replayable():
    case = EvalCase(**_case(context=[]))

    assert case.conversation == []
    assert case.is_replayable is True


def test_context_is_case_insensitive_for_prefixes():
    case = EvalCase(**_case(context=["PATIENT: Yes."]))

    assert case.conversation == [("user", "Yes.")]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _payload(**overrides) -> dict:
    base = {"dataset_name": "test", "version": "1.0", "agent": "intake",
            "test_cases": [_case()]}
    base.update(overrides)
    return base


def test_rejects_duplicate_ids(tmp_path):
    path = _write(tmp_path, _payload(test_cases=[_case(), _case()]))

    with pytest.raises(DatasetError) as exc:
        load_dataset(path)

    assert "duplicate" in str(exc.value).lower()


def test_rejects_an_empty_dataset(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path, _payload(test_cases=[])))


def test_rejects_an_empty_input(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path, _payload(test_cases=[_case(input="  ")])))


def test_rejects_a_missing_test_case_id(tmp_path):
    payload = _payload()
    payload["test_cases"] = [{"scenario": "x", "input": "y", "expected_output": "z"}]

    with pytest.raises(DatasetError):
        load_dataset(_write(tmp_path, payload))


def test_missing_file_raises_a_dataset_error(tmp_path):
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "absent.json")


def test_invalid_json_raises_a_dataset_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)


def test_non_object_json_raises_a_dataset_error(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(DatasetError):
        load_dataset(path)


def test_validation_errors_list_the_problems(tmp_path):
    payload = _payload(test_cases=[_case(test_case_id="")])

    with pytest.raises(DatasetError) as exc:
        load_dataset(_write(tmp_path, payload))

    assert "problem" in str(exc.value)


def test_valid_dataset_builds_from_models():
    dataset = EvalDataset(test_cases=[EvalCase(**_case())])

    assert dataset.critical_checks() == {"one_question_only"}
