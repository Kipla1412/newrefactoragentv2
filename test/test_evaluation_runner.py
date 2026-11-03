"""Tests for the offline evaluation runner.

The adapter and the tracker are both injected, so these tests cover the
orchestration — per-case scoring, aggregation, the JSON report, and failure
handling — without touching a model or MLflow.
"""

from __future__ import annotations

import json

from evaluation.adapter import IntakeObservation
from evaluation.dataset import EvalCase, EvalDataset
from evaluation.runner import EvaluationReport, EvaluationRunner


class _StubAdapter:
    """Returns a scripted observation per case id."""

    def __init__(self, answers: dict[str, str] | None = None, fail: str | None = None):
        self._answers = answers or {}
        self._fail = fail
        self.seen: list[str] = []

    async def run_case(self, case: EvalCase) -> IntakeObservation:
        self.seen.append(case.id)
        error = "boom" if self._fail == case.id else None
        return IntakeObservation(
            case_id=case.id,
            answers=(self._answers.get(case.id, "What is your full name?"),),
            greeting=None,
            ended=False,
            error=error,
            turns=1,
            history_turns=0,
            event_count=3,
            tool_calls=(),
            duration_ms=2.0,
        )


class _StubTracker:
    def __init__(self):
        self.started: list[tuple[str, dict]] = []
        self.cases: list[tuple[str, dict]] = []
        self.summary: dict | None = None
        self.failures: list | None = None
        self.ended = 0

    def start(self, run_name, dataset_info):
        self.started.append((run_name, dataset_info))

    def log_case(self, case_id, scores):
        self.cases.append((case_id, scores))

    def log_summary(self, summary, failures):
        self.summary = summary
        self.failures = failures

    def end(self):
        self.ended += 1


def _dataset(*checks: str) -> EvalDataset:
    return EvalDataset(
        dataset_name="unit-dataset",
        test_cases=[
            EvalCase(
                test_case_id="C-1",
                scenario="opening",
                input="Hello.",
                expected_output="Ask for the name.",
                critical_checks=list(checks) or ["one_question_only"],
            ),
            EvalCase(
                test_case_id="C-2",
                scenario="refusal",
                input="I'd rather not say.",
                expected_output="Move on.",
                critical_checks=["one_question_only"],
            ),
        ]
    )


def _runner(config, adapter=None, tracker=None):
    return EvaluationRunner(
        config,
        adapter=adapter or _StubAdapter(),
        tracker=tracker or _StubTracker(),
        llm_metrics=False,
    )


# ---------------------------------------------------------------------------
# Single case and full dataset
# ---------------------------------------------------------------------------


def test_run_case_evaluates_only_that_case(config):
    adapter = _StubAdapter()

    report = _runner(config, adapter).run_case("C-2", _dataset())

    assert [case.case_id for case in report.cases] == ["C-2"]
    assert adapter.seen == ["C-2"]


def test_run_all_evaluates_every_case(config):
    adapter = _StubAdapter()

    report = _runner(config, adapter).run_all(_dataset())

    assert [case.case_id for case in report.cases] == ["C-1", "C-2"]
    assert adapter.seen == ["C-1", "C-2"]


def test_results_carry_check_outcomes(config):
    report = _runner(config).run_case("C-1", _dataset("one_question_only"))

    result = report.cases[0]
    names = {check["name"] for check in result.checks}
    assert {"no_error", "one_question_only"} <= names
    assert result.fully_passed is True


def test_a_failing_check_is_reported_but_does_not_abort_the_run(config):
    tracker = _StubTracker()
    adapter = _StubAdapter(answers={"C-2": "Is this right? And your name?"})

    report = _runner(config, adapter, tracker).run_all(_dataset("one_question_only"))

    assert len(report.cases) == 2
    failing = next(case for case in report.cases if case.case_id == "C-2")
    assert failing.fully_passed is False
    assert any(not check["passed"] for check in failing.checks)


def test_an_agent_error_is_reported_and_counted(config):
    adapter = _StubAdapter(fail="C-1")

    report = _runner(config, adapter).run_all(_dataset())

    errored = next(case for case in report.cases if case.case_id == "C-1")
    assert errored.ok is False
    assert errored.error == "boom"
    assert report.summary["errors"] == 1


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def test_summary_aggregates_cases_and_checks(config):
    report = _runner(config).run_all(_dataset())

    summary = report.summary
    assert summary["cases"] == 2
    assert summary["checks_total"] == 4  # no_error + one_question_only, twice
    assert summary["checks_passed"] == 4
    assert summary["cases_passed"] == 2
    assert summary["check_score"] == 1.0
    assert summary["duration_ms"] >= 0


def test_summary_lists_skipped_checks(config):
    report = _runner(config).run_all(_dataset("not_a_real_check"))

    assert "not_a_real_check" in report.summary["skipped_checks"]


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------


def test_tracking_receives_the_dataset_and_per_case_scores(config):
    tracker = _StubTracker()

    _runner(config, tracker=tracker).run_all(_dataset())

    assert tracker.started, "tracking was never started"
    assert tracker.started[0][1]["cases"] == 2
    assert [case_id for case_id, _ in tracker.cases] == ["C-1", "C-2"]
    assert tracker.summary is not None
    assert tracker.ended == 1


def test_tracking_receives_the_failure_list(config):
    tracker = _StubTracker()
    adapter = _StubAdapter(fail="C-1")

    _runner(config, adapter, tracker).run_all(_dataset())

    assert tracker.failures is not None
    assert any(failure["case_id"] == "C-1" for failure in tracker.failures)


def test_llm_metrics_are_reported_as_unavailable_without_deepeval(config):
    report = _runner(config).run_all(_dataset())

    # llm_metrics=False on the runner used by these tests.
    assert report.llm_metrics["status"] in {"disabled", "unavailable"}


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def test_report_is_json_serializable(config, tmp_path):
    report = _runner(config).run_all(_dataset())

    path = EvaluationRunner.save(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_name"] == "unit-dataset"
    assert payload["summary"]["cases"] == 2
    assert len(payload["cases"]) == 2
    assert payload["cases"][0]["checks"]


def test_report_dataclass_round_trips_through_to_dict(config):
    report = _runner(config).run_case("C-1", _dataset())

    assert isinstance(report, EvaluationReport)
    assert report.to_dict()["cases"][0]["case_id"] == "C-1"
