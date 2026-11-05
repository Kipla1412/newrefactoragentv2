"""Tests for the offline evaluation runner.

The case runner, the tracker and the repository are all injected, and the engine
is built from stub metrics, so these tests cover the orchestration — per-case
scoring, aggregation, the JSON report, and failure handling — without touching a
model, MLflow, the network or a dataset file.
"""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.core.engine import EvaluationEngine, MetricSelection
from evaluation.core.models import (
    EvaluationContext,
    MetricResult,
    MetricStatus,
    utc_now,
)
from evaluation.core.registry import MetricRegistry
from evaluation.datasets import EvalCase, EvalDataset, FileDatasetSource
from evaluation.reporting import EvaluationReport
from evaluation.runtime.offline import OfflineRunner


class _NoErrorMetric:
    """Stub metric: the agent ran without reporting an error."""

    name = "no_error"

    def measure(self, context: EvaluationContext) -> MetricResult:
        passed = context.error is None
        return MetricResult(
            metric_name=self.name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            passed=passed,
            reason="agent completed" if passed else (context.error or "agent failed"),
            status=MetricStatus.OK,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )


class _OneQuestionOnlyMetric:
    """Stub metric mirroring the intake rule: at most one question mark."""

    name = "one_question_only"

    def measure(self, context: EvaluationContext) -> MetricResult:
        count = context.answer.count("?")
        passed = count <= 1
        return MetricResult(
            metric_name=self.name,
            agent_name=context.agent_name,
            execution_id=context.execution_id,
            session_id=context.session_id,
            score=1.0 if passed else 0.0,
            threshold=1.0,
            passed=passed,
            reason=f"{count} question mark(s)",
            status=MetricStatus.OK,
            evaluation_mode=context.mode,
            timestamp=utc_now(),
        )


#: Stub metric factories, keyed by the names the engine selection configures.
_STUB_METRICS = {
    "no_error": _NoErrorMetric,
    "one_question_only": _OneQuestionOnlyMetric,
}

#: Metric names the engine is configured with unless a test overrides them.
DEFAULT_METRICS = ("no_error", "one_question_only")


class _StubAdapter:
    """Returns a scripted evaluation context per case id.

    Implements the ``CaseRunner`` port, so the runner drives a fake agent and
    none of these tests build a model or a session.
    """

    agent_name = "intake"

    def __init__(self, answers: dict[str, str] | None = None, fail: str | None = None):
        self._answers = answers or {}
        self._fail = fail
        self.seen: list[str] = []

    async def run_case(self, case: EvalCase) -> EvaluationContext:
        self.seen.append(case.id)
        answer = self._answers.get(case.id, "What is your full name?")
        return EvaluationContext(
            agent_name=self.agent_name,
            input=case.input,
            output=answer,
            context=tuple(case.context),
            ended=False,
            error="boom" if self._fail == case.id else None,
            execution_id=case.id,
            tool_calls=(),
            duration_ms=2.0,
            metadata={"answer": answer, "history_turns": 0, "event_count": 3},
        )


class _StubTracker:
    """Records what the runner reports (the ``TrackerPort``)."""

    def __init__(self):
        self.started: list[tuple[str, dict]] = []
        self.cases: list[tuple[str, object]] = []
        self.summary: dict | None = None
        self.failures: list | None = None
        self.ended = 0

    @property
    def enabled(self) -> bool:
        return True

    def start(self, run_name, tags):
        self.started.append((run_name, tags))

    def log_execution(self, result, **tags):
        self.cases.append((tags.get("case"), result))

    def log_summary(self, summary, failures=None):
        self.summary = summary
        self.failures = failures

    def end(self):
        self.ended += 1


class _StubRepository:
    """Writes the report payload to the requested path (the repository port)."""

    def __init__(self):
        self.saved: list[dict] = []

    def save(self, payload: dict, name: str | None = None) -> Path:
        self.saved.append(payload)
        target = Path(name) if name else Path("evaluation_report.json")
        target.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target


def _engine(metrics: tuple[str, ...] = DEFAULT_METRICS) -> EvaluationEngine:
    """Build an engine over the stub metrics.

    Unknown names in ``metrics`` are left unregistered on purpose, so the engine
    reports them as skipped — the same way a dataset or configuration naming a
    metric the framework does not implement is reported.
    """
    registry = MetricRegistry()
    for name, factory in _STUB_METRICS.items():
        registry.register_factory(name, factory)
    return EvaluationEngine(registry, MetricSelection(common=tuple(metrics)))


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
        ],
    )


def _runner(config, adapter=None, tracker=None, *, metrics=DEFAULT_METRICS, engine=None):
    return OfflineRunner(
        config,
        engine=engine if engine is not None else _engine(metrics),
        case_runner=adapter or _StubAdapter(),
        tracker=tracker or _StubTracker(),
        repository=_StubRepository(),
        dataset_source=FileDatasetSource(),
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
    runner = _runner(config, metrics=(*DEFAULT_METRICS, "not_a_real_check"))

    report = runner.run_all(_dataset("not_a_real_check"))

    # A metric the engine cannot resolve is reported as skipped, with its name.
    assert any(
        check.startswith("not_a_real_check:")
        for check in report.summary["skipped_checks"]
    )


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


def test_llm_metrics_are_reported_as_disabled_when_none_configured(config):
    report = _runner(config).run_all(_dataset())

    # No LLM-judged metric is configured on the engine these tests inject.
    assert report.llm_metrics["status"] in {"disabled", "unavailable"}


# ---------------------------------------------------------------------------
# Report serialization
# ---------------------------------------------------------------------------


def test_report_is_json_serializable(config, tmp_path):
    runner = _runner(config)
    report = runner.run_all(_dataset())

    path = runner.save(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["dataset_name"] == "unit-dataset"
    assert payload["summary"]["cases"] == 2
    assert len(payload["cases"]) == 2
    assert payload["cases"][0]["checks"]


def test_report_dataclass_round_trips_through_to_dict(config):
    report = _runner(config).run_case("C-1", _dataset())

    assert isinstance(report, EvaluationReport)
    assert report.to_dict()["cases"][0]["case_id"] == "C-1"
