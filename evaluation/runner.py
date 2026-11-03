"""Offline evaluation runner for the intake agent.

Pipeline for each case:

.. code-block:: text

    dataset ─► IntakeAgentAdapter ─► IntakeObservation
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
            deterministic checks    DeepEval (optional)   MLflow (optional)
                    │                    │                    │
                    └────────────────────┴────────────────────┘
                                         ▼
                                   EvaluationReport ─► JSON

Execution and scoring are separate phases on purpose: cases run inside an event
loop, then scoring runs outside it, so DeepEval's synchronous judge can drive our
asynchronous ``LLMClient`` without nesting loops.

Nothing here re-implements execution, LLM access, or observability — the adapter,
``LLMClient``, and ``get_mlflow_tracker`` are all existing framework pieces.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation import metrics as metrics_module
from evaluation.adapter import IntakeAgentAdapter, IntakeObservation
from evaluation.dataset import EvalCase, EvalDataset, load_dataset
from evaluation.judge import build_judge, deepeval_installed
from evaluation.metrics import validate
from evaluation.tracking import EvaluationTracker
from utils.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_REPORT_PATH = Path("evaluation_report.json")


@dataclass(frozen=True)
class CaseResult:
    """Scored result for one case.

    Attributes:
        case_id: Dataset case identifier.
        scenario: Human label from the dataset.
        answer: The agent's reply to the evaluated turn.
        greeting: One-time greeting, when a fresh session produced one.
        ok: Whether the agent ran without error.
        error: Error text, when it did not.
        history_turns: Conversation turns replayed before the evaluated turn.
        duration_ms: Case wall-clock time.
        checks: Deterministic check outcomes.
        skipped_checks: Checks requested but not implemented.
        checks_passed: Executed checks that passed.
        checks_total: Executed checks.
        check_score: Passed / total.
        llm_scores: DeepEval metric scores, when available.
    """

    case_id: str
    scenario: str
    answer: str
    greeting: str | None
    ok: bool
    error: str | None
    history_turns: int
    duration_ms: float
    checks: tuple[dict[str, Any], ...]
    skipped_checks: tuple[str, ...]
    checks_passed: int
    checks_total: int
    check_score: float
    llm_scores: dict[str, float] = field(default_factory=dict)
    llm_reasons: dict[str, str] = field(default_factory=dict)
    tool_calls: tuple[str, ...] = ()
    event_count: int = 0

    @property
    def fully_passed(self) -> bool:
        """Whether the case ran cleanly and every executed check passed."""
        return self.ok and self.check_score >= 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the result."""
        return {
            "case_id": self.case_id,
            "scenario": self.scenario,
            "answer": self.answer,
            "greeting": self.greeting,
            "ok": self.ok,
            "error": self.error,
            "history_turns": self.history_turns,
            "duration_ms": self.duration_ms,
            "event_count": self.event_count,
            "tool_calls": list(self.tool_calls),
            "checks": list(self.checks),
            "skipped_checks": list(self.skipped_checks),
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
            "check_score": self.check_score,
            "llm_scores": self.llm_scores,
            "llm_reasons": self.llm_reasons,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """A complete evaluation run.

    Attributes:
        dataset_name: Dataset identifier.
        agent: Agent the dataset targets.
        started_at: UTC timestamp when the run began.
        duration_ms: Total run time.
        cases: Per-case results.
        llm_metrics: Status of the optional DeepEval metrics.
    """

    dataset_name: str
    agent: str
    started_at: str
    duration_ms: float
    cases: tuple[CaseResult, ...]
    llm_metrics: dict[str, Any]

    @property
    def summary(self) -> dict[str, Any]:
        """Aggregate scores for the whole run."""
        checks_passed = sum(case.checks_passed for case in self.cases)
        checks_total = sum(case.checks_total for case in self.cases)
        return {
            "cases": len(self.cases),
            "cases_passed": sum(1 for case in self.cases if case.fully_passed),
            "cases_failed": sum(1 for case in self.cases if not case.fully_passed),
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "check_score": round(checks_passed / checks_total, 4) if checks_total else 0.0,
            "errors": sum(1 for case in self.cases if not case.ok),
            "skipped_checks": sorted({
                name for case in self.cases for name in case.skipped_checks
            }),
            "duration_ms": round(self.duration_ms, 1),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the report."""
        return {
            "dataset_name": self.dataset_name,
            "agent": self.agent,
            "started_at": self.started_at,
            "duration_ms": round(self.duration_ms, 1),
            "summary": self.summary,
            "llm_metrics": self.llm_metrics,
            "cases": [case.to_dict() for case in self.cases],
        }


class EvaluationRunner:
    """Run dataset cases and score them.

    Args:
        config: Runtime configuration.
        adapter: Optional adapter override (tests inject a stubbed one).
        tracker: Optional tracker override; None builds the MLflow one.
        judge: Optional DeepEval judge; None is resolved lazily.
        llm_metrics: Set False to skip DeepEval entirely.
    """

    def __init__(
        self,
        config,
        *,
        adapter: IntakeAgentAdapter | None = None,
        tracker: EvaluationTracker | None = None,
        judge: Any | None = None,
        llm_metrics: bool = True,
        judge_model: str = "clinical_model",
    ) -> None:
        self._config = config
        self._adapter = adapter or IntakeAgentAdapter(config)
        self._tracker = tracker if tracker is not None else EvaluationTracker(config)
        self._judge = judge
        self._judge_resolved = judge is not None
        self._llm_metrics = llm_metrics
        self._judge_model = judge_model
        self._judge_session = None

    # Public API

    def run_case(self, case_id: str, dataset: EvalDataset | None = None) -> EvaluationReport:
        """Evaluate a single case.

        Args:
            case_id: The dataset case identifier.
            dataset: Optional dataset override.

        Returns:
            A report containing only that case.
        """
        resolved = dataset or load_dataset()
        return self._run([resolved.by_id(case_id)], resolved)

    def run_all(self, dataset: EvalDataset | None = None) -> EvaluationReport:
        """Evaluate every case in the dataset.

        Args:
            dataset: Optional dataset override.

        Returns:
            The full report.
        """
        resolved = dataset or load_dataset()
        return self._run(resolved.test_cases, resolved)

    @staticmethod
    def save(report: EvaluationReport, path: str | Path | None = None) -> Path:
        """Write the report as JSON.

        Args:
            report: The report to write.
            path: Destination path. Defaults to ``evaluation_report.json``.

        Returns:
            The path written.
        """
        target = Path(path) if path is not None else DEFAULT_REPORT_PATH
        target.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return target

    # Internals

    def _run(self, cases: list[EvalCase], dataset: EvalDataset) -> EvaluationReport:
        started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        began = time.perf_counter()
        judge = self._resolve_judge()

        self._tracker.start(
            f"intake-eval-{started_at}",
            {
                "name": dataset.dataset_name,
                "version": dataset.version,
                "agent": dataset.agent,
                "cases": len(cases),
            },
        )

        results: list[CaseResult] = []
        report: EvaluationReport | None = None

        try:
            for case in cases:
                observation = asyncio.run(self._adapter.run_case(case))
                validation = validate(case, observation)
                llm_scores, llm_reasons = self._score_llm(case, observation, judge)

                self._tracker.log_case(
                    case.id,
                    {
                        "check_score": validation.score,
                        "ok": 1.0 if observation.ok else 0.0,
                        **llm_scores,
                    },
                )

                results.append(
                    self._to_result(
                        case, observation, validation, llm_scores, llm_reasons
                    )
                )

            report = EvaluationReport(
                dataset_name=dataset.dataset_name,
                agent=dataset.agent,
                started_at=started_at,
                duration_ms=(time.perf_counter() - began) * 1000,
                cases=tuple(results),
                llm_metrics=self._llm_status(judge),
            )

            # Must happen while the run is still open: the aggregate metrics are
            # logged onto it. Ending the run first silently dropped them.
            self._tracker.log_summary(
                report.summary,
                [case.to_dict() for case in results if not case.fully_passed],
            )

            logger.info("Evaluation finished", extra={"fields": report.summary})

            return report

        finally:
            self._tracker.end()
            self._close_judge_session()

    @staticmethod
    def _to_result(case, observation, validation, llm_scores, llm_reasons) -> CaseResult:
        return CaseResult(
            case_id=case.id,
            scenario=case.scenario,
            answer=observation.answer,
            greeting=observation.greeting,
            ok=observation.ok,
            error=observation.error,
            history_turns=observation.history_turns,
            duration_ms=observation.duration_ms,
            checks=tuple(
                {"name": r.name, "passed": r.passed, "detail": r.detail}
                for r in validation.results
            ),
            skipped_checks=validation.skipped,
            checks_passed=validation.passed,
            checks_total=validation.total,
            check_score=validation.score,
            llm_scores=llm_scores,
            llm_reasons=llm_reasons,
            tool_calls=observation.tool_calls,
            event_count=observation.event_count,
        )

    def _resolve_judge(self):
        if self._judge_resolved:
            return self._judge
        self._judge_resolved = True

        if not self._llm_metrics or not deepeval_installed():
            return None

        try:
            from agent.session import Session

            self._judge_session = Session(self._config)
            self._judge = build_judge(
                self._judge_session.client, model=self._judge_model
            )
        except Exception as exc:  # noqa: BLE001 - metrics are optional
            logger.warning("Failed to build the DeepEval judge: %s", exc)
            self._judge = None

        return self._judge

    def _close_judge_session(self) -> None:
        if self._judge_session is None:
            return
        try:
            asyncio.run(self._judge_session.client.close())
        except Exception as exc:  # noqa: BLE001 - cleanup must not raise
            logger.debug("Failed to close the judge session: %s", exc)
        finally:
            self._judge_session = None

    def _llm_status(self, judge) -> dict[str, Any]:
        if judge is not None:
            return {"status": "ok", "metrics": ["AnswerRelevancyMetric", "TaskCompletionMetric"]}
        if not self._llm_metrics:
            return {"status": "disabled", "reason": "llm_metrics=False"}
        if not deepeval_installed():
            return {
                "status": "unavailable",
                "reason": "deepeval is not installed (uv pip install deepeval)",
            }
        return {"status": "unavailable", "reason": "judge could not be built"}

    def _score_llm(
        self, case: EvalCase, observation: IntakeObservation, judge
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Score a case with DeepEval, returning ``(scores, reasons)``."""
        if judge is None or not observation.answer:
            return {}, {}

        try:
            from deepeval.test_case import LLMTestCase
        except ImportError:
            return {}, {}

        test_case = LLMTestCase(
            input=case.input,
            actual_output=observation.answer,
            expected_output=case.expected_output,
        )

        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}

        for metric in metrics_module.build_metrics(judge, case.task):
            name = metric.__class__.__name__
            try:
                metric.measure(test_case)
                scores[name] = round(float(metric.score), 4)
                reasons[name] = str(getattr(metric, "reason", "") or "")[:400]
            except Exception as exc:  # noqa: BLE001 - one metric must not abort a run
                logger.warning("Metric %s failed for %s: %s", name, case.id, exc)
                reasons[name] = f"failed: {exc}"[:400]

        return scores, reasons
