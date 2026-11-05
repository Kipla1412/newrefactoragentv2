"""Offline evaluation — orchestrating golden-dataset runs.

Pipeline:

.. code-block:: text

    DatasetSource ─► CaseRunner ─► EvaluationContext
                                         │
                                         ▼
                                  EvaluationEngine ─► EvaluationResult
                                         │                    │
                                         ▼                    ▼
                                    Tracker            EvaluationReport
                                  (MLflow)              ─► Repository ─► JSON

This runner is deliberately thin: it resolves a dataset, drives a
:class:`~evaluation.ports.case_runner.CaseRunner`, hands each context to the shared
engine, and aggregates. It performs **no scoring of its own** — the same engine
and the same metrics serve real-time evaluation, which is what stops the two modes
from drifting apart.

Execution and scoring stay separate: a case runs inside an event loop, then the
engine's synchronous scoring runs outside it, so a synchronous judge can drive the
framework's asynchronous LLM client without nesting loops.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from evaluation.adapters.agent_case_runner import AgentCaseRunner
from evaluation.adapters.json_repository import JsonEvaluationRepository
from evaluation.adapters.mlflow_tracker import EvaluationTracker
from evaluation.core.dataset import EvalCase, EvalDataset
from evaluation.core.engine import EvaluationEngine
from evaluation.core.models import EvaluationResult, MetricStatus
from evaluation.datasets.loader import FileDatasetSource
from evaluation.reporting.models import CaseResult, EvaluationReport
from utils.logging_config import get_logger

logger = get_logger(__name__)


class OfflineRunner:
    """Run dataset cases through an agent and score them with the shared engine.

    All collaborators are injected (Dependency Inversion), so a test can run the
    whole pipeline with stub ports and no agent, no MLflow and no network.

    Args:
        config: Runtime configuration.
        engine: Evaluation engine. Required: build one with
            :func:`evaluation.composition.build_offline_runner`, so this module
            never reaches back to the composition root.
        case_runner: Executes cases. Defaults to an ``AgentCaseRunner``.
        dataset_source: Where datasets come from. Defaults to the bundled files.
        tracker: Where results are recorded. Defaults to the MLflow tracker.
        repository: Where reports are saved. Defaults to JSON on disk.
        agent: Agent name used for metric configuration and context tagging.
        dataset_version: Dataset version; ``None`` loads the default.
    """

    def __init__(
        self,
        config,
        *,
        engine: EvaluationEngine,
        case_runner: AgentCaseRunner | None = None,
        dataset_source: FileDatasetSource | None = None,
        tracker: EvaluationTracker | None = None,
        repository: JsonEvaluationRepository | None = None,
        agent: str = "intake",
        dataset_version: str | None = None,
    ) -> None:
        self._config = config
        self._agent = agent
        self._dataset_version = dataset_version
        self._engine = engine
        self._case_runner = (
            case_runner if case_runner is not None else AgentCaseRunner(config, agent_name=agent)
        )
        self._datasets = dataset_source if dataset_source is not None else FileDatasetSource()
        self._tracker = (
            tracker if tracker is not None else EvaluationTracker(config)
        )
        self._repository = (
            repository if repository is not None else JsonEvaluationRepository()
        )

    # Introspection

    @property
    def engine(self) -> EvaluationEngine:
        """The engine in use."""
        return self._engine

    @property
    def agent_name(self) -> str:
        """Agent name used for metric configuration."""
        return getattr(self._case_runner, "agent_name", self._agent)

    # Public API

    def run_case(self, case_id: str, dataset: EvalDataset | None = None) -> EvaluationReport:
        """Evaluate a single case.

        Args:
            case_id: The dataset case identifier.
            dataset: Optional dataset override.

        Returns:
            A report containing only that case.
        """
        resolved = dataset or self._datasets.load(self._agent, self._dataset_version)
        return self._run([resolved.by_id(case_id)], resolved)

    def run_all(self, dataset: EvalDataset | None = None) -> EvaluationReport:
        """Evaluate every case in the dataset.

        Args:
            dataset: Optional dataset override.

        Returns:
            The full report.
        """
        resolved = dataset or self._datasets.load(self._agent, self._dataset_version)
        return self._run(resolved.test_cases, resolved)

    def save(self, report: EvaluationReport, path: str | Path | None = None) -> Path:
        """Persist the report through the repository.

        Args:
            report: The report to write.
            path: Destination path or name. Defaults to the repository's default.

        Returns:
            The path written.
        """
        return self._repository.save(report.to_dict(), str(path) if path else None)

    # Internals

    def _run(self, cases: list[EvalCase], dataset: EvalDataset) -> EvaluationReport:
        started_at = EvaluationReport.start_run()
        began = time.perf_counter()
        agent_name = self.agent_name
        llm_metric_names = self._llm_metric_names(agent_name)

        self._tracker.start(
            f"{agent_name}-eval-{started_at}",
            {
                "name": dataset.dataset_name,
                "version": dataset.version,
                "agent": agent_name,
                "cases": len(cases),
                "metrics": ",".join(self._engine.metric_names(agent_name)),
            },
        )

        results: list[CaseResult] = []
        evaluations: list[EvaluationResult] = []

        try:
            for case in cases:
                # Phase 1: execute (inside an event loop).
                context = asyncio.run(self._case_runner.run_case(case))
                # Phase 2: score (synchronous engine).
                evaluation = self._engine.evaluate(context)

                evaluations.append(evaluation)
                self._tracker.log_execution(evaluation, case=case.id, scenario=case.scenario)

                results.append(
                    CaseResult.from_evaluation(
                        case.id,
                        case.scenario,
                        context,
                        evaluation,
                        llm_metric_names=llm_metric_names,
                    )
                )

            report = EvaluationReport(
                dataset_name=dataset.dataset_name,
                agent=agent_name,
                started_at=started_at,
                duration_ms=(time.perf_counter() - began) * 1000,
                cases=tuple(results),
                llm_metrics=self._llm_status(evaluations, llm_metric_names),
                dataset_version=dataset.version,
            )

            # Must happen while the run is still open: aggregate metrics are
            # logged onto it. Ending the run first silently dropped them.
            self._tracker.log_summary(
                report.summary, [case.to_dict() for case in report.failures]
            )

            logger.info("Evaluation finished", extra={"fields": report.summary})
            return report

        finally:
            self._tracker.end()

    def _llm_metric_names(self, agent_name: str) -> tuple[str, ...]:
        """Names backed by an LLM judge, so a semantic score is never read as a rule."""
        names: list[str] = []
        for name in self._engine.metric_names(agent_name):
            metric = self._engine.registry.get(name)
            if metric is not None and getattr(metric, "mode", "") == "llm":
                names.append(name)
        return tuple(names)

    @staticmethod
    def _llm_status(
        evaluations: list[EvaluationResult], llm_metric_names: tuple[str, ...]
    ) -> dict[str, object]:
        """Summarize why LLM-judged metrics did or did not run."""
        if not llm_metric_names:
            return {"status": "disabled", "reason": "no llm_metrics configured"}

        ran: set[str] = set()
        reasons: set[str] = set()

        for evaluation in evaluations:
            for metric in evaluation.metrics:
                if metric.metric_name not in llm_metric_names:
                    continue
                if metric.status == MetricStatus.OK:
                    ran.add(metric.metric_name)
                elif metric.reason:
                    reasons.add(metric.reason)

        if ran:
            return {"status": "ok", "metrics": sorted(ran), "skipped": sorted(reasons)}
        return {
            "status": "unavailable",
            "metrics": sorted(llm_metric_names),
            "reasons": sorted(reasons),
        }
