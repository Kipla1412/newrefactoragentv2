"""Evaluation for the agent framework.

One engine, two modes — real-time and offline — differing only in where the
evaluation event came from.

Layers (dependencies point inward, toward ``core``):

=============================  =================================================
``evaluation.core``            Abstractions: models, metric strategy, registry,
                               selection, engine. No vendor, agent or framework
                               knowledge.
``evaluation.ports``           Protocols the runtime depends on (DIP boundaries).
``evaluation.adapters``        Outbound integrations: agents, DeepEval, MLflow,
                               JSON.
``evaluation.agents``          Per-agent metric sets (plugins).
``evaluation.datasets``        Golden datasets: models, versioned loader.
``evaluation.runtime``         Real-time observer and offline runner.
``evaluation.reporting``       Run report value objects.
``evaluation.composition``     The composition root that wires the above together.
=============================  =================================================

Heavy adapters (agents, MLflow, DeepEval) are imported lazily, so importing
:mod:`evaluation` stays cheap and cannot create an import cycle with the agent
runtime.

See ``docs/evaluation.md``.
"""

from typing import Any

from evaluation.core import (
    CheckFn,
    DeterministicMetric,
    EvaluationContext,
    EvaluationEngine,
    EvaluationMode,
    EvaluationResult,
    Metric,
    MetricFactory,
    MetricRegistry,
    MetricResult,
    MetricSelection,
    MetricStatus,
    dedupe_metrics,
    utc_now,
)

#: Symbols resolved on first access, mapped to the module that defines them.
_LAZY: dict[str, str] = {
    # Composition root
    "build_registry": "evaluation.composition",
    "build_engine": "evaluation.composition",
    "build_tracker": "evaluation.composition",
    "build_observer": "evaluation.composition",
    "build_offline_runner": "evaluation.composition",
    # Datasets
    "EvalCase": "evaluation.datasets",
    "EvalDataset": "evaluation.datasets",
    "DatasetError": "evaluation.datasets",
    "FileDatasetSource": "evaluation.datasets",
    "load_dataset": "evaluation.datasets",
    "dataset_path": "evaluation.datasets",
    # Reporting
    "CaseResult": "evaluation.reporting",
    "EvaluationReport": "evaluation.reporting",
    # Runtime
    "OfflineRunner": "evaluation.runtime",
    "EvaluationObserver": "evaluation.runtime",
    # Adapters
    "AgentCaseRunner": "evaluation.adapters.agent_case_runner",
    "EvaluationTracker": "evaluation.adapters.mlflow_tracker",
    "JsonEvaluationRepository": "evaluation.adapters.json_repository",
    "build_judge": "evaluation.adapters.deepeval_judge",
    "deepeval_installed": "evaluation.adapters.deepeval_judge",
    "MLflowDeepEvalMetric": "evaluation.adapters.mlflow_scorer",
    "ScorerSpec": "evaluation.adapters.mlflow_scorer",
    "MLFLOW_SCORERS": "evaluation.adapters.mlflow_scorer",
    "feedback_to_metric_result": "evaluation.adapters.mlflow_scorer",
    "mlflow_scorers_available": "evaluation.adapters.mlflow_scorer",
}

__all__ = [
    "AgentCaseRunner",
    "CaseResult",
    "CheckFn",
    "DatasetError",
    "DeterministicMetric",
    "EvalCase",
    "EvalDataset",
    "EvaluationContext",
    "EvaluationEngine",
    "EvaluationMode",
    "EvaluationObserver",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationTracker",
    "FileDatasetSource",
    "JsonEvaluationRepository",
    "MLFLOW_SCORERS",
    "MLflowDeepEvalMetric",
    "Metric",
    "MetricFactory",
    "MetricRegistry",
    "MetricResult",
    "MetricSelection",
    "MetricStatus",
    "OfflineRunner",
    "ScorerSpec",
    "build_engine",
    "build_judge",
    "build_observer",
    "build_offline_runner",
    "build_registry",
    "build_tracker",
    "dataset_path",
    "dedupe_metrics",
    "deepeval_installed",
    "feedback_to_metric_result",
    "load_dataset",
    "mlflow_scorers_available",
    "utc_now",
]


def __getattr__(name: str) -> Any:
    """Resolve heavy symbols on first access (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
