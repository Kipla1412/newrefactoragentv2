"""Offline evaluation for the agent framework.

Offline only: it drives the real agents with a dataset and scores the replies.
It is never invoked from API request handling.

The intake agent is the first target. Layout:

=========================  ==================================================
Module                     Responsibility
=========================  ==================================================
``evaluation.dataset``     Case model, critical checks, loader and validation
``evaluation.adapter``     Runs a case through the real agent, normalizes output
``evaluation.metrics``     Deterministic validators and DeepEval metrics
``evaluation.judge``       Routes DeepEval's LLM judge through our LLMClient
``evaluation.tracking``    MLflow logging, reusing the framework's tracker
``evaluation.runner``      Orchestration and the JSON report
=========================  ==================================================

See ``docs/evaluation.md``.
"""

from evaluation.adapter import IntakeAgentAdapter, IntakeObservation
from evaluation.dataset import (
    DEFAULT_DATASET_PATH,
    DatasetError,
    EvalCase,
    EvalDataset,
    load_dataset,
)

__all__ = [
    "DEFAULT_DATASET_PATH",
    "DatasetError",
    "EvalCase",
    "EvalDataset",
    "IntakeAgentAdapter",
    "IntakeObservation",
    "load_dataset",
]
