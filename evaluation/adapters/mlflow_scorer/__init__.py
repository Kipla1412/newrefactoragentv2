"""MLflow's built-in DeepEval scorers, as framework metrics.

Split by responsibility:

- :mod:`~evaluation.adapters.mlflow_scorer.spec` — which scorers exist and what
  each needs.
- :mod:`~evaluation.adapters.mlflow_scorer.metric` — the Adapter and the
  ``Feedback`` -> ``MetricResult`` conversion.
- :mod:`~evaluation.adapters.mlflow_scorer.registration` — registry wiring.

Provider-native credentials are required (``OPENAI_API_KEY``); see
``docs/evaluation.md``.
"""

from evaluation.adapters.mlflow_scorer.metric import (
    MLflowDeepEvalMetric,
    build_scorer,
    feedback_to_metric_result,
)
from evaluation.adapters.mlflow_scorer.registration import register
from evaluation.adapters.mlflow_scorer.spec import (
    DEFAULT_MAX_REASON_CHARS,
    DEFAULT_THRESHOLD,
    MLFLOW_SCORERS,
    ScorerSpec,
    mlflow_scorers_available,
)

__all__ = [
    "DEFAULT_MAX_REASON_CHARS",
    "DEFAULT_THRESHOLD",
    "MLFLOW_SCORERS",
    "MLflowDeepEvalMetric",
    "ScorerSpec",
    "build_scorer",
    "feedback_to_metric_result",
    "mlflow_scorers_available",
    "register",
]
