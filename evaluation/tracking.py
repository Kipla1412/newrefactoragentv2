"""MLflow logging for evaluation runs.

Reuses the framework's tracker (``get_mlflow_tracker``), so evaluation runs land
in the same experiment as agent runs and honour ``MLFLOW_ENABLED``.

Two things make this different from the agents' logging:

1. **It logs by ``run_id``, never via the active run.** Every ``Session`` the
   adapter builds calls ``Session.initialize()``, which ends any active MLflow run
   and starts its own; the agent then opens another for its turn. Logging through
   the module-level API therefore scattered the evaluation metrics across several
   runs. Addressing the run by id is immune to that.
2. **Every call is best-effort.** A tracking failure must never stop a report
   being produced.
"""

from __future__ import annotations

from typing import Any

from utils.logging_config import get_logger
from utils.mlflow_tracker import get_mlflow_tracker

logger = get_logger(__name__)

#: Experiment used when the config does not name one.
DEFAULT_EXPERIMENT = "clinical-agent"


class EvaluationTracker:
    """Log one evaluation run to MLflow.

    Args:
        config: Runtime configuration. Tracking is skipped entirely when
            ``MLFLOW_ENABLED`` is false.
    """

    def __init__(self, config) -> None:
        self._mlflow = get_mlflow_tracker(config)
        self._experiment = getattr(config, "mlflow_experiment_name", None) or DEFAULT_EXPERIMENT
        self._client: Any | None = None
        self._run_id: str | None = None

    @property
    def enabled(self) -> bool:
        """Whether an MLflow backend is configured."""
        return self._mlflow is not None

    @property
    def active(self) -> bool:
        """Whether an evaluation run is currently open."""
        return self._run_id is not None

    @property
    def run_id(self) -> str | None:
        """The evaluation run id, once started."""
        return self._run_id

    def start(self, run_name: str, dataset_info: dict[str, Any]) -> None:
        """Create the evaluation run and tag it with the dataset metadata.

        Args:
            run_name: Name for the MLflow run.
            dataset_info: Dataset identifiers recorded as tags.
        """
        if not self.enabled:
            return

        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            experiment = client.get_experiment_by_name(self._experiment)
            if experiment is None:
                logger.warning(
                    "MLflow experiment %r not found; evaluation not logged",
                    self._experiment,
                )
                return

            run = client.create_run(
                experiment.experiment_id,
                run_name=run_name,
                tags={
                    "evaluation": "intake-offline",
                    **{
                        f"dataset_{name}": str(value)
                        for name, value in dataset_info.items()
                    },
                },
            )
            self._client = client
            self._run_id = run.info.run_id
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("Failed to start evaluation run: %s", exc)
            self._client = None
            self._run_id = None

    def log_case(self, case_id: str, scores: dict[str, float]) -> None:
        """Record one case's scores, namespaced by case id.

        Args:
            case_id: The dataset case identifier.
            scores: Metric name → value.
        """
        if not (self.active and scores):
            return

        try:
            for name, value in scores.items():
                self._client.log_metric(self._run_id, f"{case_id}.{name}", float(value))
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("Failed to log metrics for %s: %s", case_id, exc)

    def log_summary(self, summary: dict[str, Any], failures: list[dict[str, Any]]) -> None:
        """Record the aggregate scores, the failure list, and the duration.

        Args:
            summary: Aggregate report summary.
            failures: Cases that failed or lost a check.
        """
        if not self.active:
            return

        try:
            for name, value in summary.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    self._client.log_metric(self._run_id, f"overall.{name}", float(value))
                else:
                    self._client.log_param(self._run_id, f"overall_{name}", str(value))

            if failures:
                self._client.log_dict(
                    self._run_id, {"failures": failures}, "failures.json"
                )
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("Failed to log evaluation summary: %s", exc)

    def end(self) -> None:
        """Close the evaluation run, if one is open."""
        if not self.active:
            return

        try:
            self._client.set_terminated(self._run_id, "FINISHED")
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("Failed to end evaluation run: %s", exc)
        finally:
            self._run_id = None
            self._client = None

    def __enter__(self) -> "EvaluationTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()
