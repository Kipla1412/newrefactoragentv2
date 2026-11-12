"""MLflow tracking for evaluation results.

Reuses the framework's tracker (``get_mlflow_tracker``), so evaluation runs land
in the same experiment as agent runs and honour ``MLFLOW_ENABLED``.

Three things make this different from the agents' own logging:

1. **It logs by ``run_id``, never via the active run.** Every ``Session`` an
   adapter builds calls ``Session.initialize()``, which ends any active MLflow run
   and starts its own; an agent then opens another for its turn. Logging through
   the module-level API therefore scattered results across several runs.
   Addressing the run by id is immune to that.
2. **Every call is best-effort.** A tracking failure must never stop a report
   being produced, or an agent from replying.
3. **Real-time results get their own short run**, because the observer runs in a
   background task where there is no meaningful active run to attach to.

Rationales are truncated before they are stored: a judge's explanation is a
summary, not a transcript, and this framework handles patient data.
"""

from __future__ import annotations

from typing import Any, Mapping

from evaluation.core.models import EvaluationMode, EvaluationResult, MetricStatus
from utils.logging_config import get_logger
from utils.mlflow_tracker import get_mlflow_tracker

logger = get_logger(__name__)

#: Experiment used when the config does not name one.
DEFAULT_EXPERIMENT = "clinical-agent"

#: MLflow caps parameter values; keep rationales well inside the limit.
MAX_REASON_CHARS = 400


class EvaluationTracker:
    """Log evaluation results to MLflow.

    Implements :class:`~evaluation.ports.tracker.TrackerPort`.

    Args:
        config: Runtime configuration. Tracking is skipped entirely when
            ``MLFLOW_ENABLED`` is false.
        log_reasons: Whether to store metric rationales. Off by default: a
            rationale can paraphrase patient content.
    """

    def __init__(self, config, *, log_reasons: bool = False) -> None:
        self._mlflow = get_mlflow_tracker(config)
        self._experiment = (
            getattr(config, "mlflow_experiment_name", None) or DEFAULT_EXPERIMENT
        )
        self._log_reasons = log_reasons
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

    # Offline runs

    def start(self, run_name: str, tags: Mapping[str, Any]) -> None:
        """Create the evaluation run and tag it with dataset metadata."""
        if not self.enabled:
            return
        client, run_id = self._create_run(
            run_name,
            {"evaluation": "offline", "mode": EvaluationMode.OFFLINE.value},
            tags,
        )
        self._client = client
        self._run_id = run_id

    def log_summary(
        self,
        summary: Mapping[str, Any],
        failures: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record the aggregate scores and, optionally, the failing cases."""
        if not self.active:
            return

        try:
            for name, value in summary.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    self._client.log_metric(self._run_id, f"overall.{name}", float(value))
                else:
                    self._client.log_param(self._run_id, f"overall_{name}", str(value)[:500])

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

    # Per-execution results (offline and real-time)

    def log_execution(
        self,
        result: EvaluationResult,
        *,
        run_id: str | None = None,
        case: str | None = None,
        **tags: Any,
    ) -> None:
        """Record one execution's results.

        Three destinations, in priority order — the first that applies wins, so
        evaluation never creates a rival run when one already exists:

        1. ``run_id``: the execution's own run (where its trace and token usage
           live). Used by real-time evaluation, so scores sit beside the trace.
           Appending to a finished run is intentional: MLflow accepts it and the
           UI then shows metrics and trace together.
        2. The open offline run, with metrics prefixed by ``case`` so 16 cases do
           not overwrite one another in the same run.
        3. A short run of its own, when neither exists.

        Args:
            result: The engine's verdict for one execution.
            run_id: The execution's run, when the caller knows it.
            case: Dataset case id, used as a metric prefix offline.
            tags: Extra tags (scenario, dataset version, ...).
        """
        if not self.enabled:
            return

        context = result.context
        try:
            if run_id:
                client = self._new_client()
                target_run, terminate, prefix = run_id, False, ""
            elif self.active:
                client, target_run, terminate = self._client, self._run_id, False
                prefix = f"{case}." if case else ""
            else:
                client, target_run = self._create_run(
                    f"eval-{context.agent_name}-{context.execution_id or 'unknown'}",
                    {
                        "evaluation": context.mode.value,
                        "agent_name": context.agent_name,
                        "execution_id": context.execution_id or "",
                        "session_id": context.session_id or "",
                        "model": context.model or "",
                        **{k: str(v) for k, v in tags.items()},
                    },
                )
                terminate, prefix = True, ""

            if target_run is None:
                return

            for metric in result.metrics:
                name = f"{prefix}{metric.metric_name}"
                if metric.status == MetricStatus.SKIPPED:
                    client.set_tag(target_run, f"skipped.{name}", metric.reason[:500])
                    continue

                client.log_metric(target_run, f"{name}.score", float(metric.score))
                client.log_metric(target_run, f"{name}.threshold", float(metric.threshold))
                client.log_metric(
                    target_run, f"{name}.passed", 1.0 if metric.passed else 0.0
                )
                client.set_tag(target_run, f"{name}.status", metric.status.value)
                if self._log_reasons and metric.reason:
                    client.set_tag(
                        target_run, f"{name}.reason", metric.reason[:MAX_REASON_CHARS]
                    )

            client.log_metric(target_run, f"{prefix}passed", float(result.passed))
            client.log_metric(target_run, f"{prefix}total", float(result.total))
            client.log_metric(target_run, f"{prefix}duration_ms", float(result.duration_ms))

            # The correlation ids are a tag on every run, so a run and its trace
            # can always be traced back to one execution.
            if context.execution_id:
                client.set_tag(target_run, "execution_id", context.execution_id)
            if context.session_id:
                client.set_tag(target_run, "session_id", context.session_id)
            client.set_tag(target_run, "agent_name", context.agent_name)

            if terminate:
                client.set_terminated(target_run, "FINISHED")

        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("Failed to log execution results: %s", exc)

    @staticmethod
    def _new_client() -> Any:
        """A fresh MlflowClient, so logging never depends on the active run."""
        from mlflow.tracking import MlflowClient

        return MlflowClient()

    def log_failures(self, failures: list[dict[str, Any]]) -> None:
        """Attach a failure list to the open run."""
        if not (self.active and failures):
            return
        try:
            self._client.log_dict(self._run_id, {"failures": failures}, "failures.json")
        except Exception as exc:  # noqa: BLE001 - tracking is best-effort
            logger.warning("Failed to log failures: %s", exc)

    # Internals

    def _create_run(
        self,
        run_name: str,
        tags: Mapping[str, Any],
        extra_tags: Mapping[str, Any] | None = None,
    ) -> tuple[Any, str | None]:
        """Create an MLflow run, returning ``(client, run_id)``.

        Returns ``(client, None)`` when the experiment does not exist, so callers
        skip logging rather than raising.
        """
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        experiment = client.get_experiment_by_name(self._experiment)
        if experiment is None:
            logger.warning(
                "MLflow experiment %r not found; evaluation not logged", self._experiment
            )
            return client, None

        merged = {**{k: str(v) for k, v in tags.items()}}
        for name, value in (extra_tags or {}).items():
            merged[f"dataset_{name}"] = str(value)

        run = client.create_run(experiment.experiment_id, run_name=run_name, tags=merged)
        return client, run.info.run_id

    def __enter__(self) -> "EvaluationTracker":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end()
