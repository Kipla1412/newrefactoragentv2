"""MLflow entry point for the framework.

Returns the ``mlflow`` module itself, so existing call sites keep working
(``tracker.log_metrics(...)``, ``tracker.set_tag(...)``). On top of that it adds
the two things the framework needs to keep runs coherent:

1. **One run per execution.** :func:`begin_execution` starts exactly one run and
   tags it with the correlation ids; :func:`end_execution` closes it. Nothing
   else may end a run it did not start — ending someone else's run is what used
   to scatter metrics across runs.
2. **Correlated traces.** :func:`enable_tracing` turns on MLflow's LiteLLM
   integration, which is the correct one for this framework's client
   (``litellm.acompletion``) and covers every provider, not just OpenAI.

Nested usage is handled by nesting the new run rather than terminating the outer
one, because ``mlflow.start_run()`` raises when a run is already active.
"""

from __future__ import annotations

import mlflow

from utils.logging_config import get_logger

logger = get_logger(__name__)

_mlflow_instance = None

#: Tracks whether tracing setup has already been attempted.
_tracing_enabled: bool | None = None


def get_mlflow_tracker(config):
    """Return the configured ``mlflow`` module, or None when tracking is off.

    Idempotent: the first call configures the tracking URI, experiment and
    tracing, later calls return the same module.
    """
    global _mlflow_instance

    if not config.mlflow_enabled:
        logger.info("MLflow tracking disabled")
        return None

    if _mlflow_instance is None:
        try:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())

            mlflow.set_experiment(config.mlflow_experiment_name or "clinical-agent")
            enable_tracing()

            _mlflow_instance = mlflow
        except Exception as e:
            logger.error("MLflow initialization failed: %s", e)
            return None

    return _mlflow_instance


def enable_tracing(*, silent: bool = True) -> bool:
    """Enable MLflow tracing for the provider layer.

    Uses ``mlflow.litellm.autolog()`` rather than ``mlflow.openai.autolog()``:

    - The framework calls ``litellm.acompletion`` for every provider. The openai
      integration only sees OpenAI SDK calls, so it misses gemini/ollama/
      anthropic entirely — and double-traces OpenAI calls, because LiteLLM
      constructs an ``AsyncOpenAI`` client underneath.
    - The litellm integration registers litellm's own success/failure callbacks,
      so it is provider-agnostic and logs one trace per model call.

    Autologging is traces-only in MLflow 3.x: it does not create runs, so it
    cannot conflict with the execution run this framework owns.

    Returns:
        True when tracing was enabled (or already was), False otherwise.
    """
    global _tracing_enabled

    if _tracing_enabled:
        return True

    try:
        import litellm  # noqa: F401  (autolog imports it too; fail early and cheaply)

        mlflow.litellm.autolog(log_traces=True, silent=silent)
        _tracing_enabled = True
        logger.info("MLflow tracing enabled for LiteLLM calls")
        return True
    except Exception as e:
        logger.warning("MLflow tracing could not be enabled: %s", e)
        _tracing_enabled = False
        return False


def tracing_is_enabled() -> bool:
    """Whether :func:`enable_tracing` succeeded in this process."""
    return bool(_tracing_enabled)


def begin_execution(
    tracker,
    *,
    agent_name: str,
    session_id: str,
    execution_id: str,
    run_name: str | None = None,
):
    """Start the single run that represents one agent execution.

    The run is tagged with the ids that evaluation also uses, so scores, spans
    and token usage can be found together.

    Args:
        tracker: The value returned by :func:`get_mlflow_tracker`; None is safe.
        agent_name: Logical agent name.
        session_id: Conversation id.
        execution_id: Execution id, shared with the evaluation context.
        run_name: Override for the run name.

    Returns:
        The MLflow run, or None when tracking is off or the run cannot be started.
    """
    if tracker is None:
        return None

    try:
        name = run_name or f"{agent_name or 'agent'}-{execution_id[:8]}"
        # Never terminate a run we did not start: nest instead, so an outer
        # caller's run survives and no metrics are misfiled.
        if mlflow.active_run() is not None:
            run = tracker.start_run(run_name=name, nested=True)
        else:
            run = tracker.start_run(run_name=name)

        tracker.set_tags(
            {
                "execution_id": execution_id,
                "session_id": session_id,
                "agent_name": agent_name or "unknown",
            }
        )
        logger.debug(
            "MLflow execution run started",
            extra={"fields": {"agent_name": agent_name, "run_id": run.info.run_id}},
        )
        return run
    except Exception as e:
        logger.warning("Failed to start the execution run: %s", e)
        return None


def end_execution(tracker) -> None:
    """Close the execution run, ending only the innermost active run."""
    if tracker is None:
        return
    try:
        if mlflow.active_run() is not None:
            tracker.end_run()
    except Exception as e:
        logger.warning("Failed to end the execution run: %s", e)


def execution_run_id(tracker) -> str | None:
    """The active run's id, for attaching evaluation to the same run."""
    if tracker is None:
        return None
    try:
        run = mlflow.active_run()
    except Exception:  # noqa: BLE001 - tracing is best-effort
        return None
    return run.info.run_id if run else None
