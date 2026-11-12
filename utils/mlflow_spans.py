"""MLflow GenAI spans for agent execution.

One execution is one **run** plus one **trace**:

.. code-block:: text

    run (execution_id, session_id, agent_name)
      └── trace
            └── AGENT span            <- created with run_id=..., which sets
                ├── LLM span             the trace's mlflow.sourceRun metadata
                ├── TOOL span            so the trace and run are found together
                └── LLM span

Evaluation metrics are logged to that same run, so scores, spans, tool behaviour
and token usage are one correlated unit.

Everything here degrades to a no-op: when tracking is off, when tracing is
unavailable, or when a span cannot be created, the caller is unaffected. Spans
must never change what an agent does.

**PHI.** Span inputs and outputs are visible in the MLflow UI, so content is off
by default: spans record sizes, counts, roles and tool *argument keys* rather
than text. Set ``LOG_PROMPTS=true`` to include content, matching the framework's
existing logging policy.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

from utils.logging_config import get_logger, prompts_enabled, redact

logger = get_logger(__name__)

# Span types from ``mlflow.entities.SpanType``; named as strings so a missing
# tracing dependency cannot break this module.
SPAN_AGENT = "AGENT"
SPAN_LLM = "LLM"
SPAN_TOOL = "TOOL"

#: Tool arguments whose values are never written to a span.
SENSITIVE_ARGUMENT_KEYS: frozenset[str] = frozenset(
    {"password", "secret", "token", "api_key", "authorization"}
)


def tracing_enabled() -> bool:
    """Whether MLflow tracing can be used in this process."""
    try:
        from mlflow.tracing.fluent import start_span  # noqa: F401
    except Exception:
        return False
    return True


def describe_messages(messages: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Summarize messages without exposing their content (PHI-safe)."""
    if not messages:
        return {"message_count": 0, "roles": [], "content_chars": 0}

    roles: list[str] = []
    content_chars = 0
    for message in messages:
        roles.append(str(message.get("role", "unknown")))
        content = message.get("content")
        if isinstance(content, str):
            content_chars += len(content)
        elif content is not None:
            content_chars += len(str(content))

    return {
        "message_count": len(messages),
        "roles": roles,
        "content_chars": content_chars,
    }


def describe_tool_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    """Summarize tool arguments by key, masking names that look sensitive.

    Values are withheld unless ``LOG_PROMPTS`` is on, because tool arguments can
    carry patient identifiers.
    """
    if not arguments:
        return {}

    safe = redact(dict(arguments))
    if prompts_enabled():
        return safe

    described: dict[str, Any] = {}
    for key, value in safe.items():
        if str(key).lower() in SENSITIVE_ARGUMENT_KEYS:
            described[key] = "***"
        else:
            described[key] = f"<{type(value).__name__}:{len(str(value))}>"
    return described


def describe_tool_result(result: Any) -> dict[str, Any]:
    """Summarize a tool result without exposing its content."""
    success = bool(getattr(result, "success", False))
    output = getattr(result, "output", None) or getattr(result, "error", None)
    summary: dict[str, Any] = {"success": success}
    if prompts_enabled():
        summary["output"] = str(output)[:2000] if output is not None else None
    elif output is not None:
        summary["output_chars"] = len(str(output))
    return summary


@contextmanager
def agent_span(
    *,
    name: str,
    run_id: str | None = None,
    execution_id: str = "",
    session_id: str = "",
    agent_name: str = "",
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open the root span for one agent execution.

    Args:
        name: Span name, typically the agent name.
        run_id: Execution's MLflow run. Passed to ``start_span`` so the trace is
            linked to the run (``mlflow.sourceRun``), which is what makes the
            trace and the run discoverable together.
        execution_id: Correlation id shared with evaluation.
        session_id: Conversation the execution belongs to.
        agent_name: Logical agent name.
        attributes: Extra span attributes.

    Yields:
        The live span, or None when tracing is unavailable.
    """
    with _span(
        name=name or "agent",
        span_type=SPAN_AGENT,
        run_id=run_id,
        inputs={
            "execution_id": execution_id,
            "session_id": session_id,
            "agent_name": agent_name,
        },
        attributes={"execution_id": execution_id, "session_id": session_id},
        extra_attributes=attributes,
    ) as span:
        yield span


@contextmanager
def llm_span(
    *,
    model: str | None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    attempt: int | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a span around one model call.

    Yields:
        The live span, or None. Callers record tokens and the response summary
        once the call returns, via :func:`finish_llm_span`.
    """
    with _span(
        name=f"llm:{model or 'default'}",
        span_type=SPAN_LLM,
        run_id=None,
        inputs=describe_messages(messages),
        attributes={
            "model": model or "",
            "attempt": attempt if attempt is not None else 0,
        },
        extra_attributes=attributes,
    ) as span:
        yield span


def finish_llm_span(span: Any, *, usage: Any | None, response_chars: int) -> None:
    """Record tokens and the response summary on an LLM span.

    Args:
        span: The span yielded by :func:`llm_span` (None is ignored).
        usage: Token usage object, when the provider reported one.
        response_chars: Length of the reply, so the span is useful without
            storing its text.
    """
    if span is None:
        return

    summary: dict[str, Any] = {"response_chars": response_chars}
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
        summary.update(
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": total,
            }
        )
        _set_attributes(
            span,
            {
                **{k: v for k, v in (
                    ("prompt_tokens", prompt),
                    ("completion_tokens", completion),
                    ("total_tokens", total),
                ) if v is not None}
            },
        )

    _set_outputs(span, summary)


@contextmanager
def tool_span(
    *,
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Open a span around one tool invocation.

    Yields:
        The live span, or None. Callers record the outcome with
        :func:`finish_tool_span`.
    """
    with _span(
        name=f"tool:{tool_name}",
        span_type=SPAN_TOOL,
        run_id=None,
        inputs={"arguments": describe_tool_arguments(arguments)},
        attributes={"tool": tool_name},
        extra_attributes=attributes,
    ) as span:
        yield span


def finish_tool_span(span: Any, result: Any) -> None:
    """Record a tool result's outcome on its span (content gated by LOG_PROMPTS)."""
    if span is None:
        return
    summary = describe_tool_result(result)
    _set_outputs(span, summary)
    _set_attributes(span, {"success": summary.get("success", False)})


# Internals


@contextmanager
def _span(
    *,
    name: str,
    span_type: str,
    run_id: str | None,
    inputs: Mapping[str, Any] | None,
    attributes: Mapping[str, Any] | None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Create a span, or yield None when tracing is unavailable.

    The ``yield`` sits **outside** every ``try`` block: a span must never
    intercept an exception raised by the caller's body, and a context manager
    must never yield twice. Only span creation, attribute setting and teardown
    are guarded — an agent's behaviour is never affected by tracing.
    """
    if not tracing_enabled():
        yield None
        return

    span_cm = None
    try:
        import mlflow

        kwargs: dict[str, Any] = {"name": name, "span_type": span_type}
        if run_id:
            kwargs["run_id"] = run_id
        span_cm = mlflow.start_span(**kwargs)
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("Span '%s' could not be created: %s", name, exc)

    if span_cm is None:
        yield None
        return

    live = None
    try:
        live = span_cm.__enter__()
        _set_inputs(live, inputs or {})
        _set_attributes(live, {**(attributes or {}), **(extra_attributes or {})})
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("Span '%s' could not be started: %s", name, exc)

    try:
        yield live
    finally:
        if live is not None:
            try:
                span_cm.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - tracing is best-effort
                logger.debug("Span '%s' could not be ended: %s", name, exc)


def _set_inputs(span: Any, inputs: Mapping[str, Any]) -> None:
    try:
        span.set_inputs(dict(inputs))
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("Could not set span inputs: %s", exc)


def _set_outputs(span: Any, outputs: Mapping[str, Any]) -> None:
    try:
        span.set_outputs(dict(outputs))
    except Exception as exc:  # noqa: BLE001 - tracing is best-effort
        logger.debug("Could not set span outputs: %s", exc)


def _set_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    clean = {k: v for k, v in attributes.items() if v is not None}
    if not clean:
        return
    try:
        span.set_attributes(clean)
    except Exception:  # noqa: BLE001 - some span types reject non-scalar values
        for key, value in clean.items():
            try:
                span.set_attribute(key, value)
            except Exception as exc:  # noqa: BLE001 - tracing is best-effort
                logger.debug("Could not set span attribute '%s': %s", key, exc)
