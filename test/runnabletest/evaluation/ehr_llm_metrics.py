"""Try the agentic LLM metrics on one EHR reply.

Usage (from anywhere):

    uv run python test/runnabletest/evaluation/ehr_llm_metrics.py

Scores a single synthetic EHR turn with the metrics ``agents.yaml`` configures for
the ``ehr`` agent — TaskCompletion, PlanQuality, StepEfficiency — through the real
evaluation engine, so what you see here is exactly what a run records.

No agent, no session, no FHIR server, no dataset: this is for checking that the
metrics work and what their scores look like. It does call the judge, so it needs
``OPENAI_API_KEY`` (MLflow's scorers do not use the framework's
``API_KEY``/``BASE_URL``) and ``evaluation.llm.allow_external_calls: true``.

The turn is synthetic. Do not paste real patient text into this file.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../test/runnabletest/evaluation/ehr_llm_metrics.py -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_config  # noqa: E402
from evaluation.composition import build_engine  # noqa: E402
from evaluation.core.models import (  # noqa: E402
    EvaluationContext,
    EvaluationMode,
    MetricStatus,
)
from utils.mlflow_spans import agent_span, tool_span  # noqa: E402

#: A synthetic EHR retrieval turn, in the shape the agent produces.
SAMPLE_INPUT = (
    "Show me the allergies recorded for patient 12345."
)

SAMPLE_OUTPUT = (
    "I looked up patient 12345 and found one recorded allergy: penicillin, "
    "reaction reported as a rash. No other allergies are recorded in the chart."
)

SAMPLE_EXPECTED = (
    "Retrieve the patient's allergy list and report the recorded allergy and "
    "reaction, or state clearly that none are recorded."
)

SAMPLE_TOOL = "fhir__get_patient_core_by_id"


def _synthetic_trace() -> str | None:
    """Record the trace a real agent turn would produce.

    plan_quality, step_efficiency and tool_correctness are trace analysts: with no
    trace they either skip or, worse, return a meaningless score. This builds the
    same AGENT + TOOL span shape the agent loop creates, so the metrics have
    something real to read.
    """
    with agent_span(
        name="ehr",
        execution_id="ehr-llm-metrics-demo",
        agent_name="ehr",
    ) as span:
        with tool_span(tool_name=SAMPLE_TOOL) as tool:
            if tool is not None:
                tool.set_outputs({"allergies": ["penicillin"]})
        trace_id = getattr(span, "trace_id", None)

    return trace_id


def main() -> int:
    config = load_config(PROJECT_ROOT)

    if not getattr(config.evaluation.llm, "allow_external_calls", False):
        print(
            "External judge calls are disabled.\n"
            "Set evaluation.llm.allow_external_calls: true in agents.yaml "
            "to run these metrics.",
            file=sys.stderr,
        )
        return 2

    engine = build_engine(config)
    names = engine.metric_names("ehr")

    if not names:
        print(
            "No metrics configured for agent 'ehr'. Add them under "
            "evaluation.agents.ehr.llm_metrics in agents.yaml.",
            file=sys.stderr,
        )
        return 2

    context = EvaluationContext(
        agent_name="ehr",
        input=SAMPLE_INPUT,
        output=SAMPLE_OUTPUT,
        execution_id="ehr-llm-metrics-demo",
        session_id=None,
        metadata={
            "expected_output": SAMPLE_EXPECTED,
            "expected_tool_calls": [{"name": SAMPLE_TOOL}],
            "mlflow_trace_id": _synthetic_trace(),
        },
        mode=EvaluationMode.OFFLINE,
    )

    print(f"metrics: {', '.join(names)}\n")
    result = engine.evaluate(context)

    for metric in result.metrics:
        verdict = "PASS" if metric.passed else "FAIL"
        if metric.status == MetricStatus.SKIPPED:
            verdict = "SKIP"
        elif metric.status == MetricStatus.ERROR:
            verdict = "ERR "

        print(f"{verdict}  {metric.metric_name:20s} {metric.score:5.2f}"
              f"  (threshold {metric.threshold:.2f})")
        if metric.reason:
            print(f"      {metric.reason}")

    print(f"\n{result.passed}/{result.total} passed · {result.duration_ms:.0f}ms")
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
