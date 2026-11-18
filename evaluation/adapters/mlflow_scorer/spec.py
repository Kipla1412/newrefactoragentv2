"""What each MLflow DeepEval scorer is, and what it needs to run.

Verified against the installed source (``mlflow 3.17.0``, ``deepeval 4.3.2``)
rather than the documentation:

- ``mlflow.genai.scorers.deepeval.get_scorer(name, model=None, **metric_kwargs)``
  returns a scorer called as ``scorer(inputs=, outputs=, expectations=)``.
- It returns an MLflow ``Feedback`` and **never raises** — a failed judgement
  comes back as ``Feedback(error=<exception>)``.
- ``DeepEvalScorer.__init__`` accepts only a *model URI string*
  (``"openai:/gpt-4.1-mini"``). There is no seam to inject the framework's
  ``LLMClient``, so these scorers authenticate from provider-native environment
  variables (``OPENAI_API_KEY``) — not from ``API_KEY``/``BASE_URL``.
- MLflow sources ``retrieval_context`` and tool calls from **trace spans only**,
  so any metric needing them cannot run without a trace.

Declaring those requirements here — rather than discovering them as opaque judge
errors at run time — is why this module exists separately from the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Default judge threshold when a scorer reports none.
DEFAULT_THRESHOLD = 0.5

#: Rationales are judge summaries, not raw transcripts; still bounded.
DEFAULT_MAX_REASON_CHARS = 1000


@dataclass(frozen=True)
class ScorerSpec:
    """What one MLflow DeepEval scorer needs to be runnable.

    Attributes:
        scorer_name: The MLflow/DeepEval metric name, e.g. ``"AnswerRelevancy"``.
        requires_context: Fields that must be present on the evaluation context
            or its metadata, e.g. ``"expected_output"``.
        requires_trace: True when MLflow reads the input from trace spans only.
            This framework does not emit per-turn traces, so such a metric is
            skipped with an explicit reason.
        requires_config: Metric kwargs the caller must supply in configuration,
            e.g. ``"expected_schema"``.
    """

    scorer_name: str
    requires_context: tuple[str, ...] = ()
    requires_trace: bool = False
    requires_config: tuple[str, ...] = ()


#: Registry name -> scorer specification.
#:
#: Names are snake_case to match the framework's metric convention; the mapping
#: to MLflow's CamelCase names lives here and nowhere else.
#:
#: Requirements are taken from DeepEval's own ``_required_params``, not guessed:
#: the agentic quality metrics need only ``input`` + ``actual_output``, so they
#: run against a plain reply. Anything needing ``tools_called`` is marked
#: ``requires_trace`` because MLflow sources tool calls from trace spans, which
#: this framework does not feed the scorer.
MLFLOW_SCORERS: dict[str, ScorerSpec] = {
    # RAG / general
    "answer_relevancy": ScorerSpec("AnswerRelevancy"),
    "faithfulness": ScorerSpec("Faithfulness", requires_trace=True),
    "toxicity": ScorerSpec("Toxicity"),
    "json_correctness": ScorerSpec(
        "JsonCorrectness", requires_config=("expected_schema",)
    ),
    # Agentic quality. DeepEval declares only input/actual_output for these, but
    # MLflow's implementation is a *trace analyst*: without a trace it cannot see
    # the plan or the steps. Observed behaviour without one — plan_quality
    # returned a vacuous 1.00 ("no plans to evaluate") and step_efficiency 0.00.
    # Both are meaningless, so they require a trace.
    "task_completion": ScorerSpec(
        "TaskCompletion", requires_context=("expected_output",)
    ),
    "plan_quality": ScorerSpec("PlanQuality", requires_trace=True),
    "plan_adherence": ScorerSpec("PlanAdherence", requires_trace=True),
    "step_efficiency": ScorerSpec("StepEfficiency", requires_trace=True),
    # Tool behaviour — needs the tool calls MLflow reads from a trace
    "argument_correctness": ScorerSpec("ArgumentCorrectness", requires_trace=True),
    # ToolCorrectness also needs the calls the run *should* have made, which come
    # from the case's expected_tool_calls.
    "tool_correctness": ScorerSpec(
        "ToolCorrectness",
        requires_trace=True,
        requires_context=("expected_tool_calls",),
    ),
}


def mlflow_scorers_available() -> bool:
    """Whether MLflow's DeepEval integration and ``deepeval`` can be imported."""
    try:
        import mlflow.genai.scorers.deepeval  # noqa: F401
    except Exception:
        return False
    return True
