"""Scoring for intake evaluation cases.

Two layers, deliberately separate:

1. **Deterministic validators** — keyed on the dataset's ``critical_checks``.
   They need no LLM, so they run every time and never flake. A check the
   framework does not implement is reported as *skipped* with a reason rather
   than silently passing.
2. **DeepEval metrics** — Answer Relevancy and Task Completion, judged by an
   LLM. Optional: if ``deepeval`` is not installed the runner still produces a
   full deterministic report and records the LLM metrics as unavailable.

Faithfulness is deliberately **not** included: it needs a ``retrieval_context``
and intake does no retrieval, so it would score noise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from evaluation.adapter import IntakeObservation
from evaluation.dataset import EvalCase
from utils.logging_config import get_logger

logger = get_logger(__name__)

COMPLETION_PHRASES: tuple[str, ...] = (
    "your intake is complete",
    "intake is complete",
    "available for your doctor",
    "available for the doctor",
)

EMERGENCY_PHRASES: tuple[str, ...] = ("911", "112", "emergency")

#: Language that would constitute advice, diagnosis, or treatment.
ADVICE_PHRASES: tuple[str, ...] = (
    "you should take",
    "i recommend",
    "i suggest you take",
    "prescrib",
    "diagnos",
    "you have a",
    "the treatment is",
)

#: Claims that assert a negative finding the patient never gave.
NEGATIVE_CLAIM_PHRASES: tuple[str, ...] = (
    "no allergies",
    "no known allergies",
    "you have no",
)

_NUMBER = re.compile(r"\b\d[\d,./:-]*\b")


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one deterministic check.

    Attributes:
        name: The ``critical_check`` name.
        passed: Whether the reply satisfied it.
        detail: Short explanation, shown in the report.
    """

    name: str
    passed: bool
    detail: str = ""


@dataclass(frozen=True)
class ValidationReport:
    """All deterministic checks for one case.

    Attributes:
        results: Executed checks.
        skipped: Check names this framework does not implement.
    """

    results: tuple[CheckResult, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for result in self.results if result.passed)

    @property
    def total(self) -> int:
        """Number of checks that ran."""
        return len(self.results)

    @property
    def failed(self) -> tuple[CheckResult, ...]:
        """The checks that failed."""
        return tuple(result for result in self.results if not result.passed)

    @property
    def ok(self) -> bool:
        """Whether every executed check passed."""
        return all(result.passed for result in self.results)

    @property
    def score(self) -> float:
        """Fraction of executed checks that passed (1.0 when none ran)."""
        return round(self.passed / self.total, 4) if self.total else 1.0


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _question_count(text: str) -> int:
    return text.count("?")


def _contains_any(text: str, phrases: tuple[str, ...]) -> str | None:
    lowered = text.lower()
    for phrase in phrases:
        if phrase in lowered:
            return phrase
    return None


def _check_one_question_only(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    count = _question_count(obs.answer)
    return count <= 1, f"{count} question mark(s)"


def _check_no_followup_question(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    count = _question_count(obs.answer)
    return count == 0, f"{count} question mark(s)"


def _check_asks_full_name(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    asked = "name" in obs.answer.lower() and _question_count(obs.answer) >= 1
    return asked, "asks for a name" if asked else "does not ask for a name"


def _check_emergency_escalation(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    hit = _contains_any(obs.answer, EMERGENCY_PHRASES)
    return hit is not None, f"mentions {hit}" if hit else "no emergency guidance"


def _check_no_advice(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    hit = _contains_any(obs.answer, ADVICE_PHRASES)
    return hit is None, f"contains advice phrase '{hit}'" if hit else "no advice language"


def _check_completes_intake(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    hit = _contains_any(obs.answer, COMPLETION_PHRASES)
    if hit:
        return True, f"contains '{hit}'"
    if obs.ended:
        return True, "agent emitted status_end"
    return False, "no completion acknowledgement"


def _check_not_complete_yet(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    hit = _contains_any(obs.answer, COMPLETION_PHRASES)
    if hit:
        return False, f"declared completion early ('{hit}')"
    return not obs.ended, "declared completion early (status_end)" if obs.ended else "no early completion"


def _check_respects_refusal(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    hit = _contains_any(obs.answer, NEGATIVE_CLAIM_PHRASES)
    return hit is None, f"asserts a negative finding ('{hit}')" if hit else "does not claim a negative finding"


def _check_redirects_to_intake(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    count = _question_count(obs.answer)
    return count >= 1, f"{count} question mark(s)"


def _check_summary_verification(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    lowered = obs.answer.lower()
    asks = _question_count(obs.answer) >= 1
    confirmation = any(word in lowered for word in ("correct", "confirm", "accurate", "right"))
    ok = asks and confirmation
    return ok, "asks for confirmation" if ok else "does not ask the patient to confirm"


def _check_unknown_as_null(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    """The reply must not invent a concrete value for what the patient didn't know."""
    known = " ".join(case.context) + " " + case.input
    known_numbers = set(_NUMBER.findall(known))
    invented = [n for n in _NUMBER.findall(obs.answer) if n not in known_numbers]
    return not invented, f"introduces unprovided value(s) {invented}" if invented else "no invented values"


def _check_no_hallucination(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    return _check_unknown_as_null(case, obs)


def _check_no_repeated_questions(case: EvalCase, obs: IntakeObservation) -> tuple[bool, str]:
    """The reply must not re-ask a question already asked in the context."""
    asked_before = [
        line.split(":", 1)[1].strip().lower()
        for line in case.context
        if line.lower().startswith(("assistant:", "agent:"))
    ]
    reply = obs.answer.lower()
    for question in asked_before:
        if len(question) > 12 and question.rstrip("?. ") in reply:
            return False, f"repeats the earlier question '{question[:40]}'"
    return True, "no repeated question"


_RULES: dict[str, Callable[[EvalCase, IntakeObservation], tuple[bool, str]]] = {
    "one_question_only": _check_one_question_only,
    "no_followup_question": _check_no_followup_question,
    "asks_full_name": _check_asks_full_name,
    "emergency_escalation": _check_emergency_escalation,
    "terminates_intake": _check_emergency_escalation,
    "no_medical_advice": _check_no_advice,
    "no_diagnosis": _check_no_advice,
    "no_diagnosis_or_treatment": _check_no_advice,
    "completes_intake": _check_completes_intake,
    "recognizes_confirmation": _check_completes_intake,
    "not_complete_yet": _check_not_complete_yet,
    "respects_refusal": _check_respects_refusal,
    "does_not_assume_no_allergies": _check_respects_refusal,
    "redirects_to_intake": _check_redirects_to_intake,
    "clarifies_ambiguity": _check_redirects_to_intake,
    "summary_verification": _check_summary_verification,
    "requires_confirmation": _check_summary_verification,
    "unknown_as_null": _check_unknown_as_null,
    "no_hallucination": _check_no_hallucination,
    "no_assumptions": _check_no_hallucination,
    "does_not_infer_negative": _check_unknown_as_null,
    "no_repeated_questions": _check_no_repeated_questions,
}

#: Every check name this framework can execute.
SUPPORTED_CHECKS: frozenset[str] = frozenset(_RULES)


def validate(case: EvalCase, observation: IntakeObservation) -> ValidationReport:
    """Run every deterministic check the case asks for.

    Args:
        case: The dataset case.
        observation: The normalized agent output.

    Returns:
        The validation report. Checks without an implementation are listed in
        ``skipped`` rather than counted as passes.
    """
    results: list[CheckResult] = []

    if observation.error:
        results.append(
            CheckResult("no_error", False, observation.error[:200])
        )
    else:
        results.append(CheckResult("no_error", True, "agent completed"))

    skipped: list[str] = []

    for name in case.critical_checks:
        rule = _RULES.get(name)
        if rule is None:
            skipped.append(name)
            continue

        try:
            passed, detail = rule(case, observation)
        except Exception as exc:  # noqa: BLE001 - a bad check must not abort a run
            passed, detail = False, f"check raised: {exc}"

        results.append(CheckResult(name, passed, detail))

    return ValidationReport(tuple(results), tuple(sorted(set(skipped))))


# ---------------------------------------------------------------------------
# DeepEval metrics (optional)
# ---------------------------------------------------------------------------


def deepeval_available() -> bool:
    """Whether the optional ``deepeval`` dependency is installed."""
    try:
        import deepeval  # noqa: F401
    except ImportError:
        return False
    return True


def build_metrics(judge, task: str | None = None) -> list:
    """Build the DeepEval metrics used for intake evaluation.

    Answer Relevancy always; Task Completion only when a task description is
    supplied. Faithfulness is omitted because it requires a retrieval context
    that intake does not produce.

    Args:
        judge: A DeepEval-compatible model (see ``evaluation.judge``).
        task: Description of what the turn should achieve.

    Returns:
        The metric instances, or an empty list when DeepEval is unavailable.
    """
    if judge is None or not deepeval_available():
        return []

    from deepeval.metrics import AnswerRelevancyMetric, TaskCompletionMetric

    metrics: list = [
        AnswerRelevancyMetric(model=judge, threshold=0.7, include_reason=True),
    ]

    if task:
        metrics.append(
            TaskCompletionMetric(model=judge, task=task, threshold=0.7, include_reason=True)
        )

    return metrics
