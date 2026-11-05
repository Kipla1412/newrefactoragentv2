"""Intake Agent metric set.

Registers the deterministic clinical rules plus the clinical-facing names used in
configuration. Nothing is reimplemented: an alias shares its target's
``canonical_name``, so configuring both measures the rule once.
"""

from __future__ import annotations

from evaluation.agents.intake.rules import CHECKS
from evaluation.core.registry import MetricRegistry

#: Clinical-facing names mapped onto existing rules.
ALIASES: dict[str, str] = {
    "emergency_compliance": "emergency_escalation",
    "medical_boundary_compliance": "no_medical_advice",
    "intake_completion": "completes_intake",
}

#: Metrics the Intake Agent runs by default (see ``agents.yaml``).
#:
#: These are **per-turn** rules: how the agent behaves while intake is in
#: progress. Correctly, most of them pass only when the turn does *not* carry a
#: clinical action (asking a question, not diagnosing).
DEFAULT_METRICS: tuple[str, ...] = (
    "medical_boundary_compliance",
    "no_repeated_questions",
)

#: End-of-intake outcome rules.
#:
#: These only make sense when intake is finishing — an emergency, or a completed
#: summary. Applied to a normal mid-intake turn they would fail, so they are NOT
#: part of the default set: they run when a test case asks for them in
#: ``critical_checks`` (``emergency_escalation``, ``emergency_compliance``,
#: ``completes_intake``, ``intake_completion``). Keeping them out of the default
#: set is what stops "did not end the intake early" being scored as a failure.
END_OF_INTAKE_METRICS: tuple[str, ...] = (
    "emergency_compliance",
    "intake_completion",
)


def register(registry: MetricRegistry) -> None:
    """Register the intake rules and their clinical aliases."""
    registry.register_from_checks(CHECKS, aliases=ALIASES)
