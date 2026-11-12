"""Intake Agent metric set.

Registers the deterministic clinical rules plus the clinical-facing names used in
configuration. Nothing is reimplemented: an alias shares its target's
``canonical_name``, so configuring both measures the rule once.

Two kinds of rule live in :data:`~evaluation.agents.intake.rules.CHECKS`:

- **per-turn** — how the agent behaves while intake is in progress
  (:data:`DEFAULT_METRICS`, listed in ``agents.yaml``).
- **outcome** — emergency escalation, completion acknowledgement, summary
  verification. These fail by construction on an ordinary mid-intake turn, so
  they are not configured per agent; each dataset case requests the ones it
  expects through its ``critical_checks``, which the offline runner adds for that
  case (see ``evaluation/runtime/offline.py::OfflineRunner._engine_for``).
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

#: Metrics the Intake Agent runs on every turn (see ``agents.yaml``).
#:
#: Correctly, these pass when the turn carries no clinical action: the agent asks
#: a question rather than diagnosing, and does not repeat an earlier question.
DEFAULT_METRICS: tuple[str, ...] = (
    "medical_boundary_compliance",
    "no_repeated_questions",
)


def register(registry: MetricRegistry) -> None:
    """Register the intake rules and their clinical aliases."""
    registry.register_from_checks(CHECKS, aliases=ALIASES)
