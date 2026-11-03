"""Offline evaluation dataset for the intake agent.

The dataset is authored as "given this conversation so far, what should the next
reply be?":

.. code-block:: json

    {
      "dataset_name": "intake_agent_offline_evaluation",
      "version": "1.0",
      "description": "...",
      "agent": "Intake Agent",
      "test_cases": [
        {
          "test_case_id": "INTAKE_001",
          "scenario": "Normal intake opening",
          "context": ["Assistant: What's your full name?"],
          "input": "I'm Maya Raman.",
          "expected_output": "Acknowledge and ask one next question.",
          "expected_behaviour": ["Ask only one question"],
          "critical_checks": ["one_question_only", "no_repeated_questions"]
        }
      ]
    }

``context`` carries the turns *before* ``input``. Lines prefixed ``Assistant:``
/ ``Agent:`` replay as assistant turns and ``Patient:`` / ``User:`` as user
turns, so the adapter can rebuild the state the case assumes. Lines without a
recognised prefix are prose state descriptions and are skipped — such a case is
reported as not fully replayable rather than being silently mis-seeded.

``critical_checks`` is the assertion vocabulary the deterministic validators are
keyed on (e.g. ``one_question_only``, ``no_hallucination``,
``emergency_escalation``).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, model_validator

from utils.errors import AgentError

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "intake_dataset.json"

#: Prefix recognised when replaying ``context`` into a session.
_ROLE_PREFIXES: dict[str, str] = {
    "assistant:": "assistant",
    "agent:": "assistant",
    "patient:": "user",
    "user:": "user",
}


class DatasetError(AgentError):
    """Raised when the evaluation dataset cannot be loaded or is invalid."""


class EvalCase(BaseModel):
    """One evaluation case: a conversation context plus the next patient turn.

    Attributes:
        test_case_id: Stable identifier, unique within the dataset.
        scenario: Short human label for what the case probes.
        context: The conversation before ``input``.
        input: The patient's next message — the turn being evaluated.
        expected_output: Natural-language description of a correct reply.
        expected_behaviour: Prose behaviours the reply should exhibit.
        critical_checks: Named assertions the deterministic validators run.
    """

    test_case_id: str
    scenario: str
    context: list[str] = Field(default_factory=list)
    input: str
    expected_output: str
    expected_behaviour: list[str] = Field(default_factory=list)
    critical_checks: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "EvalCase":
        if not self.test_case_id.strip():
            raise ValueError("test_case_id must not be empty")
        if not self.scenario.strip():
            raise ValueError(f"{self.test_case_id}: scenario must not be empty")
        if not self.input.strip():
            raise ValueError(f"{self.test_case_id}: input must not be empty")
        if not self.expected_output.strip():
            raise ValueError(f"{self.test_case_id}: expected_output must not be empty")
        if any(not line.strip() for line in self.context):
            raise ValueError(f"{self.test_case_id}: context contains an empty line")
        return self

    @property
    def id(self) -> str:
        """The case identifier."""
        return self.test_case_id

    @property
    def task(self) -> str:
        """Short task label, used as the LLM-judged task description."""
        return self.scenario

    @property
    def prompts(self) -> list[str]:
        """The user turn to send to the agent."""
        return [self.input]

    @property
    def conversation(self) -> list[tuple[str, str]]:
        """Parse ``context`` into ``(role, text)`` pairs.

        Only lines with a recognised ``Assistant:`` / ``Patient:`` prefix are
        returned; prose state descriptions are skipped.
        """
        replayable: list[tuple[str, str]] = []

        for line in self.context:
            lowered = line.lower()
            for prefix, role in _ROLE_PREFIXES.items():
                if lowered.startswith(prefix):
                    text = line[len(prefix):].strip()
                    if text:
                        replayable.append((role, text))
                    break

        return replayable

    @property
    def is_replayable(self) -> bool:
        """Whether every non-empty context line can be replayed as a turn."""
        if not self.context:
            return True
        return len(self.conversation) == len(self.context)


class EvalDataset(BaseModel):
    """A named collection of evaluation cases.

    Attributes:
        dataset_name: Identifier for the dataset.
        version: Author-supplied version string.
        description: What the dataset covers.
        agent: Agent the dataset targets.
        test_cases: The evaluation cases.
    """

    dataset_name: str = "intake_offline_evaluation"
    version: str = "1.0"
    description: str | None = None
    agent: str = "intake"
    test_cases: list[EvalCase]

    @model_validator(mode="after")
    def _validate(self) -> "EvalDataset":
        if not self.test_cases:
            raise ValueError("dataset contains no test cases")

        seen: set[str] = set()
        for case in self.test_cases:
            if case.id in seen:
                raise ValueError(f"duplicate test_case_id '{case.id}'")
            seen.add(case.id)

        return self

    @property
    def cases(self) -> list[EvalCase]:
        """Alias for :attr:`test_cases`."""
        return self.test_cases

    def by_id(self, case_id: str) -> EvalCase:
        """Return one case by id.

        Args:
            case_id: The case identifier.

        Returns:
            The matching case.

        Raises:
            DatasetError: If no case has that id.
        """
        for case in self.test_cases:
            if case.id == case_id:
                return case
        raise DatasetError(f"Unknown case '{case_id}'")

    def critical_checks(self) -> set[str]:
        """Return every distinct critical check named by the dataset."""
        return {check for case in self.test_cases for check in case.critical_checks}

    def not_replayable(self) -> list[str]:
        """Return ids of cases whose context cannot be fully replayed."""
        return [case.id for case in self.test_cases if not case.is_replayable]


def load_dataset(path: str | Path | None = None) -> EvalDataset:
    """Load and validate an evaluation dataset.

    Args:
        path: Path to the JSON file. Defaults to the bundled intake dataset.

    Returns:
        The validated dataset.

    Raises:
        DatasetError: If the file is missing, is not valid JSON, or fails
            validation.
    """
    resolved = Path(path) if path is not None else DEFAULT_DATASET_PATH
    display = str(resolved)

    try:
        raw = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"Could not read dataset at {display}", cause=exc) from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"Invalid JSON in dataset {display}: {exc}") from exc

    if not isinstance(payload, dict):
        raise DatasetError(f"Invalid dataset {display}: expected a JSON object")

    try:
        return EvalDataset(**payload)
    except ValidationError as exc:
        raise DatasetError(
            f"Invalid dataset {display}: {exc.error_count()} problem(s)",
            details={"errors": exc.errors(include_url=False)},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"Invalid dataset {display}: {exc}") from exc
