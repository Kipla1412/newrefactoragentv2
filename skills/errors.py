"""Skill subsystem error types.

Skill failures reuse the framework's :class:`~utils.errors.AgentError` base so
they carry the same ``message`` / ``details`` / ``cause`` contract as the rest
of the codebase.
"""

from __future__ import annotations

from utils.errors import AgentError


class SkillError(AgentError):
    """Base class for skill discovery and loading failures."""


class SkillNotFoundError(SkillError):
    """Raised when a requested skill cannot be found.

    Covers both "no such skill directory" and "the ``SKILL.md`` disappeared
    between discovery and loading".
    """


class SkillValidationError(SkillError):
    """Raised when a ``SKILL.md`` exists but its metadata is unusable.

    Examples: missing YAML frontmatter, unparseable YAML, a missing/empty
    ``name`` or ``description``, or two skills claiming the same name.
    """
