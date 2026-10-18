"""Reusable skills for the agent framework.

A skill is a directory containing a ``SKILL.md`` file — YAML frontmatter with
``name`` and ``description``, followed by an instruction body. Skills are
prompt content, never executable code.

Typical use:

.. code-block:: python

    from skills import default_skill_registry

    registry = default_skill_registry()
    skill = registry.load("clinical-followup-questions")

    with session.context_manager.use_skill(skill.instructions):
        ...  # instructions are now part of the system message

Discovery, parsing, and loading live here; how instructions reach the model is
the :class:`~context.manager.ContextManager`'s job.
"""

from skills.errors import SkillError, SkillNotFoundError, SkillValidationError
from skills.frontmatter import (
    SKILL_FILE_NAME,
    parse_frontmatter,
    parse_skill_file,
    parse_skill_metadata,
    split_frontmatter,
)
from skills.models import Skill, SkillMetadata
from skills.registry import (
    DEFAULT_SKILLS_ROOT,
    SkillRegistry,
    default_skill_registry,
)

__all__ = [
    "DEFAULT_SKILLS_ROOT",
    "SKILL_FILE_NAME",
    "Skill",
    "SkillError",
    "SkillMetadata",
    "SkillNotFoundError",
    "SkillRegistry",
    "SkillValidationError",
    "default_skill_registry",
    "parse_frontmatter",
    "parse_skill_file",
    "parse_skill_metadata",
    "split_frontmatter",
]
