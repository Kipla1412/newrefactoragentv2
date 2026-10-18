"""Immutable value objects returned by the skill subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillMetadata:
    """Metadata parsed from a skill's YAML frontmatter.

    This is the cheap half of a skill: everything needed to list or select a
    skill without reading its instructions.

    Attributes:
        name: Unique skill identifier, from the ``name`` frontmatter field.
        description: Short summary, from the ``description`` frontmatter field.
        path: Path to the skill's ``SKILL.md`` file.
    """

    name: str
    description: str
    path: Path

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable view of the metadata.

        Returns:
            The metadata without filesystem details.
        """
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class Skill:
    """A skill's metadata plus its loaded instructions.

    Instructions are plain text. They are treated strictly as prompt content:
    the framework never imports, evaluates, or executes anything read from a
    skill file.

    Attributes:
        metadata: The parsed frontmatter.
        instructions: The markdown body of ``SKILL.md``.
    """

    metadata: SkillMetadata
    instructions: str

    @property
    def name(self) -> str:
        """The skill's unique identifier."""
        return self.metadata.name

    @property
    def description(self) -> str:
        """The skill's short summary."""
        return self.metadata.description

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable view including the instructions.

        Returns:
            Metadata plus the ``instructions`` body.
        """
        return {**self.metadata.to_dict(), "instructions": self.instructions}
