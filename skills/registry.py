"""Skill discovery and on-demand loading.

This module owns the filesystem side of skills. It is deliberately independent
of :class:`~context.manager.ContextManager` and of the agent runtime: the
registry only knows how to find skills and hand back their text, and callers
decide what to do with that text.

A skill is a directory containing a ``SKILL.md`` file. Anything else in the
skills root is ignored.

Security note: skill contents are prompt text, never executable code. Nothing
here imports, evaluates, or executes a skill, and loading a skill grants no
tool permissions — a skill that mentions a tool does not enable it.
"""

from __future__ import annotations

from pathlib import Path

from skills.errors import SkillNotFoundError, SkillValidationError
from skills.frontmatter import (
    SKILL_FILE_NAME,
    parse_skill_file,
    parse_skill_metadata,
)
from skills.models import Skill, SkillMetadata

# Directory bundled with the framework, scanned by default.
DEFAULT_SKILLS_ROOT = Path(__file__).resolve().parent / "library"


class SkillRegistry:
    """Discover skills under a root directory and load them by name.

    Discovery reads only frontmatter and is cached; instructions are read from
    disk on every :meth:`load`, so editing a ``SKILL.md`` takes effect without
    a restart.

    Attributes:
        root: Directory scanned for skill subdirectories.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        """Initialize the registry.

        Args:
            root: Directory containing one subdirectory per skill. Defaults to
                the bundled skills library.
        """
        self.root = Path(root) if root is not None else DEFAULT_SKILLS_ROOT
        self._metadata: dict[str, SkillMetadata] | None = None

    @classmethod
    def from_builtin(cls) -> "SkillRegistry":
        """Build a registry backed by the framework's bundled skills.

        Returns:
            A registry scanning :data:`DEFAULT_SKILLS_ROOT`.
        """
        return cls(DEFAULT_SKILLS_ROOT)

    def discover(self) -> dict[str, SkillMetadata]:
        """Scan the root for skills and return their metadata by name.

        A missing root is not an error — it simply yields no skills. A
        directory without a ``SKILL.md`` is not a skill and is skipped. A
        malformed ``SKILL.md`` is reported rather than ignored, because
        silently dropping it would hide a broken skill.

        Returns:
            A copy of the name → metadata mapping.

        Raises:
            SkillValidationError: If a ``SKILL.md`` is malformed, or two
                skills declare the same ``name``.
        """
        if self._metadata is None:
            self._metadata = self._scan()

        return dict(self._metadata)

    def refresh(self) -> dict[str, SkillMetadata]:
        """Discard the cached discovery result and scan again.

        Returns:
            The freshly discovered name → metadata mapping.

        Raises:
            SkillValidationError: If a ``SKILL.md`` is malformed.
        """
        self._metadata = None
        return self.discover()

    def list_skills(self) -> list[SkillMetadata]:
        """Return metadata for every discovered skill, ordered by name.

        Returns:
            Skill metadata sorted by ``name``.

        Raises:
            SkillValidationError: If a ``SKILL.md`` is malformed.
        """
        return [self.discover()[name] for name in sorted(self.discover())]

    def has(self, name: str) -> bool:
        """Return whether a skill is discoverable.

        Args:
            name: Skill identifier.

        Returns:
            True if the skill exists.

        Raises:
            SkillValidationError: If a ``SKILL.md`` is malformed.
        """
        return name in self.discover()

    def get_metadata(self, name: str) -> SkillMetadata:
        """Return a skill's metadata without loading its instructions.

        Args:
            name: Skill identifier.

        Returns:
            The skill's metadata.

        Raises:
            SkillNotFoundError: If no such skill was discovered.
            SkillValidationError: If a ``SKILL.md`` is malformed.
        """
        metadata = self.discover().get(name)
        if metadata is None:
            raise SkillNotFoundError(self._unknown_skill_message(name))
        return metadata

    def load(self, name: str) -> Skill:
        """Load a skill's metadata and instructions from disk.

        Args:
            name: Skill identifier.

        Returns:
            The fully loaded skill.

        Raises:
            SkillNotFoundError: If no such skill was discovered, or its
                ``SKILL.md`` disappeared since discovery.
            SkillValidationError: If the ``SKILL.md`` is malformed.
        """
        metadata = self.get_metadata(name)
        return parse_skill_file(metadata.path)

    def load_instructions(self, name: str) -> str:
        """Load only a skill's instruction body.

        Args:
            name: Skill identifier.

        Returns:
            The skill's instructions.

        Raises:
            SkillNotFoundError: If the skill does not exist.
            SkillValidationError: If the ``SKILL.md`` is malformed.
        """
        return self.load(name).instructions

    def _scan(self) -> dict[str, SkillMetadata]:
        """Walk the root and parse the metadata of every skill directory.

        Returns:
            A name → metadata mapping.

        Raises:
            SkillValidationError: If a ``SKILL.md`` is malformed or the same
                skill name appears twice.
        """
        discovered: dict[str, SkillMetadata] = {}

        if not self.root.is_dir():
            return discovered

        for entry in sorted(self.root.iterdir()):
            skill_file = entry / SKILL_FILE_NAME
            if not entry.is_dir() or not skill_file.is_file():
                continue

            metadata = parse_skill_metadata(skill_file)
            if metadata.name in discovered:
                raise SkillValidationError(
                    f"Duplicate skill name '{metadata.name}' in "
                    f"{metadata.path} and {discovered[metadata.name].path}"
                )
            discovered[metadata.name] = metadata

        return discovered

    def _unknown_skill_message(self, name: str) -> str:
        """Build a helpful error message for an unknown skill.

        Args:
            name: The requested skill identifier.

        Returns:
            The error message, listing what is available.
        """
        available = ", ".join(sorted(self.discover())) or "none"
        return f"Unknown skill '{name}'. Available skills: {available}"


_default_registry: SkillRegistry | None = None


def default_skill_registry() -> SkillRegistry:
    """Return the shared registry backed by the bundled skills library.

    Cached process-wide so the bundled directory is scanned once.

    Returns:
        The shared :class:`SkillRegistry`.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = SkillRegistry.from_builtin()
    return _default_registry
