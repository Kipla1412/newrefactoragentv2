"""``SKILL.md`` parsing and metadata validation.

The format is a YAML frontmatter block delimited by ``---`` lines, followed by
the instruction body:

.. code-block:: markdown

    ---
    name: my-skill
    description: What this skill does.
    ---

    Instructions for the model.

Only the **first** closing ``---`` terminates the frontmatter, so ``---``
separators inside the instruction body are preserved.

Parsing is intentionally isolated from filesystem discovery: this module turns
text into validated value objects and knows nothing about skill directories.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from skills.errors import SkillNotFoundError, SkillValidationError
from skills.models import Skill, SkillMetadata

SKILL_FILE_NAME = "SKILL.md"
_FRONTMATTER_DELIMITER = "---"


def split_frontmatter(text: str) -> tuple[str, str]:
    """Split a ``SKILL.md`` file into its frontmatter and instruction body.

    Args:
        text: Full file contents.

    Returns:
        A ``(frontmatter, body)`` pair. The body is stripped.

    Raises:
        SkillValidationError: If the file has no frontmatter block, or the
            opening block is never closed.
    """
    lines = text.splitlines()

    start = next((i for i, line in enumerate(lines) if line.strip()), None)
    if start is None or lines[start].strip() != _FRONTMATTER_DELIMITER:
        raise SkillValidationError(
            "Missing YAML frontmatter: the file must begin with a '---' line"
        )

    for end in range(start + 1, len(lines)):
        if lines[end].strip() == _FRONTMATTER_DELIMITER:
            frontmatter = "\n".join(lines[start + 1 : end])
            body = "\n".join(lines[end + 1 :]).strip()
            return frontmatter, body

    raise SkillValidationError(
        "Malformed frontmatter: the opening '---' block is never closed"
    )


def parse_frontmatter(frontmatter: str, source: Path | str = "<string>") -> SkillMetadata:
    """Validate a frontmatter block and build its metadata.

    Args:
        frontmatter: Raw YAML text between the ``---`` delimiters.
        source: Path used in error messages.

    Returns:
        The validated metadata.

    Raises:
        SkillValidationError: If the YAML is unparseable, is not a mapping, or
            ``name`` / ``description`` are missing or not non-empty strings.
    """
    try:
        data = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise SkillValidationError(f"Invalid YAML frontmatter in {source}", cause=exc) from exc

    if not isinstance(data, dict):
        raise SkillValidationError(
            f"Frontmatter in {source} must be a YAML mapping with 'name' and 'description'"
        )

    return SkillMetadata(
        name=_require_text(data, "name", source),
        description=_require_text(data, "description", source),
        path=Path(source),
    )


def parse_skill_file(path: Path) -> Skill:
    """Read and fully parse a ``SKILL.md`` file, including its instructions.

    Args:
        path: Path to the ``SKILL.md`` file.

    Returns:
        The parsed skill.

    Raises:
        SkillNotFoundError: If the file does not exist.
        SkillValidationError: If the frontmatter is invalid.
    """
    if not path.is_file():
        raise SkillNotFoundError(f"Skill file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillNotFoundError(f"Could not read skill file: {path}", cause=exc) from exc

    frontmatter, body = split_frontmatter(text)
    return Skill(metadata=parse_frontmatter(frontmatter, path), instructions=body)


def parse_skill_metadata(path: Path) -> SkillMetadata:
    """Read only the metadata half of a ``SKILL.md`` file.

    Used during discovery so listing skills never depends on instruction size.

    Args:
        path: Path to the ``SKILL.md`` file.

    Returns:
        The validated metadata.

    Raises:
        SkillNotFoundError: If the file does not exist.
        SkillValidationError: If the frontmatter is invalid.
    """
    if not path.is_file():
        raise SkillNotFoundError(f"Skill file not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SkillNotFoundError(f"Could not read skill file: {path}", cause=exc) from exc

    frontmatter, _ = split_frontmatter(text)
    return parse_frontmatter(frontmatter, path)


def _require_text(data: dict, key: str, source: Path | str) -> str:
    """Return a required non-empty string field from parsed frontmatter.

    Args:
        data: Parsed frontmatter mapping.
        key: Field name to read.
        source: Path used in error messages.

    Returns:
        The stripped field value.

    Raises:
        SkillValidationError: If the field is absent, not a string, or blank.
    """
    value = data.get(key)
    if value is None:
        raise SkillValidationError(f"Missing required frontmatter field '{key}' in {source}")
    if not isinstance(value, str):
        raise SkillValidationError(
            f"Frontmatter field '{key}' in {source} must be a string"
        )
    if not value.strip():
        raise SkillValidationError(
            f"Frontmatter field '{key}' in {source} must not be empty"
        )
    return value.strip()
