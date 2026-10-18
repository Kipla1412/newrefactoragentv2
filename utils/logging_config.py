"""Central logging configuration for the framework.

Configure once from an entrypoint, then obtain a logger the same way anywhere:

.. code-block:: python

    from utils.logging_config import get_logger

    logger = get_logger(__name__)

Library modules must **not** call :func:`setup_logging` themselves — they use
:func:`get_logger`, which configures logging on first use if nobody else has.
That keeps every module importable from a CLI, a server, or a test runner
without duplicate handlers or import-time side effects.

Configuration is environment-driven:

===========================  ==========================================  ===============
Variable                     Purpose                                     Default
===========================  ==========================================  ===============
``LOG_LEVEL``                Root level: DEBUG/INFO/WARNING/ERROR        ``INFO``
``LOG_FORMAT``               ``text`` or ``json``                        ``text``
``LOG_FILE``                 Optional extra file sink                    *(none)*
``LOG_THIRD_PARTY_LEVEL``    Level for noisy third-party loggers         ``WARNING``
``LOG_PROMPTS``              Log message content (see the PHI warning)   ``false``
===========================  ==========================================  ===============

.. warning::

   This is a clinical framework. Message content carries patient data, so
   content logging is **off** by default and gated behind ``LOG_PROMPTS``.
   Prefer :func:`describe_messages`, which logs structure and sizes only.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Mapping, Sequence

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "text"
DEFAULT_THIRD_PARTY_LEVEL = "WARNING"

TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party loggers that are chatty at INFO (LiteLLM logs every call) or
# simply not our code. Pinned down so framework records stay readable.
NOISY_LOGGERS: tuple[str, ...] = (
    "LiteLLM",
    "litellm",
    "openai",
    "anthropic",
    "httpx",
    "httpcore",
    "urllib3",
    "websockets",
    "watchfiles",
    "multipart",
    "python_multipart",
    "asyncio",
    "neo4j",
    "opensearch",
    "opensearchpy",
    "mlflow",
    "boto3",
    "botocore",
    "s3transfer",
)

# Keys whose values must never reach a log line.
SECRET_KEYS: frozenset[str] = frozenset({
    "api_key",
    "apikey",
    "key",
    "authorization",
    "auth",
    "auth_token",
    "access_token",
    "refresh_token",
    "token",
    "secret",
    "password",
    "cookie",
    "set-cookie",
})

_REDACTED = "***"
_configured = False


class JsonFormatter(logging.Formatter):
    """Render each record as a single JSON object.

    Structured context passed via ``extra={"fields": {...}}`` is flattened
    into the top level so log collectors can query those keys directly. The
    core keys (``timestamp``, ``level``, ``logger``, ``message``) always win,
    so a field can never spoof them.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            for key, value in fields.items():
                payload.setdefault(key, value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def coerce_level(value: str | int | None, default: int = logging.INFO) -> int:
    """Resolve a level name or number to a logging level.

    Args:
        value: Level name (``"debug"``, ``"INFO"``) or numeric level.
        default: Level to use when the value is missing or unrecognized.

    Returns:
        The numeric logging level.
    """
    if value is None:
        return default
    if isinstance(value, int):
        return value

    resolved = logging.getLevelName(str(value).strip().upper())
    return resolved if isinstance(resolved, int) else default


def prompts_enabled() -> bool:
    """Return whether LLM message content may be logged.

    Returns:
        True only when ``LOG_PROMPTS`` is set to a truthy value.
    """
    return os.environ.get("LOG_PROMPTS", "").strip().lower() in {"1", "true", "yes", "on"}


def redact(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of a mapping with secret-looking values masked.

    Args:
        mapping: Values that may be destined for a log line (headers, call
            kwargs, provider options).

    Returns:
        A shallow copy where any key matching :data:`SECRET_KEYS` — or
        containing ``key``/``token``/``secret``/``password`` — is masked.
    """
    safe: dict[str, Any] = {}

    for name, value in mapping.items():
        lowered = str(name).lower()
        if lowered in SECRET_KEYS or any(
            hint in lowered for hint in ("key", "token", "secret", "password")
        ):
            safe[name] = _REDACTED
        else:
            safe[name] = value

    return safe


def describe_messages(messages: Sequence[Mapping[str, Any]] | None) -> dict[str, Any]:
    """Summarize a message list without exposing its content.

    The PHI-safe alternative to logging prompts: structure, roles, and sizes
    only.

    Args:
        messages: Chat messages in provider format.

    Returns:
        A dict with ``message_count``, ``roles``, and ``content_chars``.
    """
    if not messages:
        return {"message_count": 0, "roles": [], "content_chars": 0}

    roles: list[str] = []
    content_chars = 0

    for message in messages:
        roles.append(str(message.get("role", "unknown")))
        content = message.get("content")
        if isinstance(content, str):
            content_chars += len(content)
        elif content is not None:
            content_chars += len(str(content))

    return {
        "message_count": len(messages),
        "roles": roles,
        "content_chars": content_chars,
    }


def setup_logging(
    level: str | int | None = None,
    fmt: str | None = None,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure logging for the process.

    Idempotent: the first call wins, so an entrypoint can call it without
    worrying about a module having already triggered configuration via
    :func:`get_logger`. Pass ``force=True`` to reconfigure (used by tests).

    Args:
        level: Override ``LOG_LEVEL``.
        fmt: Override ``LOG_FORMAT`` (``"text"`` or ``"json"``).
        log_file: Override ``LOG_FILE``. Pass ``""`` to disable the file sink.
        force: Reconfigure even if logging was already configured.
    """
    global _configured

    if _configured and not force:
        return

    resolved_level = coerce_level(level or os.environ.get("LOG_LEVEL"), logging.INFO)

    resolved_format = (fmt or os.environ.get("LOG_FORMAT") or DEFAULT_LOG_FORMAT).lower()
    formatter: logging.Formatter = (
        JsonFormatter() if resolved_format == "json" else logging.Formatter(TEXT_FORMAT, DATE_FORMAT)
    )

    resolved_file = log_file if log_file is not None else os.environ.get("LOG_FILE")

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    # Logs go to stderr so CLI stdout stays clean for program output.
    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if resolved_file:
        file_handler = logging.FileHandler(resolved_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.setLevel(resolved_level)

    third_party_level = coerce_level(
        os.environ.get("LOG_THIRD_PARTY_LEVEL"), logging.WARNING
    )
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module.

    Configures logging on first use, so a module imported outside an
    entrypoint still produces output instead of silently discarding records.

    Args:
        name: Logger name — conventionally ``__name__``.

    Returns:
        The module's logger.
    """
    setup_logging()
    return logging.getLogger(name)
