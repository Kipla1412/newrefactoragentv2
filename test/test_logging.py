"""Tests for the reusable logging layer.

Coverage:

- logger acquisition and one-time configuration
- idempotent setup (no stacked handlers)
- ``LOG_LEVEL`` / ``LOG_FORMAT`` / ``LOG_FILE`` environment handling
- JSON formatter output and structured ``fields``
- secret redaction and PHI-safe message description
- the ``LOG_PROMPTS`` content gate
- a guard that core framework modules no longer use ``print``
"""

from __future__ import annotations

import ast
import io
import json
import logging
import sys
from pathlib import Path

import pytest

from utils import logging_config
from utils.logging_config import (
    JsonFormatter,
    coerce_level,
    describe_messages,
    get_logger,
    prompts_enabled,
    redact,
    setup_logging,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(autouse=True)
def _restore_logging():
    """Keep a test's logging configuration from leaking into other tests."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level

    yield

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)
    logging_config._configured = False


def configure(monkeypatch, **env) -> io.StringIO:
    """Reset logging and configure it from ``env``, capturing stderr.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        **env: Environment variables to set for this configuration.

    Returns:
        The buffer that will receive log output.
    """
    for key in ("LOG_LEVEL", "LOG_FORMAT", "LOG_FILE", "LOG_THIRD_PARTY_LEVEL", "LOG_PROMPTS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    logging_config._configured = False

    buffer = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buffer)
    setup_logging(force=True)
    return buffer


# ---------------------------------------------------------------------------
# Logger acquisition and configuration
# ---------------------------------------------------------------------------


def test_get_logger_returns_a_named_logger():
    logger = get_logger("test.some.module")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.some.module"


def test_get_logger_configures_logging_on_first_use():
    root = logging.getLogger()
    root.handlers.clear()
    logging_config._configured = False

    get_logger("test.autoconfig")

    assert root.handlers, "get_logger must configure logging so records are not lost"


def test_setup_logging_is_idempotent(monkeypatch):
    configure(monkeypatch, LOG_LEVEL="INFO")
    root = logging.getLogger()
    handlers = len(root.handlers)

    setup_logging()
    setup_logging()

    assert len(root.handlers) == handlers


def test_force_replaces_handlers_without_stacking_them(monkeypatch):
    configure(monkeypatch, LOG_LEVEL="INFO")

    setup_logging(force=True)
    setup_logging(force=True)

    assert len(logging.getLogger().handlers) == 1


# ---------------------------------------------------------------------------
# Levels and formats
# ---------------------------------------------------------------------------


def test_coerce_level_accepts_names_numbers_and_junk():
    assert coerce_level("debug") == logging.DEBUG
    assert coerce_level("WARNING") == logging.WARNING
    assert coerce_level(20) == logging.INFO
    assert coerce_level("not-a-level", default=logging.ERROR) == logging.ERROR
    assert coerce_level(None, default=logging.WARNING) == logging.WARNING


def test_log_level_env_var_is_respected(monkeypatch):
    buffer = configure(monkeypatch, LOG_LEVEL="WARNING")

    logger = get_logger("test.level")
    logger.info("should be filtered")
    logger.warning("should appear")

    output = buffer.getvalue()
    assert "should appear" in output
    assert "should be filtered" not in output


def test_text_format_includes_time_level_logger_and_message(monkeypatch):
    buffer = configure(monkeypatch, LOG_LEVEL="INFO")

    get_logger("test.text").info("hello %s", "world")

    output = buffer.getvalue()
    assert "hello world" in output
    assert "test.text" in output
    assert "INFO" in output
    assert "|" in output


def test_json_format_emits_one_object_per_record(monkeypatch):
    buffer = configure(monkeypatch, LOG_LEVEL="INFO", LOG_FORMAT="json")

    logger = get_logger("test.json")
    logger.info("first")
    logger.warning("second")

    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2

    payload = json.loads(lines[0])
    assert payload["message"] == "first"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.json"
    assert "timestamp" in payload
    assert json.loads(lines[1])["level"] == "WARNING"


def test_structured_fields_are_flattened_into_json_payload(monkeypatch):
    buffer = configure(monkeypatch, LOG_LEVEL="INFO", LOG_FORMAT="json")

    get_logger("test.fields").info(
        "Tool call", extra={"fields": {"tool": "wiki_search", "success": True}}
    )

    payload = json.loads(buffer.getvalue().strip())
    assert payload["tool"] == "wiki_search"
    assert payload["success"] is True
    assert payload["message"] == "Tool call"


def test_structured_fields_cannot_spoof_core_keys(monkeypatch):
    buffer = configure(monkeypatch, LOG_LEVEL="INFO", LOG_FORMAT="json")

    get_logger("test.spoof").info(
        "real message",
        extra={"fields": {"message": "spoofed", "level": "CRITICAL"}},
    )

    payload = json.loads(buffer.getvalue().strip())
    assert payload["message"] == "real message"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.spoof"


def test_json_formatter_records_exceptions(monkeypatch):
    buffer = configure(monkeypatch, LOG_LEVEL="INFO", LOG_FORMAT="json")

    try:
        raise ValueError("boom")
    except ValueError:
        get_logger("test.exc").exception("Failed")

    payload = json.loads(buffer.getvalue().strip())
    assert "ValueError: boom" in payload["exception"]


def test_json_formatter_handles_unserializable_field_values():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="obj", args=(), exc_info=None,
    )
    record.fields = {"path": Path("/tmp/example")}

    payload = json.loads(formatter.format(record))
    assert payload["path"] == "/tmp/example"


def test_log_file_sink_is_written(monkeypatch, tmp_path):
    log_path = tmp_path / "agent.log"
    configure(monkeypatch, LOG_LEVEL="INFO", LOG_FILE=str(log_path))

    get_logger("test.file").info("to the file")

    assert "to the file" in log_path.read_text(encoding="utf-8")


def test_noisy_third_party_loggers_are_pinned(monkeypatch):
    configure(monkeypatch, LOG_LEVEL="DEBUG", LOG_THIRD_PARTY_LEVEL="WARNING")

    assert logging.getLogger("litellm").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("neo4j").level == logging.WARNING


def test_third_party_level_can_be_raised(monkeypatch):
    configure(monkeypatch, LOG_LEVEL="DEBUG", LOG_THIRD_PARTY_LEVEL="ERROR")

    assert logging.getLogger("litellm").level == logging.ERROR


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------


def test_redact_masks_secret_looking_values():
    safe = redact({
        "model": "gpt-4.1",
        "api_key": "sk-real-key",
        "Authorization": "Bearer abc",
        "auth_token": "t",
        "X-Api-Key": "k",
        "count": 3,
    })

    assert safe["model"] == "gpt-4.1"
    assert safe["count"] == 3
    for key in ("api_key", "Authorization", "auth_token", "X-Api-Key"):
        assert safe[key] == "***"


def test_describe_messages_returns_structure_not_content():
    summary = describe_messages([
        {"role": "system", "content": "Patient Jane Doe has diabetes"},
        {"role": "user", "content": "What medications?"},
    ])

    assert summary["message_count"] == 2
    assert summary["roles"] == ["system", "user"]
    assert summary["content_chars"] == len("Patient Jane Doe has diabetes") + len(
        "What medications?"
    )
    assert "Jane Doe" not in json.dumps(summary)


def test_describe_messages_handles_empty_input():
    assert describe_messages(None) == {
        "message_count": 0,
        "roles": [],
        "content_chars": 0,
    }
    assert describe_messages([])["message_count"] == 0


def test_prompts_enabled_defaults_to_off(monkeypatch):
    monkeypatch.delenv("LOG_PROMPTS", raising=False)

    assert prompts_enabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_prompts_enabled_accepts_truthy_values(monkeypatch, value):
    monkeypatch.setenv("LOG_PROMPTS", value)

    assert prompts_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_prompts_enabled_rejects_falsy_values(monkeypatch, value):
    monkeypatch.setenv("LOG_PROMPTS", value)

    assert prompts_enabled() is False


# ---------------------------------------------------------------------------
# The print -> logging conversion guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path,attr",
    [
        ("agent/agent.py", "agent.agent"),
        ("client/llm_client.py", "client.llm_client"),
        ("safety/approval.py", "safety.approval"),
        ("tools/registry.py", "tools.registry"),
        ("utils/mlflow_tracker.py", "utils.mlflow_tracker"),
        ("customagents/voiceagent/voiceagent.py", "customagents.voiceagent.voiceagent"),
        ("customagents/voiceagent/utterance.py", "customagents.voiceagent.utterance"),
        (
            "speechtospeech/providers/stt/streamsarvam.py",
            "speechtospeech.providers.stt.streamsarvam",
        ),
        (
            "speechtospeech/providers/tts/streamsarvam.py",
            "speechtospeech.providers.tts.streamsarvam",
        ),
        ("api/wsrouters/webs2s.py", "api.wsrouters.webs2s"),
    ],
)
def test_core_modules_expose_a_module_logger(module_path, attr):
    """Each converted module publishes a module-level ``logger``."""
    import importlib

    module = importlib.import_module(attr)

    assert isinstance(module.logger, logging.Logger)
    assert module.logger.name == attr


FRAMEWORK_PACKAGES = (
    "agent",
    "api",
    "client",
    "config",
    "context",
    "customagents",
    "hooks",
    "prompts",
    "registries",
    "safety",
    "skills",
    "speechtospeech",
    "tools",
    "utils",
)


def test_framework_modules_obtain_loggers_through_get_logger():
    """The documented convention, enforced.

    ``utils/logging_config.py`` is the one module allowed to touch stdlib
    logging; everything else goes through ``get_logger`` so configuration
    stays in a single place.
    """
    offenders = []

    for package in FRAMEWORK_PACKAGES:
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            if path.name == "logging_config.py":
                continue
            if "logging.getLogger(" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert offenders == [], (
        "these modules bypass utils.logging_config.get_logger: " + ", ".join(offenders)
    )


def test_no_framework_module_uses_print():
    """Diagnostics must go through logging so they can be leveled and filtered.

    This walks whole packages rather than a hand-maintained file list. A
    per-file allowlist silently stops covering modules that are added later —
    or whose conversion is reverted — which is exactly how the Sarvam voice
    providers ended up with prints while the suite stayed green.
    """
    offenders: list[str] = []

    for package in FRAMEWORK_PACKAGES:
        for path in sorted((PROJECT_ROOT / package).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))

            lines = sorted(
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            )

            if lines:
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}: {lines}")

    assert offenders == [], "print() is still used in: " + "; ".join(offenders)
