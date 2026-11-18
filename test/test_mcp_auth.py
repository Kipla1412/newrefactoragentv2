"""MCP credentials: whose token reaches the FHIR server.

The MCP connection used to send *this API's* IAM token to the FHIR server, which
rejects it (401) — the token authenticates a user here, not there. The failure was
then logged as a warning and the agent ran on without any tools, which is why a
request could return 200 while every tool call failed.
"""

from __future__ import annotations

import os

import pytest

from tools.mcp.mcp_manager import _merge_headers


class TestHeaderMerge:
    def test_a_request_token_is_forwarded_when_nothing_is_configured(self):
        assert _merge_headers(None, "iam-token") == {
            "Authorization": "Bearer iam-token"
        }

    def test_a_configured_authorization_wins(self):
        """The server's own credential must not be replaced by ours."""
        merged = _merge_headers(
            {"Authorization": "Bearer fhir-token"}, "iam-token"
        )

        assert merged["Authorization"] == "Bearer fhir-token"

    def test_other_configured_headers_survive(self):
        merged = _merge_headers({"X-API-Key": "k"}, "iam-token")

        assert merged == {"X-API-Key": "k", "Authorization": "Bearer iam-token"}

    def test_the_configured_mapping_is_not_mutated(self):
        configured = {"X-API-Key": "k"}

        _merge_headers(configured, "iam-token")

        assert configured == {"X-API-Key": "k"}

    def test_no_token_and_no_headers_is_empty(self):
        assert _merge_headers(None, None) == {}


class TestConfiguredFhirHeaders:
    """FHIR_MCP_HEADERS lets the server's own credential be configured."""

    @staticmethod
    def _config(monkeypatch, headers_json: str | None):
        from config.config import Config

        monkeypatch.setenv("FHIR_BASE_URL", "https://fhir.example/mcp")
        if headers_json is None:
            monkeypatch.delenv("FHIR_MCP_HEADERS", raising=False)
        else:
            monkeypatch.setenv("FHIR_MCP_HEADERS", headers_json)
        return Config()

    def test_no_variable_means_no_headers(self, monkeypatch):
        assert self._config(monkeypatch, None).fhir_mcp_headers == {}

    def test_json_headers_are_parsed(self, monkeypatch):
        config = self._config(
            monkeypatch, '{"Authorization": "Bearer fhir-token"}'
        )

        assert config.fhir_mcp_headers == {"Authorization": "Bearer fhir-token"}

    def test_they_reach_the_server_configuration(self, monkeypatch):
        config = self._config(
            monkeypatch, '{"Authorization": "Bearer fhir-token"}'
        )

        assert config.mcp_servers_config["fhir"].headers == {
            "Authorization": "Bearer fhir-token"
        }

    def test_invalid_json_is_ignored_not_fatal(self, monkeypatch):
        config = self._config(monkeypatch, "not json")

        assert config.fhir_mcp_headers == {}

    def test_a_non_object_is_ignored(self, monkeypatch):
        config = self._config(monkeypatch, '["Bearer x"]')

        assert config.fhir_mcp_headers == {}

    def test_no_fhir_url_means_no_servers(self, monkeypatch):
        from config.config import Config

        monkeypatch.delenv("FHIR_BASE_URL", raising=False)

        assert Config().mcp_servers_config == {}
