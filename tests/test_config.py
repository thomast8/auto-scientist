"""Tests for domain configuration schema."""

import logging

import pytest
from pydantic import ValidationError

from auto_scientist.config import DomainConfig


class TestDomainConfig:
    def test_required_fields(self):
        dc = DomainConfig(
            name="test", description="Test domain",
            data_paths=["data.csv"],
        )
        assert dc.name == "test"
        assert dc.data_paths == ["data.csv"]

    def test_defaults(self):
        dc = DomainConfig(name="t", description="d", data_paths=[])
        assert dc.run_command == "uv run {script_path}"
        assert dc.run_cwd == "."
        assert dc.run_timeout_minutes == 120
        assert dc.version_prefix == "v"
        assert dc.protected_paths == []
        assert not hasattr(dc, "success_criteria") or "success_criteria" not in dc.model_fields
        assert not hasattr(dc, "domain_knowledge") or "domain_knowledge" not in dc.model_fields

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DomainConfig(name="t")  # missing description and data_paths

    def test_no_success_criteria_field(self):
        """DomainConfig should not have success_criteria (moved to ExperimentState)."""
        assert "success_criteria" not in DomainConfig.model_fields

    def test_no_domain_knowledge_field(self):
        """DomainConfig should not have domain_knowledge (moved to ExperimentState)."""
        assert "domain_knowledge" not in DomainConfig.model_fields


class TestDomainConfigSerialization:
    def test_roundtrip(self):
        dc = DomainConfig(
            name="test", description="Test domain",
            data_paths=["a.csv", "b.csv"],
            run_command="python {script_path}",
            protected_paths=["src/"],
        )
        json_str = dc.model_dump_json()
        loaded = DomainConfig.model_validate_json(json_str)
        assert loaded.name == "test"
        assert loaded.data_paths == ["a.csv", "b.csv"]
        assert loaded.run_command == "python {script_path}"
        assert loaded.protected_paths == ["src/"]

    def test_custom_run_command(self):
        dc = DomainConfig(
            name="t", description="d", data_paths=[],
            run_command="python3 -u {script_path}",
        )
        assert dc.run_command == "python3 -u {script_path}"

    def test_protected_paths_list(self):
        dc = DomainConfig(
            name="t", description="d", data_paths=[],
            protected_paths=["src/", "data/"],
        )
        assert dc.protected_paths == ["src/", "data/"]

    def test_no_experiment_dependencies_field(self):
        """experiment_dependencies removed; scripts declare deps via PEP 723 inline metadata."""
        assert "experiment_dependencies" not in DomainConfig.model_fields


class TestDataPathsCoercion:
    def test_dict_data_paths_coerced_to_list(self):
        dc = DomainConfig(
            name="t", description="d",
            data_paths={"file1": "a.csv", "file2": "b.csv"},
        )
        assert dc.data_paths == ["a.csv", "b.csv"]

    def test_dict_data_paths_logs_deprecation_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="auto_scientist.config"):
            DomainConfig(
                name="t", description="d",
                data_paths={"file1": "a.csv"},
            )
        assert any("deprecat" in msg.lower() for msg in caplog.messages)

    def test_list_data_paths_no_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="auto_scientist.config"):
            DomainConfig(
                name="t", description="d",
                data_paths=["a.csv"],
            )
        assert not any("deprecat" in msg.lower() for msg in caplog.messages)
