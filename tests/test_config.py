"""Tests for domain configuration schema."""

import pytest
from pydantic import ValidationError

from auto_scientist.config import DomainConfig, SuccessCriterion


class TestSuccessCriterion:
    def test_required_fields(self):
        sc = SuccessCriterion(name="acc", description="accuracy", metric_key="accuracy")
        assert sc.name == "acc"
        assert sc.description == "accuracy"
        assert sc.metric_key == "accuracy"

    def test_defaults(self):
        sc = SuccessCriterion(name="a", description="b", metric_key="c")
        assert sc.target_min is None
        assert sc.target_max is None
        assert sc.required is True

    def test_optional_targets(self):
        sc = SuccessCriterion(
            name="a", description="b", metric_key="c",
            target_min=0.5, target_max=1.0, required=False,
        )
        assert sc.target_min == 0.5
        assert sc.target_max == 1.0
        assert sc.required is False

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SuccessCriterion(name="a", description="b")  # missing metric_key


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
        assert dc.run_command == "uv run python -u {script_path}"
        assert dc.run_cwd == "."
        assert dc.run_timeout_minutes == 120
        assert dc.version_prefix == "v"
        assert dc.success_criteria == []
        assert dc.domain_knowledge == ""
        assert dc.protected_paths == []
        assert dc.experiment_dependencies == []

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DomainConfig(name="t")  # missing description and data_paths

    def test_with_success_criteria(self):
        sc = SuccessCriterion(name="a", description="b", metric_key="c")
        dc = DomainConfig(
            name="t", description="d", data_paths=[],
            success_criteria=[sc],
        )
        assert len(dc.success_criteria) == 1
        assert dc.success_criteria[0].name == "a"
