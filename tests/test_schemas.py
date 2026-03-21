"""Tests for Pydantic output schemas (agent output validation)."""

import pytest
from pydantic import ValidationError

from auto_scientist.schemas import (
    AnalystOutput,
    CriteriaRevisionOutput,
    CriterionDefinition,
    CriterionResult,
    IterationCriterionResult,
    PlanChange,
    ScientistPlanOutput,
)


# ---------------------------------------------------------------------------
# CriterionResult
# ---------------------------------------------------------------------------

class TestCriterionResult:
    def test_valid(self):
        r = CriterionResult(
            name="accuracy", measured_value=0.95, target=">= 0.9", status="pass",
        )
        assert r.name == "accuracy"
        assert r.measured_value == 0.95
        assert r.status == "pass"

    def test_measured_value_string(self):
        r = CriterionResult(
            name="accuracy", measured_value="0.95", target=">= 0.9", status="pass",
        )
        assert r.measured_value == "0.95"

    def test_measured_value_none(self):
        r = CriterionResult(
            name="accuracy", measured_value=None, target=">= 0.9", status="unable_to_measure",
        )
        assert r.measured_value is None

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            CriterionResult(
                name="accuracy", measured_value=0.95, target=">= 0.9", status="maybe",
            )

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            CriterionResult(name="accuracy", measured_value=0.95, target=">= 0.9")

    def test_extra_fields_ignored(self):
        r = CriterionResult(
            name="accuracy", measured_value=0.95, target=">= 0.9",
            status="pass", extra_field="should be ignored",
        )
        assert not hasattr(r, "extra_field")


# ---------------------------------------------------------------------------
# IterationCriterionResult
# ---------------------------------------------------------------------------

class TestIterationCriterionResult:
    def test_valid(self):
        r = IterationCriterionResult(name="loss_decreases", status="pass", measured_value="0.01")
        assert r.status == "pass"

    def test_invalid_status(self):
        with pytest.raises(ValidationError):
            IterationCriterionResult(name="x", status="unknown", measured_value="0")


# ---------------------------------------------------------------------------
# AnalystOutput
# ---------------------------------------------------------------------------

class TestAnalystOutput:
    @pytest.fixture
    def minimal_valid(self):
        return {
            "criteria_results": [],
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["data loaded"],
            "iteration_criteria_results": [],
        }

    def test_minimal_valid(self, minimal_valid):
        a = AnalystOutput.model_validate(minimal_valid)
        assert a.success_score is None
        assert a.domain_knowledge == ""
        assert a.data_summary is None

    def test_full_iteration_0(self, minimal_valid):
        minimal_valid["domain_knowledge"] = "This is SpO2 pulse oximetry data"
        minimal_valid["data_summary"] = {"rows": 1000, "columns": 5}
        a = AnalystOutput.model_validate(minimal_valid)
        assert a.domain_knowledge == "This is SpO2 pulse oximetry data"
        assert a.data_summary == {"rows": 1000, "columns": 5}

    def test_with_criteria_results(self, minimal_valid):
        minimal_valid["criteria_results"] = [
            {"name": "accuracy", "measured_value": 0.95, "target": ">= 0.9", "status": "pass"},
            {"name": "latency", "measured_value": None, "target": "< 500", "status": "unable_to_measure"},
        ]
        minimal_valid["success_score"] = 50
        a = AnalystOutput.model_validate(minimal_valid)
        assert len(a.criteria_results) == 2
        assert a.success_score == 50

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AnalystOutput.model_validate({"criteria_results": [], "key_metrics": {}})

    def test_extra_fields_ignored(self, minimal_valid):
        minimal_valid["unexpected_key"] = "should be ignored"
        a = AnalystOutput.model_validate(minimal_valid)
        dumped = a.model_dump()
        assert "unexpected_key" not in dumped

    def test_model_dump_roundtrip(self, minimal_valid):
        """model_dump() should produce a plain dict usable with .get()."""
        a = AnalystOutput.model_validate(minimal_valid)
        d = a.model_dump()
        assert isinstance(d, dict)
        assert d.get("observations") == ["data loaded"]
        assert d.get("success_score") is None


# ---------------------------------------------------------------------------
# PlanChange
# ---------------------------------------------------------------------------

class TestPlanChange:
    def test_valid(self):
        c = PlanChange(what="add regularization", why="prevent overfitting", how="L2 penalty", priority=1)
        assert c.priority == 1

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            PlanChange(what="add regularization", why="prevent overfitting", priority=1)


# ---------------------------------------------------------------------------
# CriterionDefinition
# ---------------------------------------------------------------------------

class TestCriterionDefinition:
    def test_valid(self):
        c = CriterionDefinition(
            name="accuracy", description="Model accuracy", metric_key="accuracy", condition=">= 0.9",
        )
        assert c.condition == ">= 0.9"


# ---------------------------------------------------------------------------
# CriteriaRevisionOutput
# ---------------------------------------------------------------------------

class TestCriteriaRevisionOutput:
    def test_valid(self):
        r = CriteriaRevisionOutput(
            changes="Relaxed accuracy threshold",
            revised_criteria=[
                {"name": "accuracy", "description": "Model accuracy", "metric_key": "accuracy", "condition": ">= 0.8"},
            ],
        )
        assert len(r.revised_criteria) == 1


# ---------------------------------------------------------------------------
# ScientistPlanOutput
# ---------------------------------------------------------------------------

class TestScientistPlanOutput:
    @pytest.fixture
    def minimal_valid(self):
        return {
            "hypothesis": "Adding regularization will reduce overfitting",
            "strategy": "incremental",
            "changes": [
                {"what": "add L2", "why": "reduce overfitting", "how": "sklearn param", "priority": 1},
            ],
            "expected_impact": "Lower test error",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "## Iteration 1\nAdding L2 regularization.",
            "success_criteria": [
                {"name": "test_error", "description": "Test error", "metric_key": "test_error", "condition": "< 0.1"},
            ],
        }

    def test_minimal_valid(self, minimal_valid):
        p = ScientistPlanOutput.model_validate(minimal_valid)
        assert p.hypothesis == "Adding regularization will reduce overfitting"
        assert p.strategy == "incremental"
        assert len(p.changes) == 1
        assert p.should_stop is False
        assert p.top_level_criteria is None
        assert p.criteria_revision is None

    def test_with_top_level_criteria(self, minimal_valid):
        minimal_valid["top_level_criteria"] = [
            {"name": "accuracy", "description": "Overall accuracy", "metric_key": "accuracy", "condition": ">= 0.9"},
        ]
        p = ScientistPlanOutput.model_validate(minimal_valid)
        assert len(p.top_level_criteria) == 1

    def test_with_criteria_revision(self, minimal_valid):
        minimal_valid["criteria_revision"] = {
            "changes": "Relaxed threshold",
            "revised_criteria": [
                {"name": "accuracy", "description": "Accuracy", "metric_key": "accuracy", "condition": ">= 0.8"},
            ],
        }
        p = ScientistPlanOutput.model_validate(minimal_valid)
        assert p.criteria_revision.changes == "Relaxed threshold"

    def test_invalid_strategy(self, minimal_valid):
        minimal_valid["strategy"] = "random"
        with pytest.raises(ValidationError):
            ScientistPlanOutput.model_validate(minimal_valid)

    def test_should_stop_true_with_reason(self, minimal_valid):
        minimal_valid["should_stop"] = True
        minimal_valid["stop_reason"] = "All criteria met"
        p = ScientistPlanOutput.model_validate(minimal_valid)
        assert p.should_stop is True
        assert p.stop_reason == "All criteria met"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ScientistPlanOutput.model_validate({"hypothesis": "test"})

    def test_extra_fields_ignored(self, minimal_valid):
        minimal_valid["reasoning"] = "this should be dropped"
        p = ScientistPlanOutput.model_validate(minimal_valid)
        assert "reasoning" not in p.model_dump()

    def test_model_dump_roundtrip(self, minimal_valid):
        """model_dump() should produce a dict compatible with orchestrator .get() patterns."""
        p = ScientistPlanOutput.model_validate(minimal_valid)
        d = p.model_dump()
        assert isinstance(d, dict)
        assert d.get("strategy") == "incremental"
        assert d.get("should_stop") is False
        assert isinstance(d.get("changes"), list)
        assert d["changes"][0].get("what") == "add L2"
