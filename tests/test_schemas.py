"""Tests for Pydantic output schemas (agent output validation)."""

import pytest
from pydantic import ValidationError

from auto_scientist.schemas import (
    AnalystOutput,
    CoderRunResult,
    CompletenessAssessmentOutput,
    HypothesisPrediction,
    PlanChange,
    PredictionOutcome,
    ScientistPlanOutput,
    SubQuestionAssessment,
)

# ---------------------------------------------------------------------------
# AnalystOutput
# ---------------------------------------------------------------------------


class TestAnalystOutput:
    @pytest.fixture
    def minimal_valid(self):
        return {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": ["data loaded"],
        }

    def test_minimal_valid(self, minimal_valid):
        a = AnalystOutput.model_validate(minimal_valid)
        assert a.domain_knowledge == ""
        assert a.data_summary is None

    def test_full_iteration_0(self, minimal_valid):
        minimal_valid["domain_knowledge"] = "This is SpO2 pulse oximetry data"
        minimal_valid["data_summary"] = "Files: data.csv (1000 rows, 5 columns)"
        a = AnalystOutput.model_validate(minimal_valid)
        assert a.domain_knowledge == "This is SpO2 pulse oximetry data"
        assert a.data_summary == "Files: data.csv (1000 rows, 5 columns)"

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AnalystOutput.model_validate({"key_metrics": []})

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


# ---------------------------------------------------------------------------
# PlanChange
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CoderRunResult
# ---------------------------------------------------------------------------


class TestCoderRunResult:
    def test_valid_success(self):
        r = CoderRunResult(success=True, return_code=0, timed_out=False, error=None, attempts=1)
        assert r.success is True
        assert r.return_code == 0

    def test_valid_failure(self):
        r = CoderRunResult(success=False, return_code=1, error="ImportError: no module named foo")
        assert r.success is False
        assert r.error == "ImportError: no module named foo"

    def test_defaults(self):
        r = CoderRunResult(success=True)
        assert r.return_code == -1
        assert r.timed_out is False
        assert r.error is None
        assert r.attempts == 1

    def test_missing_success_raises(self):
        with pytest.raises(ValidationError):
            CoderRunResult.model_validate({})

    def test_extra_fields_ignored(self):
        r = CoderRunResult.model_validate({"success": True, "extra": "ignored"})
        assert not hasattr(r, "extra")

    def test_timed_out(self):
        r = CoderRunResult(
            success=False, return_code=124, timed_out=True, error="Timed out after 120 minutes"
        )
        assert r.timed_out is True

    def test_bool_coercion_from_string(self):
        """LLM might write 'true' as a string in some edge cases."""
        r = CoderRunResult.model_validate({"success": True, "timed_out": False})
        assert r.success is True


# ---------------------------------------------------------------------------
# PlanChange
# ---------------------------------------------------------------------------


class TestPlanChange:
    def test_valid(self):
        c = PlanChange(
            what="add regularization", why="prevent overfitting", how="L2 penalty", priority=1
        )
        assert c.priority == 1

    def test_missing_field(self):
        with pytest.raises(ValidationError):
            PlanChange(what="add regularization", why="prevent overfitting", priority=1)


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
                {
                    "what": "add L2",
                    "why": "reduce overfitting",
                    "how": "sklearn param",
                    "priority": 1,
                },
            ],
            "expected_impact": "Lower test error",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "## Iteration 1\nAdding L2 regularization.",
        }

    def test_minimal_valid(self, minimal_valid):
        p = ScientistPlanOutput.model_validate(minimal_valid)
        assert p.hypothesis == "Adding regularization will reduce overfitting"
        assert p.strategy == "incremental"
        assert len(p.changes) == 1
        assert p.should_stop is False

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


# ---------------------------------------------------------------------------
# HypothesisPrediction
# ---------------------------------------------------------------------------


class TestHypothesisPrediction:
    def test_valid(self):
        p = HypothesisPrediction(
            prediction="residual correlation < 0.1",
            diagnostic="compute Pearson r between residuals and x",
            if_confirmed="noise is additive, continue with OLS",
            if_refuted="noise is multiplicative, switch to WLS",
        )
        assert p.follows_from is None

    def test_with_follows_from(self):
        p = HypothesisPrediction(
            prediction="re-test spline after structural change",
            diagnostic="profile smoothing parameter",
            if_confirmed="spline is now identifiable",
            if_refuted="still unidentifiable, fix parameter",
            follows_from="smoothing parameter has interior minimum",
        )
        assert p.follows_from == "smoothing parameter has interior minimum"

    def test_extra_fields_ignored(self):
        p = HypothesisPrediction(
            prediction="test",
            diagnostic="test",
            if_confirmed="ok",
            if_refuted="nope",
            extra_field="ignored",
        )
        assert not hasattr(p, "extra_field")


# ---------------------------------------------------------------------------
# PredictionOutcome
# ---------------------------------------------------------------------------


class TestPredictionOutcome:
    def test_valid_outcomes(self):
        for outcome in ("confirmed", "refuted", "inconclusive"):
            p = PredictionOutcome(
                prediction="test",
                outcome=outcome,
                evidence="measured 0.5",
            )
            assert p.outcome == outcome

    def test_invalid_outcome_rejected(self):
        with pytest.raises(ValidationError):
            PredictionOutcome(prediction="test", outcome="maybe", evidence="n/a")

    def test_extra_fields_ignored(self):
        p = PredictionOutcome(
            prediction="test",
            outcome="confirmed",
            evidence="ok",
            bonus="nope",
        )
        assert not hasattr(p, "bonus")


# ---------------------------------------------------------------------------
# ScientistPlanOutput - testable_predictions
# ---------------------------------------------------------------------------


class TestScientistPlanOutputPredictions:
    @pytest.fixture
    def minimal_plan(self):
        return {
            "hypothesis": "test",
            "strategy": "incremental",
            "changes": [{"what": "x", "why": "y", "how": "z", "priority": 1}],
            "expected_impact": "better",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "title\nbody",
        }

    def test_defaults_to_empty_list(self, minimal_plan):
        p = ScientistPlanOutput.model_validate(minimal_plan)
        assert p.testable_predictions == []

    def test_with_predictions(self, minimal_plan):
        minimal_plan["testable_predictions"] = [
            {
                "prediction": "spline fits better locally",
                "diagnostic": "compare regional RMSE",
                "if_confirmed": "focus on local fit",
                "if_refuted": "problem is elsewhere",
            },
        ]
        p = ScientistPlanOutput.model_validate(minimal_plan)
        assert len(p.testable_predictions) == 1
        assert p.testable_predictions[0].follows_from is None

    def test_with_follows_from(self, minimal_plan):
        minimal_plan["testable_predictions"] = [
            {
                "prediction": "re-test after change",
                "diagnostic": "profile again",
                "if_confirmed": "now works",
                "if_refuted": "still broken",
                "follows_from": "original prediction",
            },
        ]
        p = ScientistPlanOutput.model_validate(minimal_plan)
        assert p.testable_predictions[0].follows_from == "original prediction"


# ---------------------------------------------------------------------------
# AnalystOutput - prediction_outcomes
# ---------------------------------------------------------------------------


class TestAnalystOutputPredictions:
    @pytest.fixture
    def minimal_analysis(self):
        return {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": [],
        }

    def test_defaults_to_empty_list(self, minimal_analysis):
        a = AnalystOutput.model_validate(minimal_analysis)
        assert a.prediction_outcomes == []

    def test_with_outcomes(self, minimal_analysis):
        minimal_analysis["prediction_outcomes"] = [
            {
                "prediction": "spline fits better locally",
                "outcome": "confirmed",
                "evidence": "regional RMSE 0.31 vs 0.58",
            },
        ]
        a = AnalystOutput.model_validate(minimal_analysis)
        assert len(a.prediction_outcomes) == 1
        assert a.prediction_outcomes[0].outcome == "confirmed"


# ---------------------------------------------------------------------------
# SubQuestionAssessment
# ---------------------------------------------------------------------------


class TestSubQuestionAssessment:
    def test_valid(self):
        s = SubQuestionAssessment(
            question="What factors drive outlet clarity?",
            coverage="thorough",
            evidence=["setl_mh R²=0.94"],
            gaps=[],
        )
        assert s.coverage == "thorough"

    def test_invalid_coverage(self):
        with pytest.raises(ValidationError):
            SubQuestionAssessment(
                question="test",
                coverage="unknown",
                evidence=[],
                gaps=[],
            )

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            SubQuestionAssessment(
                question="",
                coverage="shallow",
                evidence=[],
                gaps=[],
            )

    def test_extra_fields_ignored(self):
        s = SubQuestionAssessment(
            question="test",
            coverage="shallow",
            evidence=[],
            gaps=[],
            extra="ignored",
        )
        assert "extra" not in s.model_dump()


# ---------------------------------------------------------------------------
# CompletenessAssessmentOutput
# ---------------------------------------------------------------------------


class TestCompletenessAssessmentOutput:
    @pytest.fixture
    def valid_assessment(self):
        return {
            "sub_questions": [
                {
                    "question": "What factors drive outlet clarity?",
                    "coverage": "thorough",
                    "evidence": ["setl_mh R²=0.94", "variance decomposition"],
                    "gaps": [],
                },
                {
                    "question": "Are there nonlinear effects?",
                    "coverage": "shallow",
                    "evidence": ["quadratic test refuted"],
                    "gaps": ["Only one functional form tested"],
                },
            ],
            "overall_coverage": "partial",
            "recommendation": "continue",
        }

    def test_valid(self, valid_assessment):
        c = CompletenessAssessmentOutput.model_validate(valid_assessment)
        assert len(c.sub_questions) == 2
        assert c.overall_coverage == "partial"
        assert c.recommendation == "continue"

    def test_invalid_overall_coverage(self, valid_assessment):
        valid_assessment["overall_coverage"] = "unknown"
        with pytest.raises(ValidationError):
            CompletenessAssessmentOutput.model_validate(valid_assessment)

    def test_invalid_recommendation(self, valid_assessment):
        valid_assessment["recommendation"] = "maybe"
        with pytest.raises(ValidationError):
            CompletenessAssessmentOutput.model_validate(valid_assessment)

    def test_extra_fields_ignored(self, valid_assessment):
        valid_assessment["extra"] = "should be dropped"
        c = CompletenessAssessmentOutput.model_validate(valid_assessment)
        assert "extra" not in c.model_dump()

    def test_model_dump_roundtrip(self, valid_assessment):
        c = CompletenessAssessmentOutput.model_validate(valid_assessment)
        d = c.model_dump()
        assert isinstance(d, dict)
        assert d["sub_questions"][0]["coverage"] == "thorough"
        assert d["sub_questions"][1]["gaps"] == ["Only one functional form tested"]

    def test_empty_sub_questions_rejected(self):
        with pytest.raises(ValidationError):
            CompletenessAssessmentOutput.model_validate(
                {
                    "sub_questions": [],
                    "overall_coverage": "incomplete",
                    "recommendation": "continue",
                }
            )

    def test_recommendation_coerced_when_coverage_not_thorough(self):
        """LLM says stop but coverage is partial -> coerced to continue."""
        c = CompletenessAssessmentOutput.model_validate(
            {
                "sub_questions": [
                    {"question": "test", "coverage": "shallow", "evidence": [], "gaps": ["gap"]},
                ],
                "overall_coverage": "partial",
                "recommendation": "stop",
            }
        )
        assert c.recommendation == "continue"

    def test_recommendation_stop_allowed_when_thorough(self):
        c = CompletenessAssessmentOutput.model_validate(
            {
                "sub_questions": [
                    {"question": "test", "coverage": "thorough", "evidence": ["solid"], "gaps": []},
                ],
                "overall_coverage": "thorough",
                "recommendation": "stop",
            }
        )
        assert c.recommendation == "stop"

    def test_recommendation_coerced_when_incomplete(self):
        """LLM says stop but coverage is 'incomplete' -> recommendation coerced to continue."""
        c = CompletenessAssessmentOutput.model_validate(
            {
                "sub_questions": [
                    {
                        "question": "What causes outlet turbidity?",
                        "coverage": "unexplored",
                        "evidence": [],
                        "gaps": ["never measured"],
                    }
                ],
                "overall_coverage": "incomplete",
                "recommendation": "stop",
            }
        )
        # 'incomplete' is not 'thorough', so recommendation must be coerced to 'continue'
        assert c.recommendation == "continue"

    def test_sub_question_defaults_evidence_and_gaps_to_empty_lists(self):
        """Omitting evidence and gaps produces empty lists, not validation errors."""
        s = SubQuestionAssessment(
            question="Does temperature affect catalyst lifetime?",
            coverage="shallow",
        )
        assert s.evidence == []
        assert s.gaps == []
