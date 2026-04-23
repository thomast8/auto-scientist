"""Tests for the structured debate Pydantic models."""

import pytest
from auto_core.agents.debate_models import (
    ConcernLedgerEntry,
    CriticOutput,
    DebateResult,
)
from pydantic import ValidationError


class TestConcern:
    def test_valid_concern(self):
        from auto_core.agents.debate_models import Concern

        c = Concern(
            claim="Mixed-unit soil moisture",
            severity="high",
            confidence="high",
            category="methodology",
        )
        assert c.claim == "Mixed-unit soil moisture"
        assert c.severity == "high"

    def test_invalid_severity_raises(self):
        from auto_core.agents.debate_models import Concern

        with pytest.raises(ValidationError):
            Concern(claim="x", severity="critical", confidence="high", category="methodology")

    def test_invalid_confidence_raises(self):
        from auto_core.agents.debate_models import Concern

        with pytest.raises(ValidationError):
            Concern(claim="x", severity="high", confidence="very_high", category="methodology")

    def test_invalid_category_raises(self):
        from auto_core.agents.debate_models import Concern

        with pytest.raises(ValidationError):
            Concern(claim="x", severity="high", confidence="high", category="invalid")

    def test_new_categories_accepted(self):
        """New persona categories: trajectory, falsification, consistency."""
        from auto_core.agents.debate_models import Concern

        for cat in ("trajectory", "falsification", "consistency"):
            c = Concern(claim="x", severity="high", confidence="high", category=cat)
            assert c.category == cat

    def test_old_categories_rejected(self):
        """Old persona categories novelty/feasibility are no longer valid."""
        from auto_core.agents.debate_models import Concern

        for cat in ("novelty", "feasibility"):
            with pytest.raises(ValidationError):
                Concern(claim="x", severity="high", confidence="high", category=cat)

    def test_extra_fields_tolerated(self):
        from auto_core.agents.debate_models import Concern

        c = Concern(
            claim="x",
            severity="high",
            confidence="high",
            category="methodology",
            extra_field="ignored",
        )
        assert c.claim == "x"


class TestCriticOutput:
    def test_valid_output(self):
        co = CriticOutput(
            concerns=[
                {
                    "claim": "Data quality issue",
                    "severity": "high",
                    "confidence": "medium",
                    "category": "methodology",
                }
            ],
            alternative_hypotheses=["Try log-transform"],
            overall_assessment="Plan has issues.",
        )
        assert len(co.concerns) == 1
        assert co.concerns[0].claim == "Data quality issue"

    def test_empty_concerns_valid(self):
        co = CriticOutput(
            concerns=[],
            alternative_hypotheses=[],
            overall_assessment="No issues found.",
        )
        assert co.concerns == []

    def test_extra_fields_tolerated(self):
        co = CriticOutput(
            concerns=[],
            alternative_hypotheses=[],
            overall_assessment="ok",
            bonus="ignored",
        )
        assert co.overall_assessment == "ok"


class TestConcernLedgerEntry:
    def test_full_entry(self):
        entry = ConcernLedgerEntry(
            claim="Mixed units",
            severity="high",
            confidence="high",
            category="methodology",
            persona="Methodologist",
            critic_model="google:gemini-3.1-pro-preview",
        )
        assert entry.persona == "Methodologist"

    def test_model_dump_serializable(self):
        entry = ConcernLedgerEntry(
            claim="x",
            severity="low",
            confidence="low",
            category="other",
            persona="p",
            critic_model="m",
        )
        d = entry.model_dump()
        assert isinstance(d, dict)
        assert d["claim"] == "x"


class TestDebateResult:
    def test_debate_result(self):
        co = CriticOutput(
            concerns=[],
            alternative_hypotheses=[],
            overall_assessment="ok",
        )
        result = DebateResult(
            persona="Methodologist",
            critic_model="google:gemini-3.1-pro-preview",
            critic_output=co,
            raw_transcript=[{"role": "critic", "content": "ok"}],
        )
        assert result.persona == "Methodologist"
        assert result.critic_output.overall_assessment == "ok"
        assert len(result.raw_transcript) == 1
