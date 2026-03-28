"""Tests for the structured debate Pydantic models."""

import pytest
from pydantic import ValidationError

from auto_scientist.agents.debate_models import (
    ConcernLedgerEntry,
    CriticOutput,
    DebateResult,
    DebateRound,
    DefenseResponse,
    ScientistDefense,
)


class TestConcern:
    def test_valid_concern(self):
        from auto_scientist.agents.debate_models import Concern

        c = Concern(
            claim="Mixed-unit soil moisture",
            severity="high",
            confidence="high",
            category="methodology",
        )
        assert c.claim == "Mixed-unit soil moisture"
        assert c.severity == "high"

    def test_invalid_severity_raises(self):
        from auto_scientist.agents.debate_models import Concern

        with pytest.raises(ValidationError):
            Concern(claim="x", severity="critical", confidence="high", category="methodology")

    def test_invalid_confidence_raises(self):
        from auto_scientist.agents.debate_models import Concern

        with pytest.raises(ValidationError):
            Concern(claim="x", severity="high", confidence="very_high", category="methodology")

    def test_invalid_category_raises(self):
        from auto_scientist.agents.debate_models import Concern

        with pytest.raises(ValidationError):
            Concern(claim="x", severity="high", confidence="high", category="invalid")

    def test_new_categories_accepted(self):
        """New persona categories: trajectory, falsification, consistency."""
        from auto_scientist.agents.debate_models import Concern

        for cat in ("trajectory", "falsification", "consistency"):
            c = Concern(claim="x", severity="high", confidence="high", category=cat)
            assert c.category == cat

    def test_old_categories_rejected(self):
        """Old persona categories novelty/feasibility are no longer valid."""
        from auto_scientist.agents.debate_models import Concern

        for cat in ("novelty", "feasibility"):
            with pytest.raises(ValidationError):
                Concern(claim="x", severity="high", confidence="high", category=cat)

    def test_extra_fields_tolerated(self):
        from auto_scientist.agents.debate_models import Concern

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


class TestDefenseResponse:
    def test_valid_response(self):
        dr = DefenseResponse(
            concern="Data quality issue",
            verdict="accepted",
            reasoning="Valid point, will fix.",
        )
        assert dr.verdict == "accepted"

    def test_invalid_verdict_raises(self):
        with pytest.raises(ValidationError):
            DefenseResponse(
                concern="x",
                verdict="maybe",
                reasoning="unsure",
            )

    def test_all_verdicts_accepted(self):
        for verdict in ("accepted", "rejected", "partially_accepted"):
            dr = DefenseResponse(concern="x", verdict=verdict, reasoning="ok")
            assert dr.verdict == verdict


class TestScientistDefense:
    def test_valid_defense(self):
        sd = ScientistDefense(
            responses=[
                {
                    "concern": "Data issue",
                    "verdict": "accepted",
                    "reasoning": "Will fix.",
                }
            ],
            additional_points="Checked via web search.",
        )
        assert len(sd.responses) == 1

    def test_default_additional_points(self):
        sd = ScientistDefense(responses=[])
        assert sd.additional_points == ""

    def test_extra_fields_tolerated(self):
        sd = ScientistDefense(responses=[], extra="ignored")
        assert sd.responses == []


class TestConcernLedgerEntry:
    def test_full_entry(self):
        entry = ConcernLedgerEntry(
            claim="Mixed units",
            severity="high",
            confidence="high",
            category="methodology",
            persona="Methodologist",
            critic_model="google:gemini-3.1-pro-preview",
            scientist_verdict="accepted",
            scientist_reasoning="Valid, need normalization.",
        )
        assert entry.persona == "Methodologist"
        assert entry.scientist_verdict == "accepted"

    def test_entry_without_defense(self):
        entry = ConcernLedgerEntry(
            claim="Redundant approach",
            severity="medium",
            confidence="low",
            category="trajectory",
            persona="Falsification Expert",
            critic_model="openai:gpt-5.4",
        )
        assert entry.scientist_verdict is None
        assert entry.scientist_reasoning is None

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
        assert d["scientist_verdict"] is None


class TestDebateRoundAndResult:
    def test_debate_round_critic_only(self):
        co = CriticOutput(
            concerns=[],
            alternative_hypotheses=[],
            overall_assessment="ok",
        )
        dr = DebateRound(critic_output=co)
        assert dr.scientist_defense is None

    def test_debate_round_with_defense(self):
        co = CriticOutput(
            concerns=[],
            alternative_hypotheses=[],
            overall_assessment="ok",
        )
        sd = ScientistDefense(responses=[])
        dr = DebateRound(critic_output=co, scientist_defense=sd)
        assert dr.scientist_defense is not None

    def test_debate_result(self):
        co = CriticOutput(
            concerns=[],
            alternative_hypotheses=[],
            overall_assessment="ok",
        )
        result = DebateResult(
            persona="Methodologist",
            critic_model="google:gemini-3.1-pro-preview",
            rounds=[DebateRound(critic_output=co)],
            raw_transcript=[{"role": "critic", "content": "ok"}],
        )
        assert result.persona == "Methodologist"
        assert len(result.rounds) == 1
        assert len(result.raw_transcript) == 1
