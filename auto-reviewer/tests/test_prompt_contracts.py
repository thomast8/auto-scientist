"""Structural regression guards for reviewer prompt and agent-label contracts.

These tests lock in the observer/planner/critic separation and the
reviewer-vocabulary agent-name labels that `yes-we-name-them-scalable-sprout.md`
established. They are pure source inspection - no LLM, no I/O - so they run
in milliseconds and catch silent drift on edits that would otherwise only
surface in a live smoke run.
"""

from __future__ import annotations

import inspect

import pytest
from auto_reviewer.agents import findings, hunter, surveyor
from auto_reviewer.prompts.adversary import build_adversary_system
from auto_reviewer.prompts.hunter import HUNTER_SYSTEM
from auto_reviewer.prompts.surveyor import SURVEYOR_SYSTEM, build_surveyor_system


class TestSurveyorScopeBoundary:
    """The Surveyor must stay an observer; its prompt must fence off bug claims."""

    def test_scope_boundary_block_is_present(self) -> None:
        system = build_surveyor_system("claude")
        assert "<scope_boundary>" in system
        assert "</scope_boundary>" in system

    def test_out_of_scope_bug_claim_is_fenced(self) -> None:
        assert '"This is an off-by-one bug"' in SURVEYOR_SYSTEM

    def test_severity_is_redefined_as_notability(self) -> None:
        assert "notability" in SURVEYOR_SYSTEM.lower()

    def test_examples_do_not_reintroduce_bug_claims(self) -> None:
        # Old phrasing asserted the bug exists.
        assert '"Eviction loop mutates dict while iterating"' not in SURVEYOR_SYSTEM
        # New phrasing describes the co-mutation pattern.
        assert "iterates `self._map` while" in SURVEYOR_SYSTEM

    def test_recap_enforces_pattern_framing(self) -> None:
        # Positive in-scope vs negative out-of-scope example in the recap.
        assert "share state on this path" in SURVEYOR_SYSTEM
        assert "race in A and B" in SURVEYOR_SYSTEM


class TestHunterHypothesisFraming:
    """The Hunter plans hypotheses for the Prober; it does not discover bugs."""

    def test_recap_forbids_discovery_phrasing(self) -> None:
        assert "hypotheses the Prober will probe" in HUNTER_SYSTEM
        assert 'not as discovered bugs ("I found bug X")' in HUNTER_SYSTEM

    def test_schema_prediction_framing_is_hypothesis_not_bug(self) -> None:
        # Post-review fix: line 36 used to read `prediction = "the bug is X"`,
        # contradicting the recap. The new wording frames the prediction as a
        # testable claim.
        assert '"the bug is X"' not in HUNTER_SYSTEM
        assert "under condition Y, behavior X would fire" in HUNTER_SYSTEM


class TestAdversaryScopeBoundary:
    """The Adversary critiques the BugPlan; it does not write code or assert bugs."""

    @pytest.mark.parametrize("provider", ["claude", "gpt"])
    def test_scope_boundary_block_is_present_on_both_providers(self, provider: str) -> None:
        system = build_adversary_system(provider)
        assert "<scope_boundary>" in system
        assert "</scope_boundary>" in system

    @pytest.mark.parametrize("provider", ["claude", "gpt"])
    def test_code_level_suggestion_is_fenced(self, provider: str) -> None:
        system = build_adversary_system(provider)
        assert "code-level, Prober's domain" in system

    @pytest.mark.parametrize("provider", ["claude", "gpt"])
    def test_surveyor_lane_is_fenced(self, provider: str) -> None:
        # Prevents personas (especially Security / API Break) from drifting
        # into "the Surveyor should have flagged X" concerns.
        system = build_adversary_system(provider)
        assert "Surveyor's lane, not ours" in system


class TestReviewerAgentNameLabels:
    """Reviewer agent runners must pass the reviewer vocabulary to the SDK layer.

    The `agent_name=` argument flows into log warnings and validation-error
    messages. Scientist-vocabulary labels leaking into reviewer runs make the
    retry/error surface inconsistent with the reviewer's TUI panel names.
    """

    def test_surveyor_uses_reviewer_vocabulary(self) -> None:
        src = inspect.getsource(surveyor)
        assert 'agent_name="Analyst"' not in src
        assert 'agent_name="Surveyor"' in src
        # Third positional arg on validate_json_output is also the label.
        assert 'SurveyorOutput, "Analyst"' not in src
        assert 'SurveyorOutput, "Surveyor"' in src

    def test_hunter_uses_reviewer_vocabulary(self) -> None:
        src = inspect.getsource(hunter)
        assert 'agent_name="Scientist"' not in src
        assert 'agent_name="Scientist revision"' not in src
        assert 'agent_name="Hunter"' in src
        assert 'agent_name="Hunter revision"' in src
        assert 'HunterPlanOutput, "Scientist"' not in src
        assert 'HunterPlanOutput, "Scientist revision"' not in src

    def test_findings_uses_reviewer_vocabulary(self) -> None:
        src = inspect.getsource(findings)
        assert 'agent_name="Report"' not in src
        assert 'agent_name="Findings"' in src
