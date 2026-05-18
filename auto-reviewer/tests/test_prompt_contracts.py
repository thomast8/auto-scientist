"""Structural regression guards for reviewer prompt and agent-label contracts.

These tests lock in the observer/planner/critic separation and the
reviewer-vocabulary agent-name labels that `yes-we-name-them-scalable-sprout.md`
established. They are pure source inspection - no LLM, no I/O - so they run
in milliseconds and catch silent drift on edits that would otherwise only
surface in a live smoke run.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from auto_core.notebook import append_entry
from auto_core.sdk_utils import validate_report_structure
from auto_core.state import ExperimentState
from auto_reviewer.agents import adversary, findings, hunter, prober, stop_gate, surveyor
from auto_reviewer.prompts.adversary import build_adversary_system
from auto_reviewer.prompts.findings import FINDINGS_USER
from auto_reviewer.prompts.hunter import HUNTER_SYSTEM
from auto_reviewer.prompts.prober import PROBER_SYSTEM
from auto_reviewer.prompts.stop_gate import (
    ASSESSMENT_USER,
    STOP_ADVERSARY_USER,
    STOP_DEBATE_SYSTEM,
    STOP_REVISION_USER,
)
from auto_reviewer.prompts.surveyor import SURVEYOR_SYSTEM, build_surveyor_system
from auto_reviewer.schemas import HunterPlanOutput


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


class TestReviewerDeadEnds:
    """Auto-reviewer gets the same dead-end contract as auto-scientist."""

    def test_hunter_schema_accepts_dead_ends(self) -> None:
        plan = HunterPlanOutput.model_validate(
            {
                "hypothesis": "Cache eviction cannot race on the single-thread path",
                "strategy": "incremental",
                "changes": [],
                "expected_impact": "No reproducer signal",
                "should_stop": False,
                "stop_reason": None,
                "notebook_entry": "Closed the single-thread race hypothesis.",
                "dead_ends": [
                    {
                        "description": "single-thread eviction race",
                        "evidence": "probe 1.0 passed with deterministic scheduler",
                    }
                ],
            }
        )

        assert plan.dead_ends[0].description == "single-thread eviction race"
        assert plan.dead_ends[0].evidence == "probe 1.0 passed with deterministic scheduler"

    def test_hunter_schema_defaults_dead_ends_to_empty(self) -> None:
        plan = HunterPlanOutput.model_validate(
            {
                "hypothesis": "Probe the cache invalidation path",
                "strategy": "incremental",
                "changes": [],
                "expected_impact": "A failing repro if the suspicion is valid",
                "should_stop": False,
                "stop_reason": None,
                "notebook_entry": "Planning the next probe.",
            }
        )

        assert plan.dead_ends == []

    def test_hunter_json_schema_mentions_dead_ends(self) -> None:
        assert "dead_ends" in hunter.HUNTER_PLAN_SCHEMA["properties"]
        assert "dead_ends" in HUNTER_SYSTEM

    def test_hunter_user_prompts_inject_dead_ends(self) -> None:
        section = hunter._build_dead_ends_section("1. single-thread race - probe passed")
        assert "<dead_ends>" in section
        assert "Do not re-chase" in section

        for template in (hunter.HUNTER_USER, hunter.HUNTER_REVISION_USER):
            rendered = template.format(
                goal="review PR",
                domain_knowledge="repo context",
                version="v02",
                analysis_json="{}",
                notebook_content="toc",
                prediction_history="tree",
                pending_abductions_section="",
                dead_ends_section=section,
                original_plan="{}",
                concern_ledger="[]",
            )
            assert "<dead_ends>" in rendered
            assert "single-thread race" in rendered

    def test_adversary_prompt_flags_rechasing_dead_ends(self) -> None:
        _system, user = adversary._build_critic_prompt(
            plan={"hypothesis": "single-thread eviction race"},
            notebook_content="toc",
            domain_knowledge="repo context",
            dead_ends="1. single-thread eviction race - probe passed",
        )

        assert "<dead_ends>" in user
        assert "re-chases an entry" in user

    def test_findings_prompt_surfaces_ruled_out_paths(self) -> None:
        rendered = FINDINGS_USER.format(
            state_json="{}",
            notebook_toc="toc",
            prediction_tree="tree",
            dead_ends_section="<dead_ends>closed path</dead_ends>",
            workspace_path="/tmp/review",
        )
        assert "<dead_ends>closed path</dead_ends>" in rendered

    def test_findings_report_structure_requires_ruled_out_section(self) -> None:
        report_without_ruled_out = """\
# Review

## Summary
Done.

## Confirmed bugs
None.

## Refuted suspicions
Single-thread race refuted.

## Ungrounded findings
None.

## Open questions
None.

## Known limitations
Sandboxed probe only.
"""

        issues = validate_report_structure(report_without_ruled_out)

        assert "Missing section: ruled out" in issues

    def test_stop_gate_prompt_templates_accept_dead_ends(self) -> None:
        assessment = ASSESSMENT_USER.format(
            goal="review PR",
            stop_reason="all clear",
            domain_knowledge="repo context",
            prediction_history="tree",
            pending_abductions_section="",
            dead_ends_section="<dead_ends>closed path</dead_ends>",
            notebook_content="toc",
        )
        stop_system = STOP_DEBATE_SYSTEM.format(
            persona_text="coverage auditor",
            persona_instructions="challenge gaps",
            critic_output_schema="{}",
        )
        stop_user = STOP_ADVERSARY_USER.format(
            goal="review PR",
            domain_knowledge="repo context",
            notebook_section="<notebook_toc>toc</notebook_toc>",
            analysis_json="{}",
            prediction_history="tree",
            dead_ends_section="<dead_ends>closed path</dead_ends>",
            stop_reason="all clear",
            completeness_assessment="{}",
        )
        revision_user = STOP_REVISION_USER.format(
            goal="review PR",
            domain_knowledge="repo context",
            notebook_content="toc",
            analysis_json="{}",
            prediction_history="tree",
            dead_ends_section="<dead_ends>closed path</dead_ends>",
            stop_reason="all clear",
            completeness_assessment="{}",
            concern_ledger="[]",
            plan_schema="{}",
            version="v03",
        )

        assert "<dead_ends>closed path</dead_ends>" in assessment
        assert "coverage auditor" in stop_system
        assert "<dead_ends>closed path</dead_ends>" in stop_user
        assert "not re-chase" in revision_user

    @pytest.mark.parametrize(
        "func",
        [
            hunter.run_hunter,
            hunter.run_hunter_revision,
            adversary.run_single_critic_debate,
            adversary.run_debate,
            findings.run_findings,
            stop_gate.run_completeness_assessment,
            stop_gate.run_single_stop_debate,
            stop_gate.run_hunter_stop_revision,
        ],
    )
    def test_shared_role_entrypoints_accept_dead_ends(self, func) -> None:
        param = inspect.signature(func).parameters["dead_ends"]
        assert param.default == ""

    @pytest.mark.asyncio
    async def test_hunter_entrypoint_injects_dead_ends(self, tmp_path) -> None:
        notebook = tmp_path / "lab_notebook.xml"
        append_entry(notebook, "initial review note", version="v00", source="surveyor")
        captured: dict[str, str] = {}

        async def fake_retry(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return {
                "hypothesis": "probe cache invalidation",
                "strategy": "incremental",
                "changes": [],
                "expected_impact": "signal",
                "should_stop": False,
                "stop_reason": None,
                "notebook_entry": "next probe",
            }

        with (
            patch.object(hunter, "get_backend", return_value=object()),
            patch.object(hunter, "agent_retry_loop", side_effect=fake_retry),
        ):
            await hunter.run_hunter(
                analysis={"observations": []},
                notebook_path=notebook,
                version="v01",
                dead_ends="single-thread race closed",
            )

        assert "<dead_ends>" in captured["prompt"]
        assert "single-thread race closed" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_findings_entrypoint_injects_dead_ends(self, tmp_path) -> None:
        notebook = tmp_path / "lab_notebook.xml"
        append_entry(notebook, "review complete", version="v00", source="hunter")
        captured: dict[str, str] = {}

        async def fake_retry(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return "# Review of PR\n\n## Summary\n\nDone."

        with (
            patch.object(findings, "get_backend", return_value=object()),
            patch.object(findings, "agent_retry_loop", side_effect=fake_retry),
        ):
            await findings.run_findings(
                state=ExperimentState(domain="owner/repo#1", goal="review PR"),
                notebook_path=notebook,
                output_dir=tmp_path,
                dead_ends="single-thread race closed",
            )

        assert "Dead ends:" in captured["prompt"]
        assert "single-thread race closed" in captured["prompt"]

    @pytest.mark.asyncio
    async def test_stop_gate_entrypoint_injects_dead_ends(self, tmp_path) -> None:
        notebook = tmp_path / "lab_notebook.xml"
        append_entry(notebook, "stop proposed", version="v00", source="hunter")
        captured: dict[str, str] = {}

        async def fake_retry(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return {
                "sub_questions": [],
                "overall_coverage": "partial",
                "recommendation": "continue",
            }

        with (
            patch.object(stop_gate, "get_backend", return_value=object()),
            patch.object(stop_gate, "agent_retry_loop", side_effect=fake_retry),
        ):
            await stop_gate.run_completeness_assessment(
                goal="review PR",
                stop_reason="complete",
                notebook_path=notebook,
                dead_ends="single-thread race closed",
            )

        assert "<dead_ends>" in captured["prompt"]
        assert "single-thread race closed" in captured["prompt"]

    def test_prober_openai_network_access_requires_explicit_opt_in(self) -> None:
        param = inspect.signature(prober.run_prober).parameters["network_access"]
        assert param.default is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("network_access", [False, True])
    async def test_prober_openai_network_access_flows_to_sdk_options(
        self,
        tmp_path,
        network_access: bool,
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_query(prompt, options):
            captured["prompt"] = prompt
            captured["options"] = options
            version_dir = tmp_path / "v01"
            version_dir.mkdir(parents=True, exist_ok=True)
            (version_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "success": True,
                        "return_code": 0,
                        "timed_out": False,
                        "error": None,
                        "attempts": 1,
                    }
                )
            )
            yield SimpleNamespace(type="result", usage={}, session_id="session-1")

        with patch.object(prober, "get_backend", return_value=SimpleNamespace(query=fake_query)):
            result = await prober.run_prober(
                plan={"hypothesis": "probe cache invalidation", "changes": []},
                previous_script=tmp_path / "missing.py",
                output_dir=tmp_path,
                version="v01",
                provider="openai",
                network_access=network_access,
            )

        assert result == tmp_path / "v01" / "run_result.json"
        assert captured["options"].network_access is network_access
        assert "<runtime_contract>" in captured["options"].system_prompt


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


class TestProberResultsArtifacts:
    """The Prober must emit `results.txt` so auto-core's Analyst gate passes.

    The shared orchestrator keys the next iteration's Surveyor on
    `version_entry.results_path`, which `evaluate()` only sets when a
    `results.txt` lives next to the version's run_result.json. Without
    the redirect, a clean reviewer iteration 1 crashes iteration 2 with
    "ANALYZE: skipped (no results file)".
    """

    def test_invocation_redirects_stdout_to_results_txt(self) -> None:
        # PROBER_SYSTEM is ecosystem-agnostic now (Intake picks the runner),
        # but the redirect tail that creates results.txt / stderr.txt /
        # exitcode.txt must still be mandated in the prompt so every run
        # produces the files auto-core's Analyst gate keys on.
        assert "> results.txt 2>stderr.txt; echo $? > exitcode.txt" in PROBER_SYSTEM

    def test_recap_mentions_results_artifact(self) -> None:
        # Guards against future edits that trim the recap and drop the
        # reminder about the three runtime-artifact files.
        assert "results.txt" in PROBER_SYSTEM.split("<recap>")[-1]
        assert "stderr.txt" in PROBER_SYSTEM.split("<recap>")[-1]
        assert "exitcode.txt" in PROBER_SYSTEM.split("<recap>")[-1]


class TestReviewerVocabularyInPrompts:
    """Reviewer prompts must not drift back into auto-scientist vocabulary.

    The `lab notebook` / `## Open abductions` phrasing leaked from the
    auto-scientist sibling package during extraction. LLM agents read these
    prompts every turn - seeing science-framed vocabulary nudges them toward
    a "run an experiment" frame instead of a "find a bug, reproduce it" frame.
    These assertions catch regressions on future prompt edits.

    Note: the underlying notebook file on disk is still `lab_notebook.xml`
    with a `<lab_notebook>` XML root, because that is an `auto_core` contract.
    Only the human-readable prose changed.
    """

    def test_surveyor_uses_investigation_log_not_lab_notebook(self) -> None:
        assert "lab notebook" not in SURVEYOR_SYSTEM
        assert "investigation log" in SURVEYOR_SYSTEM

    def test_hunter_uses_investigation_log_not_lab_notebook(self) -> None:
        assert "lab notebook" not in HUNTER_SYSTEM
        assert "investigation log" in HUNTER_SYSTEM

    def test_findings_report_header_is_open_questions(self) -> None:
        from auto_reviewer.prompts.findings import FINDINGS_SYSTEM

        # User-visible section header in the final review report.
        assert "## Open abductions" not in FINDINGS_SYSTEM
        assert "## Open questions" in FINDINGS_SYSTEM


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

    def test_findings_writes_report_via_write_tool(self) -> None:
        # The Findings artifact is a file on disk written by the agent,
        # not a string returned through the text channel. The validator
        # reads report.md from disk and the orchestrator round-trips it;
        # a revert that goes back to text-channel-as-artifact would
        # silently reintroduce the "WARNING: incomplete" clobber bug.
        agent_src = inspect.getsource(findings)
        assert '"Write"' in agent_src
        assert "report_path.read_text" in agent_src

        from auto_reviewer.prompts.findings import FINDINGS_SYSTEM

        assert "Write tool" in FINDINGS_SYSTEM
        assert "Return the markdown report as a plain string" not in FINDINGS_SYSTEM
