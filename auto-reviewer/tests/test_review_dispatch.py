"""End-to-end dispatch test for auto-reviewer on the shared orchestrator.

Proves that importing `auto_reviewer` installs a RoleRegistry whose agent
functions resolve to the reviewer-flavored modules (not the scientist ones)
and that the orchestrator's phase handlers dispatch through them.

Covers the wiring gap the modular-extraction refactor created: before the
RoleRegistry carried agent_fns, `auto-reviewer review` would silently fall
back to auto_scientist agents because the orchestrator had hardcoded
`from auto_scientist.agents.* import run_*` imports.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

# Importing auto_reviewer installs its registry (auto_reviewer/__init__.py
# calls install_reviewer_registry()). All tests below assume this side effect
# has happened. This is the whole point: the registry is a process-global,
# and the app that imports last wins.
import auto_reviewer  # noqa: F401 - import for side effect
import pytest
from auto_core.agent_dispatch import AGENT_FNS, DEBATE_PERSONAS, get_agent_fn
from auto_core.orchestrator import Orchestrator
from auto_reviewer.state import ReviewState

# ---------------------------------------------------------------------------
# Registry wiring: the right modules are registered.
# ---------------------------------------------------------------------------


class TestRegistryContents:
    """AGENT_FNS should point at auto_reviewer.agents.* modules, not scientist."""

    def test_analyst_bound_to_surveyor(self):
        assert AGENT_FNS["analyst"].__module__ == "auto_reviewer.agents.surveyor"

    def test_scientist_bound_to_hunter(self):
        assert AGENT_FNS["scientist"].__module__ == "auto_reviewer.agents.hunter"

    def test_coder_bound_to_prober(self):
        assert AGENT_FNS["coder"].__module__ == "auto_reviewer.agents.prober"

    def test_ingestor_bound_to_intake(self):
        assert AGENT_FNS["ingestor"].__module__ == "auto_reviewer.agents.intake"

    def test_report_bound_to_findings(self):
        assert AGENT_FNS["report"].__module__ == "auto_reviewer.agents.findings"

    def test_debate_bound_to_adversary(self):
        assert AGENT_FNS["debate"].__module__ == "auto_reviewer.agents.adversary"

    def test_stop_gate_bound_to_reviewer_stop_gate(self):
        for key in ("completeness_assessment", "scientist_stop_revision", "single_stop_debate"):
            assert AGENT_FNS[key].__module__ == "auto_reviewer.agents.stop_gate"

    def test_review_personas_registered(self):
        names = {p["name"] for p in DEBATE_PERSONAS}
        # Auto-Reviewer's adversary catalog (not the scientist personas).
        assert names == {"Security", "Concurrency", "API Break", "Input Fuzz"}

    def test_dispatcher_rejects_unknown_key(self):
        with pytest.raises(RuntimeError, match="No agent function registered"):
            get_agent_fn("does_not_exist")


# ---------------------------------------------------------------------------
# Dispatch: the orchestrator actually calls the reviewer agents, end-to-end
# through the phase handlers.
# ---------------------------------------------------------------------------


class TestOrchestratorDispatchToReviewerAgents:
    """Run the orchestrator's internal phase handlers with mocked agent
    entry functions; assert the reviewer modules are the ones invoked."""

    @pytest.mark.asyncio
    @patch("auto_reviewer.agents.intake.run_ingestor", new_callable=AsyncMock)
    async def test_run_ingestion_calls_intake(self, mock_intake, tmp_path: Path):
        canonical = tmp_path / "review_workspace" / "canonical"
        canonical.mkdir(parents=True)
        mock_intake.return_value = canonical

        raw = tmp_path / "raw.patch"
        raw.write_text("diff --git a/x b/x\n")

        state = ReviewState(
            domain="owner/repo#42",
            goal="find correctness bugs",
            phase="ingestion",
            data_path=str(raw),
        )
        orchestrator = Orchestrator(
            state=state,
            data_path=raw,
            output_dir=tmp_path / "review_workspace",
        )
        result = await orchestrator._run_ingestion()

        assert result == canonical
        mock_intake.assert_called_once()
        # The scientist's ingestor must NOT have been touched. Importing
        # auto_scientist would clobber the registry, so we verify the
        # reviewer module is still the one registered.
        assert AGENT_FNS["ingestor"].__module__ == "auto_reviewer.agents.intake"

    @pytest.mark.asyncio
    @patch(
        "auto_reviewer.agents.intake.run_ingestor",
        new_callable=AsyncMock,
        side_effect=RuntimeError("intake LLM error"),
    )
    async def test_intake_error_propagates(self, _mock, tmp_path: Path):
        raw = tmp_path / "raw.patch"
        raw.write_text("diff\n")

        state = ReviewState(
            domain="owner/repo#42",
            goal="g",
            phase="ingestion",
            data_path=str(raw),
        )
        orchestrator = Orchestrator(
            state=state,
            data_path=raw,
            output_dir=tmp_path / "review_workspace",
        )
        with pytest.raises(RuntimeError, match="intake LLM error"):
            await orchestrator._run_ingestion()

    @pytest.mark.asyncio
    @patch("auto_reviewer.agents.surveyor.run_analyst", new_callable=AsyncMock)
    async def test_run_analyst_initial_calls_surveyor(self, mock_surveyor, tmp_path: Path):
        mock_surveyor.return_value = {
            "suspicions": [],
            "touched_symbols": [],
            "observations": [],
            "prediction_outcomes": [],
            "repo_knowledge": "",
            "diff_summary": None,
        }

        # Workspace layout the orchestrator expects.
        data_dir = tmp_path / "review_workspace" / "canonical"
        data_dir.mkdir(parents=True)

        state = ReviewState(
            domain="owner/repo#42",
            goal="g",
            phase="iteration",
            iteration=0,
            data_path=str(data_dir),
            raw_data_path=str(data_dir / "diff.patch"),
        )
        orchestrator = Orchestrator(
            state=state,
            data_path=data_dir,
            output_dir=tmp_path / "review_workspace",
        )
        await orchestrator._run_analyst_initial()
        mock_surveyor.assert_called_once()

    @pytest.mark.asyncio
    async def test_hunter_dispatch_selects_review_module(self):
        """Smoke check: calling get_agent_fn('scientist') at dispatch time
        returns a dispatcher that routes to auto_reviewer.agents.hunter."""
        fn = get_agent_fn("scientist")
        # The dispatcher re-resolves via sys.modules, so unittest.mock.patch
        # on the hunter module still intercepts at call time.
        with patch(
            "auto_reviewer.agents.hunter.run_scientist",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ) as mock_hunter:
            result = await fn(
                analysis={},
                notebook_path=Path("/tmp/nope"),
                version="v00",
            )
            assert result == {"ok": True}
            mock_hunter.assert_called_once()
