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

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Importing auto_reviewer installs its registry (auto_reviewer/__init__.py
# calls install_reviewer_registry()). All tests below assume this side effect
# has happened. This is the whole point: the registry is a process-global,
# and the app that imports last wins.
import auto_reviewer  # noqa: F401 - import for side effect
import pytest
from auto_core.agent_dispatch import AGENT_FNS, DEBATE_PERSONAS, get_agent_fn
from auto_core.model_config import AgentModelConfig, ModelConfig
from auto_core.orchestrator import Orchestrator
from auto_reviewer.config import ReviewConfig
from auto_reviewer.state import ReviewState

# ---------------------------------------------------------------------------
# Registry wiring: the right modules are registered.
# ---------------------------------------------------------------------------


class TestRegistryContents:
    """AGENT_FNS should point at auto_reviewer.agents.* modules, not scientist."""

    def test_observer_bound_to_surveyor(self):
        assert AGENT_FNS["observer"].__module__ == "auto_reviewer.agents.surveyor"

    def test_planner_bound_to_hunter(self):
        assert AGENT_FNS["planner"].__module__ == "auto_reviewer.agents.hunter"

    def test_implementer_bound_to_prober(self):
        assert AGENT_FNS["implementer"].__module__ == "auto_reviewer.agents.prober"

    def test_canonicalizer_bound_to_intake(self):
        assert AGENT_FNS["canonicalizer"].__module__ == "auto_reviewer.agents.intake"

    def test_reporter_bound_to_findings(self):
        assert AGENT_FNS["reporter"].__module__ == "auto_reviewer.agents.findings"

    def test_adversary_bound_to_adversary_module(self):
        assert AGENT_FNS["adversary"].__module__ == "auto_reviewer.agents.adversary"

    def test_stop_gate_bound_to_reviewer_stop_gate(self):
        for key in ("assessor", "stop_reviser", "stop_adversary"):
            assert AGENT_FNS[key].__module__ == "auto_reviewer.agents.stop_gate"

    def test_review_personas_registered(self):
        names = {p["name"] for p in DEBATE_PERSONAS}
        # Auto-Reviewer's adversary catalog (not the scientist personas).
        assert names == {"Design Intent", "Security", "Concurrency", "API Break", "Input Fuzz"}

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
    @patch("auto_reviewer.agents.intake.run_intake", new_callable=AsyncMock)
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
        assert AGENT_FNS["canonicalizer"].__module__ == "auto_reviewer.agents.intake"

    @pytest.mark.asyncio
    @patch("auto_reviewer.agents.intake.run_intake", new_callable=AsyncMock)
    async def test_intake_receives_prompt_as_goal(self, mock_intake, tmp_path: Path):
        """The user's natural-language prompt must flow through state.goal
        into run_intake's `goal` kwarg. The intake agent is the sole
        interpreter of the pointer now that the CLI no longer pre-seeds
        repo_path / pr_ref."""
        canonical = tmp_path / "review_workspace" / "canonical"
        canonical.mkdir(parents=True)
        mock_intake.return_value = canonical

        prompt_text = "review PR #42 on owner/repo against main"
        state = ReviewState(
            domain="review_pr_42_on_owner_repo",
            goal=prompt_text,
            phase="ingestion",
            data_path=str(tmp_path),
        )
        orchestrator = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path / "review_workspace",
        )
        await orchestrator._run_ingestion()

        mock_intake.assert_called_once()
        assert mock_intake.call_args.kwargs["goal"] == prompt_text

    @pytest.mark.asyncio
    @patch(
        "auto_reviewer.agents.intake.run_intake",
        new_callable=AsyncMock,
        side_effect=RuntimeError("intake boom"),
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
        with pytest.raises(RuntimeError, match="intake boom"):
            await orchestrator._run_ingestion()

    @pytest.mark.asyncio
    @patch("auto_reviewer.agents.surveyor.run_surveyor", new_callable=AsyncMock)
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
        """Smoke check: calling get_agent_fn('planner') at dispatch time
        returns a dispatcher that routes to auto_reviewer.agents.hunter."""
        fn = get_agent_fn("planner")
        # The dispatcher re-resolves via sys.modules, so unittest.mock.patch
        # on the hunter module still intercepts at call time.
        with patch(
            "auto_reviewer.agents.hunter.run_hunter",
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

    @pytest.mark.asyncio
    async def test_prober_receives_configured_sandbox_network_access(self, tmp_path: Path):
        workspace = tmp_path / "review_workspace"
        workspace.mkdir()
        state = ReviewState(
            domain="owner/repo#42",
            goal="g",
            phase="iteration",
            iteration=0,
            data_path=str(tmp_path),
        )
        orchestrator = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=workspace,
            model_config=ModelConfig(defaults=AgentModelConfig(provider="openai", model="gpt-5.5")),
        )
        orchestrator.config = ReviewConfig(
            name="review",
            description="review config",
            repo_path=str(tmp_path),
            run_command="python {script_path}",
            implementer_sandbox_network_access=True,
        )
        captured: dict[str, object] = {}

        async def fake_prober(**kwargs):
            captured.update(kwargs)
            return workspace / "v00" / "run_result.json"

        with (
            patch("auto_core.ensure_deps._ensure_pip"),
            patch("auto_core.ensure_deps._preinstall_scientific_packages"),
            patch("auto_reviewer.agents.prober.run_prober", side_effect=fake_prober),
        ):
            await orchestrator._run_coder({"hypothesis": "probe cache invalidation"})

        assert captured["provider"] == "openai"
        assert captured["network_access"] is True


class TestIntakeGeneratedConfigGuards:
    """Generated ReviewConfig cannot opt into operator-only sandbox network access."""

    @pytest.mark.asyncio
    @patch("auto_reviewer.agents.intake.safe_query")
    async def test_generated_config_cannot_enable_sandbox_network_access(
        self, mock_query, tmp_path: Path
    ):
        from auto_core.sdk_backend import SDKMessage
        from auto_reviewer.agents.intake import run_intake

        output_dir = tmp_path / "review_workspace"
        data_dir = output_dir / "data"
        touched_dir = data_dir / "touched_files"
        touched_dir.mkdir(parents=True)
        (data_dir / "diff.patch").write_text("diff --git a/x.py b/x.py\n")
        (data_dir / "pr_metadata.json").write_text(
            json.dumps(
                {
                    "title": "PR title",
                    "baseRefName": "main",
                    "headRefName": "feature/demo",
                }
            )
        )
        (touched_dir / "x.py").write_text("print('x')\n")

        repo = tmp_path / "repo"
        repo.mkdir()
        config_path = output_dir / "domain_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "name": "review",
                    "description": "review",
                    "run_cwd": str(repo),
                    "run_command": "uv run pytest -x -s {script_path}",
                    "repo_path": str(repo),
                    "pr_ref": "owner/repo#1",
                    "base_ref": "main",
                    "head_ref": "feature/demo",
                    "implementer_sandbox_network_access": True,
                }
            )
        )

        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                config_path.write_text(
                    json.dumps(
                        {
                            "name": "review",
                            "description": "review",
                            "run_cwd": str(repo),
                            "run_command": "uv run pytest -x -s {script_path}",
                            "repo_path": str(repo),
                            "pr_ref": "owner/repo#1",
                            "base_ref": "main",
                            "head_ref": "feature/demo",
                            "implementer_sandbox_network_access": False,
                        }
                    )
                )
            yield SDKMessage(type="result", usage={}, session_id="review-session")

        mock_query.side_effect = fake_query

        result = await run_intake(
            raw_data_path=tmp_path,
            output_dir=output_dir,
            goal="review owner/repo#1",
            config_path=config_path,
        )

        assert result == data_dir
        assert call_count == 2
        assert json.loads(config_path.read_text())["implementer_sandbox_network_access"] is False
