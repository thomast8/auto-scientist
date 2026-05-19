"""Reviewer agent guard wiring tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from auto_core.sdk_backend import SDKMessage, SDKOptions
from auto_reviewer.agents import intake, prober


class CapturingBackend:
    def __init__(self, callback):
        self.callback = callback
        self.options: list[SDKOptions] = []

    async def query(self, prompt: str, options: SDKOptions):
        self.options.append(options)
        self.callback()
        yield SDKMessage(type="result", session_id="s1", usage={})


@pytest.mark.asyncio
async def test_prober_default_provider_is_guarded_openai(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "review"
    repo_clone = output_dir / "repo_clone"
    repo_clone.mkdir(parents=True)
    previous = output_dir / "v00" / "run_result.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("{}")

    def _write_result() -> None:
        version_dir = output_dir / "v01"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "run_result.json").write_text(
            json.dumps({"success": True, "return_code": 0, "timed_out": False, "error": None})
        )

    captured_provider = {}
    backend = CapturingBackend(_write_result)

    def _get_backend(provider: str):
        captured_provider["provider"] = provider
        return backend

    monkeypatch.setattr(prober, "get_backend", _get_backend)

    await prober.run_prober(
        plan={"prediction": "x"},
        previous_script=previous,
        output_dir=output_dir,
        version="v01",
        data_path=str(repo_clone),
    )

    assert captured_provider["provider"] == "openai"
    options = backend.options[0]
    assert options.pre_tool_use_hook is not None
    assert options.network_access is False


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_prober_wires_workspace_guard(monkeypatch, tmp_path: Path, provider: str) -> None:
    output_dir = tmp_path / "review"
    repo_clone = output_dir / "repo_clone"
    repo_clone.mkdir(parents=True)
    previous = output_dir / "v00" / "run_result.json"
    previous.parent.mkdir(parents=True)
    previous.write_text("{}")

    def _write_result() -> None:
        version_dir = output_dir / "v01"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "run_result.json").write_text(
            json.dumps({"success": True, "return_code": 0, "timed_out": False, "error": None})
        )

    backend = CapturingBackend(_write_result)
    monkeypatch.setattr(prober, "get_backend", lambda _provider: backend)

    await prober.run_prober(
        plan={"prediction": "x"},
        previous_script=previous,
        output_dir=output_dir,
        version="v01",
        data_path=str(repo_clone),
        provider=provider,
    )

    options = backend.options[0]
    assert options.cwd == output_dir
    assert options.pre_tool_use_hook is not None
    assert options.pre_tool_use_hook.workspace == output_dir.resolve()
    assert options.pre_tool_use_hook.repo_clone == repo_clone.resolve()
    assert options.network_access is False


@pytest.mark.asyncio
async def test_intake_default_provider_is_guarded_openai(monkeypatch, tmp_path: Path) -> None:
    output_dir = tmp_path / "review"
    repo_clone = output_dir / "repo_clone"
    repo_clone.mkdir(parents=True)
    config_path = output_dir / "domain_config.json"

    def _write_intake_outputs() -> None:
        data_dir = output_dir / "data"
        touched = data_dir / "touched_files"
        touched.mkdir(parents=True, exist_ok=True)
        (data_dir / "diff.patch").write_text("diff --git a/a.py b/a.py\n")
        (data_dir / "pr_metadata.json").write_text(
            json.dumps({"title": "PR", "baseRefName": "main", "headRefName": "feature"})
        )
        (touched / "a.py").write_text("print('x')\n")
        config_path.write_text(
            json.dumps(
                {
                    "name": "review",
                    "repo_path": str(repo_clone),
                    "pr_ref": "owner/repo#1",
                    "base_ref": "main",
                    "head_ref": "feature",
                    "run_command": "python {script_path}",
                }
            )
        )

    captured_provider = {}
    backend = CapturingBackend(_write_intake_outputs)

    def _get_backend(provider: str):
        captured_provider["provider"] = provider
        return backend

    monkeypatch.setattr(intake, "get_backend", _get_backend)

    await intake.run_intake(
        raw_data_path=repo_clone,
        output_dir=output_dir,
        goal="review current branch",
        config_path=config_path,
    )

    assert captured_provider["provider"] == "openai"
    options = backend.options[0]
    assert options.pre_tool_use_hook is not None
    assert options.pre_tool_use_hook.repo_clone == repo_clone.resolve()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
async def test_intake_wires_workspace_guard(monkeypatch, tmp_path: Path, provider: str) -> None:
    output_dir = tmp_path / "review"
    repo_clone = output_dir / "repo_clone"
    repo_clone.mkdir(parents=True)
    config_path = output_dir / "domain_config.json"

    def _write_intake_outputs() -> None:
        data_dir = output_dir / "data"
        touched = data_dir / "touched_files"
        touched.mkdir(parents=True, exist_ok=True)
        (data_dir / "diff.patch").write_text("diff --git a/a.py b/a.py\n")
        (data_dir / "pr_metadata.json").write_text(
            json.dumps({"title": "PR", "baseRefName": "main", "headRefName": "feature"})
        )
        (touched / "a.py").write_text("print('x')\n")
        config_path.write_text(
            json.dumps(
                {
                    "name": "review",
                    "repo_path": str(repo_clone),
                    "pr_ref": "owner/repo#1",
                    "base_ref": "main",
                    "head_ref": "feature",
                    "run_command": "python {script_path}",
                }
            )
        )

    backend = CapturingBackend(_write_intake_outputs)
    monkeypatch.setattr(intake, "get_backend", lambda _provider: backend)

    await intake.run_intake(
        raw_data_path=repo_clone,
        output_dir=output_dir,
        goal="review current branch",
        config_path=config_path,
        provider=provider,
    )

    options = backend.options[0]
    assert options.cwd == output_dir
    assert options.pre_tool_use_hook is not None
    assert options.pre_tool_use_hook.workspace == output_dir.resolve()
    assert options.pre_tool_use_hook.repo_clone == repo_clone.resolve()
