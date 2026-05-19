"""Direct agent tests for the OpenAI SDK defaults."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from auto_core.sdk_backend import SDKMessage, SDKOptions

from auto_scientist.agents import analyst, coder, scientist


class CapturingBackend:
    def __init__(self, callback=None) -> None:
        self.callback = callback or (lambda: None)
        self.options: list[SDKOptions] = []
        self.prompts: list[str] = []

    async def query(self, prompt: str, options: SDKOptions):
        self.prompts.append(prompt)
        self.options.append(options)
        result = self.callback()
        yield SDKMessage(type="result", result=result, session_id="s1", usage={})


@pytest.mark.asyncio
async def test_analyst_defaults_to_openai_backend(monkeypatch, tmp_path: Path) -> None:
    captured_provider = {}
    result = json.dumps(
        {
            "key_metrics": [],
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
        }
    )
    backend = CapturingBackend(lambda: result)

    def _get_backend(provider: str):
        captured_provider["provider"] = provider
        return backend

    monkeypatch.setattr(analyst, "get_backend", _get_backend)
    results_path = tmp_path / "results.txt"
    results_path.write_text("rmse: 0.5")
    notebook_path = tmp_path / "notebook.md"
    notebook_path.write_text("# Notebook")

    output = await analyst.run_analyst(
        results_path=results_path,
        plot_paths=[],
        notebook_path=notebook_path,
    )

    assert output["observations"] == ["ok"]
    assert captured_provider["provider"] == "openai"
    assert (
        "Tool calls are allowed before the final JSON response." in backend.options[0].system_prompt
    )


@pytest.mark.asyncio
async def test_coder_defaults_to_openai_backend(monkeypatch, tmp_path: Path) -> None:
    captured_provider = {}

    def _write_outputs() -> None:
        version_dir = tmp_path / "v01"
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "experiment.py").write_text("print('ok')\n")
        (version_dir / "run_result.json").write_text(
            json.dumps({"success": True, "return_code": 0, "timed_out": False, "error": None})
        )

    backend = CapturingBackend(_write_outputs)

    def _get_backend(provider: str):
        captured_provider["provider"] = provider
        return backend

    monkeypatch.setattr(coder, "get_backend", _get_backend)

    script_path = await coder.run_coder(
        plan={"hypothesis": "test", "changes": []},
        previous_script=tmp_path / "missing" / "experiment.py",
        output_dir=tmp_path,
        version="v01",
    )

    assert script_path == tmp_path / "v01" / "experiment.py"
    assert captured_provider["provider"] == "openai"
    assert "python3 {script_path}" in backend.options[0].system_prompt
    assert "uv run" not in backend.prompts[0]


@pytest.mark.asyncio
async def test_scientist_defaults_to_openai_backend(monkeypatch, tmp_path: Path) -> None:
    captured_provider = {}
    plan = json.dumps(
        {
            "hypothesis": "test hypothesis",
            "strategy": "incremental",
            "changes": [{"what": "do thing", "why": "because", "how": "like this", "priority": 1}],
            "expected_impact": "improvement",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "Testing incremental approach",
        }
    )
    backend = CapturingBackend(lambda: plan)

    def _get_backend(provider: str):
        captured_provider["provider"] = provider
        return backend

    monkeypatch.setattr(scientist, "get_backend", _get_backend)
    notebook_path = tmp_path / "notebook.md"
    notebook_path.write_text("# Notebook")

    output = await scientist.run_scientist(
        analysis={"observations": []},
        notebook_path=notebook_path,
        version="v01",
    )

    assert output["hypothesis"] == "test hypothesis"
    assert captured_provider["provider"] == "openai"
    assert (
        "Tool calls are allowed before the final JSON response." in backend.options[0].system_prompt
    )
