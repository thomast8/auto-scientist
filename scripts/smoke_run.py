#!/usr/bin/env python3
"""Smoke run: full pipeline with mocked LLM calls.

Runs the real orchestrator through ingestion -> 2 iterations -> report with
all LLM boundaries mocked. The debate loop, summarizer polling, state machine,
file I/O, and console output all run for real.

Usage:
    uv run python scripts/smoke_run.py
    uv run python scripts/smoke_run.py --output-dir /tmp/smoke_test
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from auto_scientist.agents.critic import MIN_RESPONSE_LENGTH
from auto_scientist.model_config import AgentModelConfig, ModelConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState


def _pad(text: str) -> str:
    """Pad text to meet MIN_RESPONSE_LENGTH for critic responses."""
    if len(text) >= MIN_RESPONSE_LENGTH:
        return text
    return text + " " + "x" * (MIN_RESPONSE_LENGTH - len(text) - 1)


# ---------------------------------------------------------------------------
# Canned return values
# ---------------------------------------------------------------------------

ANALYSIS_ITER0 = {
    "domain_knowledge": "Test domain: 2-column CSV with x and y values",
    "data_summary": "2 rows, 2 columns (x, y). Simple numeric data.",
    "criteria_results": [],
}

ANALYSIS_ITER1 = {
    "criteria_results": [{"name": "C1", "status": "pass"}],
}

PLAN_ITER0 = {
    "hypothesis": "Explore the data to understand its structure",
    "strategy": "exploration",
    "changes": [
        {
            "what": "explore data",
            "why": "first look",
            "how": "summary stats",
            "priority": 1,
        }
    ],
    "expected_impact": "Baseline understanding of data",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "## Exploration\nFirst look at the data structure.",
}

PLAN_ITER1 = {
    "hypothesis": "Linear relationship between x and y",
    "strategy": "incremental",
    "changes": [
        {
            "what": "fit model",
            "why": "validate hypothesis",
            "how": "linear regression",
            "priority": 1,
        }
    ],
    "expected_impact": "Confirm linear relationship",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "## Linear model\nFitting a linear model to x,y data.",
    "top_level_criteria": [
        {
            "name": "C1",
            "description": "Model R2 exceeds 0.8",
            "metric_key": "r2",
            "condition": "> 0.8",
        }
    ],
}

REVISED_PLAN = {
    "hypothesis": "Linear relationship (revised after debate)",
    "strategy": "incremental",
    "changes": [
        {
            "what": "fit model with cross-validation",
            "why": "critic suggested validation",
            "how": "linear regression + CV",
            "priority": 1,
        }
    ],
    "expected_impact": "More robust validation of linear fit",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "## Revised plan\nAdded cross-validation per critic feedback.",
}


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _make_ingestor_mock():
    async def fake(
        raw_data_path, output_dir, goal, interactive=False,
        config_path=None, model=None, message_buffer=None,
    ):
        dest = output_dir / "data"
        dest.mkdir(parents=True, exist_ok=True)
        for f in raw_data_path.iterdir():
            if f.is_file():
                (dest / f.name).write_text(f.read_text())
        if message_buffer is not None:
            message_buffer.append("[Ingestor] Canonicalized 1 file")
        return dest

    return fake


def _make_analyst_mock():
    results = [ANALYSIS_ITER0, ANALYSIS_ITER1, ANALYSIS_ITER1]
    call_idx = 0

    async def fake(**kwargs):
        nonlocal call_idx
        buf = kwargs.get("message_buffer")
        if buf is not None:
            buf.append("[Analyst] Analyzing data")
        result = results[min(call_idx, len(results) - 1)]
        call_idx += 1
        return result

    return fake


def _make_scientist_mock():
    results = [PLAN_ITER0, PLAN_ITER1]
    call_idx = 0

    async def fake(**kwargs):
        nonlocal call_idx
        buf = kwargs.get("message_buffer")
        if buf is not None:
            buf.append("[Scientist] Formulating plan")
        result = results[min(call_idx, len(results) - 1)]
        call_idx += 1
        return result

    return fake


def _make_revision_mock():
    async def fake(**kwargs):
        buf = kwargs.get("message_buffer")
        if buf is not None:
            buf.append("[Scientist Revision] Revising plan after debate")
        return REVISED_PLAN

    return fake


def _make_coder_mock():
    async def fake(
        plan, previous_script, output_dir, version,
        domain_knowledge="", data_path="", model=None,
        message_buffer=None, run_timeout_minutes=120,
        run_command="uv run {script_path}",
        top_level_criteria=None,
    ):
        version_dir = output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        script = version_dir / "experiment.py"
        script.write_text("# Smoke test script\nprint('R2=0.95')\n")
        (version_dir / "results.txt").write_text(
            f"Smoke test results for {version}\nR2=0.95\n"
        )
        (version_dir / "run_result.json").write_text(
            json.dumps({"success": True, "return_code": 0, "timed_out": False})
        )

        if message_buffer is not None:
            message_buffer.append(f"[Coder] Writing {version} experiment script")
            message_buffer.append("[Coder] Running experiment... success")

        return script

    return fake


def _make_report_mock():
    async def fake(**kwargs):
        buf = kwargs.get("message_buffer")
        if buf is not None:
            buf.append("[Report] Generating final summary")
        return "# Smoke Test Report\n\nAll experiments completed successfully."

    return fake


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_smoke(output_dir: Path) -> None:
    """Run the full pipeline smoke test."""
    # Set up raw data
    data_dir = output_dir / "raw_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "sample.csv").write_text("x,y\n1,2\n3,4\n5,6\n")

    experiment_dir = output_dir / "experiments"

    state = ExperimentState(
        domain="smoke_test",
        goal="Test the full pipeline end-to-end with mocked LLMs",
        data_path=str(data_dir),
    )

    model_config = ModelConfig(
        defaults=AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6"),
        critics=[
            AgentModelConfig(provider="openai", model="gpt-4o"),
            AgentModelConfig(provider="google", model="gemini-2.5-pro"),
        ],
        summarizer=AgentModelConfig(provider="openai", model="gpt-4o-mini"),
    )

    orchestrator = Orchestrator(
        state=state,
        data_path=data_dir,
        output_dir=experiment_dir,
        max_iterations=2,
        model_config=model_config,
        debate_rounds=2,
        stream=False,
    )

    with (
        # Infrastructure
        patch.object(Orchestrator, "_validate_prerequisites"),

        # Agent-level mocks
        patch("auto_scientist.agents.ingestor.run_ingestor", side_effect=_make_ingestor_mock()),
        patch("auto_scientist.agents.analyst.run_analyst", side_effect=_make_analyst_mock()),
        patch("auto_scientist.agents.scientist.run_scientist", side_effect=_make_scientist_mock()),
        patch("auto_scientist.agents.scientist.run_scientist_revision", side_effect=_make_revision_mock()),
        patch("auto_scientist.agents.coder.run_coder", side_effect=_make_coder_mock()),
        patch("auto_scientist.agents.report.run_report", side_effect=_make_report_mock()),

        # LLM-level mocks (debate loop runs for real)
        patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            side_effect=[
                _pad("OAI Critique R1: The hypothesis lacks specificity about what linear means."),
                _pad("OAI Critique R2: Revised, sample size concern addressed. Need cross-validation."),
            ],
        ),
        patch(
            "auto_scientist.agents.critic.query_google",
            new_callable=AsyncMock,
            side_effect=[
                _pad("Google Critique R1: Consider confounding variables in the x-y relationship."),
                _pad("Google Critique R2: Methodology concerns partially addressed. Proceed with caution."),
            ],
        ),
        patch(
            "auto_scientist.agents.critic.collect_text_from_query",
            new_callable=AsyncMock,
            side_effect=[
                _pad("Scientist to OAI: Valid points about specificity. Will add R2 threshold."),
                _pad("Scientist to Google: Agreed on confounders. Will add residual analysis."),
            ],
        ),

        # Summarizer LLM mock
        patch(
            "auto_scientist.summarizer._query_summary",
            new_callable=AsyncMock,
            return_value="Summarizing agent progress...",
        ),
    ):
        await orchestrator.run()

    # -----------------------------------------------------------------
    # Quick validation
    # -----------------------------------------------------------------
    state_path = experiment_dir / "state.json"
    final_state = ExperimentState.load(state_path)

    checks = [
        ("Pipeline completed", final_state.phase == "stopped"),
        ("2 iterations ran", len(final_state.versions) == 2),
        ("v00 completed", final_state.versions[0].status == "completed"),
        ("v01 completed", final_state.versions[1].status == "completed"),
        ("Notebook exists", (experiment_dir / "lab_notebook.xml").exists()),
        ("Debate on iter 1", (experiment_dir / "v01" / "debate.json").exists()),
        ("No debate on iter 0", not (experiment_dir / "v00" / "debate.json").exists()),
        ("Report generated", (experiment_dir / "report.md").exists()),
        ("Criteria defined", final_state.success_criteria is not None and len(final_state.success_criteria) > 0),
        ("v01 scored", final_state.versions[1].score is not None),
        ("Best version tracked", final_state.best_version == "v01"),
    ]

    # Validate debate transcript
    debate_ok = False
    debate_path = experiment_dir / "v01" / "debate.json"
    if debate_path.exists():
        debate_data = json.loads(debate_path.read_text())
        critiques = debate_data.get("critiques", [])
        if len(critiques) == 2:
            models = {c["model"] for c in critiques}
            transcripts_ok = all(
                len(c["transcript"]) == 3
                and c["transcript"][0]["role"] == "critic"
                and c["transcript"][1]["role"] == "scientist"
                and c["transcript"][2]["role"] == "critic"
                for c in critiques
            )
            debate_ok = (
                "openai:gpt-4o" in models
                and "google:gemini-2.5-pro" in models
                and transcripts_ok
            )
    checks.append(("Debate transcripts valid", debate_ok))

    print("\n" + "=" * 50)
    print("SMOKE TEST VALIDATION")
    print("=" * 50)

    all_ok = True
    for label, passed in checks:
        status = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
        print(f"  [{status}] {label}")
        if not passed:
            all_ok = False

    print(f"\nOutput dir: {experiment_dir}")
    print("=" * 50)

    if not all_ok:
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke run: full pipeline with mocked LLMs")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: temp dir)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        asyncio.run(run_smoke(output_dir))
    else:
        with tempfile.TemporaryDirectory(prefix="smoke_run_") as tmp:
            asyncio.run(run_smoke(Path(tmp)))


if __name__ == "__main__":
    main()
