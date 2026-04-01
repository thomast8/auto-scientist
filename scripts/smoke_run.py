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
from unittest.mock import patch

from auto_scientist.agent_result import AgentResult
from auto_scientist.model_config import AgentModelConfig, ModelConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.state import ExperimentState

_MIN_RESPONSE_LENGTH = 50


def _pad(text: str) -> str:
    """Pad text to minimum length for critic responses."""
    if len(text) >= _MIN_RESPONSE_LENGTH:
        return text
    return text + " " + "x" * (_MIN_RESPONSE_LENGTH - len(text) - 1)


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
    "top_level_criteria": [
        {
            "name": "C1",
            "description": "Model R2 exceeds 0.8",
            "metric_key": "r2",
            "condition": "> 0.8",
        }
    ],
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


def _make_delayed_side_effect(delay: float, values: list):
    """Return an async callable that sleeps then yields values from a list."""
    idx = 0

    async def _delayed(*args, **kwargs):
        nonlocal idx
        await asyncio.sleep(delay)
        val = values[min(idx, len(values) - 1)]
        idx += 1
        return val

    return _delayed


async def _drip_buffer(buf: list[str] | None, entries: list[str], delay: float = 0.3):
    """Feed buffer entries one at a time with delays between them."""
    if buf is None:
        await asyncio.sleep(delay * len(entries))
        return
    for entry in entries:
        buf.append(entry)
        await asyncio.sleep(delay)


def _make_ingestor_mock():
    async def fake(
        raw_data_path,
        output_dir,
        goal,
        interactive=False,
        config_path=None,
        model=None,
        message_buffer=None,
    ):
        dest = output_dir / "data"
        dest.mkdir(parents=True, exist_ok=True)
        for f in raw_data_path.iterdir():
            if f.is_file():
                (dest / f.name).write_text(f.read_text())
        await _drip_buffer(
            message_buffer,
            [
                "[Ingestor] Reading raw data files",
                "[Ingestor] Validating CSV schema and column types",
                "[Ingestor] Copying canonical data to output directory",
                "[Ingestor] Canonicalized 1 file",
            ],
        )
        return dest

    return fake


def _make_analyst_mock():
    results = [ANALYSIS_ITER0, ANALYSIS_ITER1, ANALYSIS_ITER1]
    call_idx = 0

    async def fake(**kwargs):
        nonlocal call_idx
        buf = kwargs.get("message_buffer")
        await _drip_buffer(
            buf,
            [
                "[Analyst] Loading data and checking structure",
                "[Analyst] Computing summary statistics",
                "[Analyst] Evaluating success criteria",
                "[Analyst] Analysis complete",
            ],
        )
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
        await _drip_buffer(
            buf,
            [
                "[Scientist] Reviewing analysis results",
                "[Scientist] Formulating hypothesis",
                "[Scientist] Designing experimental strategy",
                "[Scientist] Plan complete",
            ],
        )
        result = results[min(call_idx, len(results) - 1)]
        call_idx += 1
        return result

    return fake


def _make_revision_mock():
    async def fake(**kwargs):
        buf = kwargs.get("message_buffer")
        await _drip_buffer(
            buf,
            [
                "[Revision] Reading critic feedback",
                "[Revision] Incorporating suggestions into plan",
                "[Revision] Revised plan ready",
            ],
        )
        return REVISED_PLAN

    return fake


def _make_coder_mock():
    async def fake(
        plan,
        previous_script,
        output_dir,
        version,
        domain_knowledge="",
        data_path="",
        model=None,
        message_buffer=None,
        run_timeout_minutes=120,
        run_command="uv run {script_path}",
        top_level_criteria=None,
    ):
        version_dir = output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        script = version_dir / "experiment.py"
        script.write_text("# Smoke test script\nprint('R2=0.95')\n")
        (version_dir / "results.txt").write_text(f"Smoke test results for {version}\nR2=0.95\n")
        (version_dir / "run_result.json").write_text(
            json.dumps({"success": True, "return_code": 0, "timed_out": False})
        )

        await _drip_buffer(
            message_buffer,
            [
                f"[Coder] Reading plan for {version}",
                f"[Coder] Writing {version} experiment script",
                "[Coder] Verifying script syntax",
                "[Coder] Running experiment...",
                "[Coder] Collecting results and generating plots",
                "[Coder] Experiment completed successfully",
            ],
        )
        return script

    return fake


def _make_report_mock():
    async def fake(**kwargs):
        buf = kwargs.get("message_buffer")
        await _drip_buffer(
            buf,
            [
                "[Report] Gathering results from all iterations",
                "[Report] Compiling findings and recommendations",
                "[Report] Writing final summary",
            ],
        )
        return "# Smoke Test Report\n\nAll experiments completed successfully."

    return fake


# ---------------------------------------------------------------------------
# Canned LLM responses for debate mocks
# ---------------------------------------------------------------------------

# Each persona gets one critic call.


def _cr(text: str, inp: int = 100, out: int = 50) -> AgentResult:
    """Build a padded AgentResult for critic/defense mock responses."""
    return AgentResult(text=_pad(text), input_tokens=inp, output_tokens=out)


oai_critic_responses = [
    _cr("OAI R1: Hypothesis lacks specificity about linearity."),
    _cr("OAI R2: Revised, sample size concern addressed."),
    _cr("OAI R3: Skeptical of generalization to larger datasets."),
    _cr("OAI R4: Optimistic about cross-validation plan."),
    _cr("OAI R5: Final round critique, methodology sound."),
    _cr("OAI R6: Remaining concerns addressed satisfactorily."),
]

google_critic_responses = [
    _cr("Google R1: Consider confounding variables.", 80, 40),
    _cr("Google R2: Methodology concerns partially addressed.", 80, 40),
    _cr("Google R3: Skeptical view on statistical power.", 80, 40),
    _cr("Google R4: Positive about bootstrap approach.", 80, 40),
    _cr("Google R5: Final assessment, approach is reasonable.", 80, 40),
    _cr("Google R6: All major concerns resolved.", 80, 40),
]

anthropic_defense_responses = [
    _cr("Scientist: Valid points, will add R2 threshold.", 90, 45),
    _cr("Scientist: Agreed on confounders, adding residuals.", 90, 45),
    _cr("Scientist: Adding cross-validation for generalization.", 90, 45),
    _cr("Scientist: Bootstrap intervals quantify uncertainty.", 90, 45),
    _cr("Scientist: Statistical power via sample size analysis.", 90, 45),
    _cr("Scientist: All critiques incorporated into plan.", 90, 45),
]

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
        stream=False,
    )

    # Counter for varying fake token stats per agent call
    _call_count = 0
    _token_table = [
        (1200, 400, 3),  # Ingestor
        (800, 300, 2),  # Analyst iter 0
        (1500, 600, 4),  # Scientist iter 0
        (2200, 900, 8),  # Coder iter 0
        (900, 350, 2),  # Analyst iter 1
        (1800, 700, 5),  # Scientist iter 1
        (1600, 650, 4),  # Revision iter 1
        (3500, 1400, 12),  # Coder iter 1
        (700, 250, 2),  # Analyst (final scoring)
        (2000, 800, 6),  # Report
    ]

    def _fake_sdk_usage(panel):
        nonlocal _call_count
        entry = _token_table[min(_call_count, len(_token_table) - 1)]
        panel.set_stats(input_tokens=entry[0], output_tokens=entry[1], num_turns=entry[2])
        _call_count += 1

    # Reduce summarizer poll interval so it fires during short agent runs
    import auto_scientist.summarizer as _summarizer_mod

    _orig_rws = _summarizer_mod.run_with_summaries

    async def _fast_rws(*args, interval=0.5, **kwargs):
        return await _orig_rws(*args, interval=interval, **kwargs)

    with (
        # Infrastructure
        patch.object(Orchestrator, "_validate_prerequisites"),
        # Patch at both locations: module def and orchestrator import
        patch("auto_scientist.summarizer.run_with_summaries", side_effect=_fast_rws),
        patch("auto_scientist.orchestrator.run_with_summaries", side_effect=_fast_rws),
        patch.object(
            Orchestrator,
            "_apply_sdk_usage",
            staticmethod(_fake_sdk_usage),
        ),
        # Agent-level mocks
        patch("auto_scientist.agents.ingestor.run_ingestor", side_effect=_make_ingestor_mock()),
        patch("auto_scientist.agents.analyst.run_analyst", side_effect=_make_analyst_mock()),
        patch("auto_scientist.agents.scientist.run_scientist", side_effect=_make_scientist_mock()),
        patch(
            "auto_scientist.agents.scientist.run_scientist_revision",
            side_effect=_make_revision_mock(),
        ),
        patch("auto_scientist.agents.coder.run_coder", side_effect=_make_coder_mock()),
        patch("auto_scientist.agents.report.run_report", side_effect=_make_report_mock()),
        # LLM-level mocks (debate loop runs for real, with delays)
        # Each persona gets one critic call.
        patch(
            "auto_scientist.agents.critic.query_openai",
            side_effect=_make_delayed_side_effect(0.5, oai_critic_responses),
        ),
        patch(
            "auto_scientist.agents.critic.query_google",
            side_effect=_make_delayed_side_effect(1.5, google_critic_responses),
        ),
        patch(
            "auto_scientist.agents.critic.query_anthropic",
            side_effect=_make_delayed_side_effect(0.4, anthropic_defense_responses),
        ),
        # Summarizer LLM mock - varied responses so panels accumulate
        # multiple lines (tests expand/compact toggle)
        patch(
            "auto_scientist.summarizer._query_summary",
            side_effect=_make_delayed_side_effect(
                0.1,
                [
                    "Inspecting raw data files and checking column types.",
                    "Reading CSV headers and validating schema consistency.",
                    "Computing summary statistics for all numeric columns.",
                    "Evaluating data quality: missing values, outliers, distributions.",
                    "Formulating hypothesis based on observed data patterns.",
                    "Planning experimental strategy with cross-validation approach.",
                    "Writing experiment script with polynomial regression pipeline.",
                    "Running experiment, collecting metrics and generating plots.",
                    "Comparing results against baseline and previous iterations.",
                    "Reviewing methodology and suggesting alternative approaches.",
                    "Analyzing critic feedback and identifying valid concerns.",
                    "Revising plan to incorporate cross-validation per critic feedback.",
                    "Generating final report with all findings and recommendations.",
                    "[done] Pipeline step completed successfully.",
                ],
            ),
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
        (
            "Predictions tracked",
            len(final_state.prediction_history) > 0,
        ),
    ]

    # Validate debate transcript (3 personas, each with rounds)
    debate_ok = False
    debate_path = experiment_dir / "v01" / "debate.json"
    if debate_path.exists():
        debate_data = json.loads(debate_path.read_text())
        debate_results = debate_data.get("debate_results", [])
        if len(debate_results) == 3:
            models = {r["critic_model"] for r in debate_results}
            has_rounds = all(len(r.get("rounds", [])) > 0 for r in debate_results)
            debate_ok = (
                "openai:gpt-4o" in models and "google:gemini-2.5-pro" in models and has_rounds
            )
    checks.append(("Debate transcripts valid", debate_ok))

    # Validate panel tracking via PipelineLive._panels
    all_panels = orchestrator._live._panels
    checks.append(("Panels tracked after flush", len(all_panels) > 0))
    all_collapsed = all(not p.expanded for p in all_panels)
    checks.append(("All panels collapsed after stop", all_collapsed))

    has_all_lines = any(len(p.all_lines) > 0 for p in all_panels)
    checks.append(("Panels retain all_lines history", has_all_lines))

    # Check that panels have multiple lines (for expand/compact testing)
    multi_line_panels = [p for p in all_panels if len(p.all_lines) > 1]
    checks.append(
        (
            f"Panels have multi-line history ({len(multi_line_panels)}/{len(all_panels)})",
            len(multi_line_panels) > 0,
        )
    )

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
        "--output-dir",
        type=Path,
        default=None,
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
