"""Tests for the Scientist agent."""

import json
from unittest.mock import MagicMock, patch

import pytest
from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

from auto_scientist.agents.scientist import (
    _format_predictions_for_prompt,
    run_scientist,
    run_scientist_revision,
)
from auto_scientist.sdk_utils import OutputValidationError
from auto_scientist.state import PredictionRecord

SAMPLE_LEDGER = [
    {
        "claim": "weak point",
        "severity": "medium",
        "confidence": "medium",
        "category": "methodology",
        "persona": "Methodologist",
        "critic_model": "test",
    },
]

SAMPLE_LEDGER_SMALL = [
    {
        "claim": "test concern",
        "severity": "low",
        "confidence": "low",
        "category": "other",
        "persona": "Test",
        "critic_model": "test",
    },
]


SAMPLE_PLAN = {
    "hypothesis": "test hypothesis",
    "strategy": "incremental",
    "changes": [{"what": "do thing", "why": "because", "how": "like this", "priority": 1}],
    "expected_impact": "improvement",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "First hypothesis\n\nTesting incremental approach",
}


class TestRunScientist:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_returns_parsed_plan(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook content")

        result = await run_scientist(
            analysis={"observations": []},
            notebook_path=notebook_path,
            version="v01",
        )

        assert result["hypothesis"] == "test hypothesis"
        assert result["strategy"] == "incremental"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_missing_notebook_uses_fallback(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "nonexistent.md"

        result = await run_scientist(
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_empty_output_raises(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(RuntimeError, match="returned no output"):
            await run_scientist(
                analysis={},
                notebook_path=notebook_path,
                version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_has_web_search_only(self, mock_query, tmp_path):
        """Scientist should have WebSearch only (file access is the Analyst's job)."""
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        captured_options = {}

        async def fake_query(**kwargs):
            captured_options.update(kwargs)
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        await run_scientist(analysis={}, notebook_path=notebook_path, version="v01")

        assert captured_options["options"].allowed_tools == ["WebSearch"]


class TestRunScientistMessageBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_populates_message_buffer(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Planning hypothesis..."
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        buf: list[str] = []
        await run_scientist(
            analysis={},
            notebook_path=notebook_path,
            version="v01",
            message_buffer=buf,
        )
        assert len(buf) == 1
        assert "Planning hypothesis..." in buf[0]

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_revision_populates_message_buffer(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Revising plan..."
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        buf: list[str] = []
        await run_scientist_revision(
            original_plan=SAMPLE_PLAN,
            concern_ledger=SAMPLE_LEDGER,
            analysis={},
            notebook_path=notebook_path,
            version="v01",
            message_buffer=buf,
        )
        assert len(buf) == 1
        assert "Revising plan..." in buf[0]


class TestRunScientistExploration:
    """Empty analysis + no criteria -> exploration plan."""

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_exploration_plan(self, mock_query, tmp_path):
        exploration_plan = {
            "hypothesis": "Data exploration to establish baselines and identify patterns",
            "strategy": "exploratory",
            "changes": [
                {
                    "what": "Compute distributions",
                    "why": "Understand data shape",
                    "how": "Histograms and summary stats",
                    "priority": 1,
                },
            ],
            "expected_impact": "Baseline understanding of the dataset",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "Data exploration\n\nFirst look at the data",
        }

        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(exploration_plan)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        result = await run_scientist(
            analysis={},
            notebook_path=notebook_path,
            version="v00",
        )
        assert result["strategy"] == "exploratory"


class TestRunScientistRevision:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_returns_revised_plan(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        ledger = [
            {
                "claim": "This is weak",
                "severity": "high",
                "confidence": "high",
                "category": "methodology",
                "persona": "Methodologist",
                "critic_model": "test",
                "scientist_verdict": "rejected",
                "scientist_reasoning": "I disagree",
            },
        ]

        result = await run_scientist_revision(
            original_plan=SAMPLE_PLAN,
            concern_ledger=ledger,
            analysis={"observations": []},
            notebook_path=notebook_path,
            version="v01",
        )

        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_empty_output_raises(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(RuntimeError, match="returned no output"):
            await run_scientist_revision(
                original_plan=SAMPLE_PLAN,
                concern_ledger=[],
                analysis={},
                notebook_path=notebook_path,
                version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_fallback_to_assistant_text(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = json.dumps(SAMPLE_PLAN)
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist_revision(
            original_plan=SAMPLE_PLAN,
            concern_ledger=SAMPLE_LEDGER_SMALL,
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"


class TestRunScientistAssistantFallback:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_run_scientist_fallback_to_assistant_text(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = json.dumps(SAMPLE_PLAN)
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_run_scientist_markdown_fenced_response(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        fenced = f"```json\n{json.dumps(SAMPLE_PLAN)}\n```"
        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = fenced
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"


class TestScientistRetry:
    """Tests for the retry-on-validation-failure behavior."""

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_retry_on_invalid_json(self, mock_query, tmp_path):
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = MagicMock(spec=ResultMessage)
            if call_count == 1:
                msg.result = "not json"
            else:
                msg.result = json.dumps(SAMPLE_PLAN)
            yield msg

        mock_query.side_effect = fake_query
        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_exhausts_retries_raises(self, mock_query, tmp_path):
        async def fake_query(**kwargs):
            msg = MagicMock(spec=ResultMessage)
            msg.result = "always bad"
            yield msg

        mock_query.side_effect = fake_query
        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(OutputValidationError, match="Scientist"):
            await run_scientist(
                analysis={},
                notebook_path=notebook_path,
                version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_revision_retry(self, mock_query, tmp_path):
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            msg = MagicMock(spec=ResultMessage)
            if call_count == 1:
                msg.result = '{"incomplete": true}'
            else:
                msg.result = json.dumps(SAMPLE_PLAN)
            yield msg

        mock_query.side_effect = fake_query
        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        result = await run_scientist_revision(
            original_plan=SAMPLE_PLAN,
            concern_ledger=SAMPLE_LEDGER,
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"
        assert call_count == 2


class TestFormatPredictionsForPrompt:
    def test_none_returns_placeholder(self):
        result = _format_predictions_for_prompt(None)
        assert "no prediction history" in result

    def test_empty_list_returns_placeholder(self):
        result = _format_predictions_for_prompt([])
        assert "no prediction history" in result

    def test_single_pending_prediction(self):
        history = [
            PredictionRecord(
                iteration_prescribed=1,
                prediction="noise is additive",
                diagnostic="compute residual-x correlation",
                if_confirmed="OLS is appropriate",
                if_refuted="switch to WLS",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        assert "PENDING" in result
        assert "noise is additive" in result
        assert "Diagnostic:" in result
        assert "If confirmed:" in result
        assert "If refuted:" in result

    def test_resolved_prediction_shows_evidence(self):
        history = [
            PredictionRecord(
                iteration_prescribed=1,
                iteration_evaluated=1,
                prediction="spline fits better locally",
                diagnostic="compare regional RMSE",
                if_confirmed="focus on local fit",
                if_refuted="problem is elsewhere",
                outcome="confirmed",
                evidence="spline RMSE=0.31, polynomial RMSE=0.58",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        assert "CONFIRMED" in result
        assert "spline RMSE=0.31" in result
        assert "focus on local fit" in result

    def test_trajectory_chain(self):
        history = [
            PredictionRecord(
                pred_id="1.1",
                iteration_prescribed=1,
                iteration_evaluated=1,
                prediction="spline fits better locally",
                diagnostic="compare regional RMSE",
                if_confirmed="focus on local fit",
                if_refuted="problem is elsewhere",
                outcome="confirmed",
                evidence="RMSE 0.31 vs 0.58",
            ),
            PredictionRecord(
                pred_id="2.1",
                iteration_prescribed=2,
                prediction="boundary constraints reduce edge error",
                diagnostic="measure error at x boundaries",
                if_confirmed="boundary solved",
                if_refuted="need different approach",
                follows_from="1.1",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        # Both should appear, child indented under parent
        assert "spline fits better locally" in result
        assert "boundary constraints reduce edge error" in result

    def test_orphaned_follows_from_becomes_root(self):
        history = [
            PredictionRecord(
                iteration_prescribed=2,
                prediction="re-test after structural change",
                diagnostic="profile again",
                if_confirmed="now works",
                if_refuted="still broken",
                follows_from="nonexistent prediction",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        assert "re-test after structural change" in result

    def test_mixed_resolved_and_pending(self):
        history = [
            PredictionRecord(
                iteration_prescribed=1,
                iteration_evaluated=1,
                prediction="A",
                diagnostic="d",
                if_confirmed="ok",
                if_refuted="bad",
                outcome="refuted",
                evidence="did not hold",
            ),
            PredictionRecord(
                iteration_prescribed=2,
                prediction="B",
                diagnostic="d",
                if_confirmed="ok",
                if_refuted="bad",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        assert "REFUTED" in result
        assert "PENDING" in result

    def test_pred_id_shown_when_present(self):
        history = [
            PredictionRecord(
                pred_id="1.1",
                iteration_prescribed=1,
                prediction="test prediction",
                diagnostic="d",
                if_confirmed="ok",
                if_refuted="bad",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        assert "[1.1]" in result

    def test_falls_back_to_version_when_no_pred_id(self):
        history = [
            PredictionRecord(
                iteration_prescribed=2,
                prediction="old prediction",
                diagnostic="d",
                if_confirmed="ok",
                if_refuted="bad",
            ),
        ]
        result = _format_predictions_for_prompt(history)
        assert "[v02]" in result


class TestGoalInPrompts:
    """Verify that the goal placeholder is present and populated in prompts."""

    def test_goal_in_scientist_user(self):
        from auto_scientist.prompts.scientist import SCIENTIST_USER

        prompt = SCIENTIST_USER.format(
            goal="discover causal relationships",
            domain_knowledge="dk",
            prediction_history="ph",
            notebook_content="nb",
            analysis_json="{}",
            version="v01",
        )
        assert "discover causal relationships" in prompt
        assert "<goal>" in prompt

    def test_goal_in_scientist_revision_user(self):
        from auto_scientist.prompts.scientist import SCIENTIST_REVISION_USER

        prompt = SCIENTIST_REVISION_USER.format(
            goal="find classification rules",
            domain_knowledge="dk",
            prediction_history="ph",
            notebook_content="nb",
            analysis_json="{}",
            original_plan="{}",
            concern_ledger="[]",
            version="v01",
        )
        assert "find classification rules" in prompt
        assert "<goal>" in prompt
