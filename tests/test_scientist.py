"""Tests for the Scientist agent."""

import json
from unittest.mock import MagicMock, patch

import pytest
from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

from auto_scientist.agents.scientist import (
    _format_criteria_for_prompt,
    run_scientist,
    run_scientist_revision,
)
from auto_scientist.config import SuccessCriterion
from auto_scientist.sdk_utils import OutputValidationError


class TestFormatCriteriaForPrompt:
    def test_none_returns_placeholder(self):
        result = _format_criteria_for_prompt(None)
        assert "no top-level success criteria" in result

    def test_empty_list_returns_placeholder(self):
        result = _format_criteria_for_prompt([])
        assert "no top-level success criteria" in result

    def test_target_min_only(self):
        sc = SuccessCriterion(name="acc", description="accuracy", metric_key="acc", target_min=0.9)
        result = _format_criteria_for_prompt([sc])
        assert ">= 0.9" in result

    def test_target_max_only(self):
        sc = SuccessCriterion(name="loss", description="low loss", metric_key="loss", target_max=0.1)
        result = _format_criteria_for_prompt([sc])
        assert "<= 0.1" in result

    def test_both_bounds(self):
        sc = SuccessCriterion(
            name="f1", description="f1 score", metric_key="f1",
            target_min=0.8, target_max=1.0,
        )
        result = _format_criteria_for_prompt([sc])
        assert "[0.8, 1.0]" in result

    def test_required_vs_optional_labels(self):
        req = SuccessCriterion(name="a", description="d", metric_key="a", required=True)
        opt = SuccessCriterion(name="b", description="d", metric_key="b", required=False)
        result = _format_criteria_for_prompt([req, opt])
        assert "REQUIRED" in result
        assert "optional" in result

    def test_multiple_numbered(self):
        criteria = [
            SuccessCriterion(name=f"c{i}", description="d", metric_key=f"c{i}")
            for i in range(3)
        ]
        result = _format_criteria_for_prompt(criteria)
        assert result.startswith("1.")
        assert "2." in result
        assert "3." in result


SAMPLE_PLAN = {
    "hypothesis": "test hypothesis",
    "strategy": "incremental",
    "changes": [{"what": "do thing", "why": "because", "how": "like this", "priority": 1}],
    "expected_impact": "improvement",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "First hypothesis\n\nTesting incremental approach",
    "success_criteria": [
        {"name": "metric", "description": "desc", "metric_key": "m", "condition": "> 0.5"}
    ],
}


class TestRunScientist:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
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
            analysis={"success_score": 50},
            notebook_path=notebook_path,
            version="v01",
        )

        assert result["hypothesis"] == "test hypothesis"
        assert result["strategy"] == "incremental"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_missing_notebook_uses_fallback(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "nonexistent.md"

        result = await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
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
                analysis={}, notebook_path=notebook_path, version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_web_search_only(self, mock_query, tmp_path):
        """Scientist should only have WebSearch (no code/file tools)."""
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
    @patch("auto_scientist.sdk_utils.query")
    async def test_populates_message_buffer(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            analysis={}, notebook_path=notebook_path, version="v01",
            message_buffer=buf,
        )
        assert len(buf) == 1
        assert "Planning hypothesis..." in buf[0]

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_revision_populates_message_buffer(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            debate_transcript=[{"role": "critic", "content": "weak"}],
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
    @patch("auto_scientist.sdk_utils.query")
    async def test_exploration_plan(self, mock_query, tmp_path):
        exploration_plan = {
            "hypothesis": "Data exploration to establish baselines and identify patterns",
            "strategy": "exploratory",
            "changes": [
                {"what": "Compute distributions", "why": "Understand data shape",
                 "how": "Histograms and summary stats", "priority": 1},
            ],
            "expected_impact": "Baseline understanding of the dataset",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "Data exploration\n\nFirst look at the data",
            "success_criteria": [],
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
        assert "top_level_criteria" not in result or not result.get("top_level_criteria")


class TestRunScientistCriteriaDefinition:
    """Rich analysis + no criteria -> plan includes top_level_criteria."""

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_defines_criteria(self, mock_query, tmp_path):
        plan_with_criteria = {
            "hypothesis": "Polynomial fit will model the observed pattern",
            "strategy": "structural",
            "changes": [
                {"what": "Fit polynomial", "why": "Data shows curve",
                 "how": "np.polyfit degree 2-5", "priority": 1},
            ],
            "expected_impact": "R-squared above 0.9",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "First hypothesis\n\nDefining initial approach",
            "success_criteria": [
                {"name": "R-squared", "description": "Fit quality",
                 "metric_key": "r_squared", "condition": "> 0.9"},
            ],
            "top_level_criteria": [
                {"name": "Final R-squared", "description": "Investigation goal",
                 "metric_key": "r_squared", "condition": "> 0.95"},
            ],
        }

        from claude_code_sdk import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(plan_with_criteria)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook\n## v00 exploration results")

        result = await run_scientist(
            analysis={"success_score": None, "observations": ["200 rows, polynomial shape"]},
            notebook_path=notebook_path,
            version="v01",
        )
        assert "top_level_criteria" in result
        assert len(result["top_level_criteria"]) == 1
        assert result["top_level_criteria"][0]["condition"] == "> 0.95"


class TestRunScientistCriteriaRevision:
    """Existing criteria -> plan may include criteria_revision."""

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_revises_criteria(self, mock_query, tmp_path):
        plan_with_revision = {
            "hypothesis": "Lower target is more realistic given noise",
            "strategy": "incremental",
            "changes": [
                {"what": "Add regularization", "why": "Reduce overfitting",
                 "how": "L2 penalty", "priority": 1},
            ],
            "expected_impact": "Stable R-squared around 0.90",
            "should_stop": False,
            "stop_reason": None,
            "notebook_entry": "Adjusting targets\n\nRevising criteria based on evidence",
            "success_criteria": [
                {"name": "R-squared", "description": "Fit quality",
                 "metric_key": "r_squared", "condition": "> 0.85"},
            ],
            "criteria_revision": {
                "changes": (
                    "Lowered R-squared target from 0.95 to 0.90"
                    " because noise floor limits achievable accuracy"
                ),
                "revised_criteria": [
                    {"name": "Final R-squared", "description": "Investigation goal",
                     "metric_key": "r_squared", "condition": "> 0.90"},
                ],
            },
        }

        from claude_code_sdk import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(plan_with_revision)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        existing_criteria = [
            SuccessCriterion(name="Final R-squared", description="goal",
                             metric_key="r_squared", target_min=0.95),
        ]
        result = await run_scientist(
            analysis={"success_score": 75},
            notebook_path=notebook_path,
            version="v03",
            success_criteria=existing_criteria,
        )
        assert "criteria_revision" in result
        assert result["criteria_revision"]["revised_criteria"][0]["condition"] == "> 0.90"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_accepts_success_criteria_param(self, mock_query, tmp_path):
        """run_scientist should accept a success_criteria parameter."""
        from claude_code_sdk import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        criteria = [
            SuccessCriterion(name="acc", description="accuracy",
                             metric_key="accuracy", target_min=0.9),
        ]
        await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v02",
            success_criteria=criteria,
        )
        # Criteria should appear in the prompt
        assert "accuracy" in captured_prompt["prompt"]


class TestRunScientistRevision:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_returns_revised_plan(self, mock_query, tmp_path):
        from claude_code_sdk import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        transcript = [
            {"role": "critic", "content": "This is weak"},
            {"role": "scientist", "content": "I disagree"},
        ]

        result = await run_scientist_revision(
            original_plan=SAMPLE_PLAN,
            debate_transcript=transcript,
            analysis={"success_score": 50},
            notebook_path=notebook_path,
            version="v01",
        )

        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
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
                debate_transcript=[],
                analysis={},
                notebook_path=notebook_path,
                version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_fallback_to_assistant_text(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            debate_transcript=[{"role": "critic", "content": "test"}],
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"


class TestRunScientistAssistantFallback:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_run_scientist_fallback_to_assistant_text(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            analysis={}, notebook_path=notebook_path, version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_run_scientist_markdown_fenced_response(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            analysis={}, notebook_path=notebook_path, version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"


class TestScientistRetry:
    """Tests for the retry-on-validation-failure behavior."""

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
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
            analysis={}, notebook_path=notebook_path, version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_exhausts_retries_raises(self, mock_query, tmp_path):
        async def fake_query(**kwargs):
            msg = MagicMock(spec=ResultMessage)
            msg.result = "always bad"
            yield msg

        mock_query.side_effect = fake_query
        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(OutputValidationError, match="Scientist"):
            await run_scientist(
                analysis={}, notebook_path=notebook_path, version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
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
            debate_transcript=[{"role": "critic", "content": "weak"}],
            analysis={},
            notebook_path=notebook_path,
            version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"
        assert call_count == 2


class TestScientistStructuredOutput:
    """Tests for the direct API structured output path."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query_anthropic")
    async def test_anthropic_model_uses_direct_api(self, mock_query, tmp_path):
        mock_query.return_value = json.dumps(SAMPLE_PLAN)
        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v01",
            model="claude-sonnet-4-6",
            use_structured_output=True,
        )
        assert result["hypothesis"] == "test hypothesis"
        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs.get("system_prompt") is not None
        assert call_kwargs.get("response_schema") is not None

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query_openai")
    async def test_openai_model_uses_direct_api(self, mock_query, tmp_path):
        mock_query.return_value = json.dumps(SAMPLE_PLAN)
        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v01",
            model="gpt-5.4",
            use_structured_output=True,
        )
        assert result["hypothesis"] == "test hypothesis"
        mock_query.assert_called_once()

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query_google")
    async def test_google_model_uses_direct_api(self, mock_query, tmp_path):
        mock_query.return_value = json.dumps(SAMPLE_PLAN)
        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v01",
            model="gemini-2.5-pro",
            use_structured_output=True,
        )
        assert result["hypothesis"] == "test hypothesis"
        mock_query.assert_called_once()

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_utils.query")
    async def test_unknown_model_falls_back_to_sdk(self, mock_query, tmp_path):
        """Unknown provider with use_structured_output falls back to SDK path."""
        async def fake_query(**kwargs):
            msg = MagicMock(spec=ResultMessage)
            msg.result = json.dumps(SAMPLE_PLAN)
            yield msg

        mock_query.side_effect = fake_query
        notebook_path = tmp_path / "notebook.md"

        result = await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v01",
            model="unknown-model",
            use_structured_output=True,
        )
        assert result["hypothesis"] == "test hypothesis"
