# Test Catch-Up Implementation Plan

**Goal:** Add practical test coverage for all untested modules, bottom-up from simple to complex.

**Architecture:** Four tiers: config + model clients (easy wins), agent logic (mock LLM calls, test prompt assembly and parsing), orchestrator (mock agents, test state machine), CLI (Click CliRunner). Each tier commits independently.

**Tech Stack:** pytest, pytest-asyncio, unittest.mock (AsyncMock, patch), tmp_path fixture, Click CliRunner.

---

### Task 1: Test Config Pydantic Models

**Files:**
- Create: `tests/test_config.py`
- Reference: `src/auto_scientist/config.py`

- [ ] **Step 1: Write tests for SuccessCriterion and DomainConfig**

```python
"""Tests for domain configuration schema."""

import pytest
from pydantic import ValidationError

from auto_scientist.config import DomainConfig, SuccessCriterion


class TestSuccessCriterion:
    def test_required_fields(self):
        sc = SuccessCriterion(name="acc", description="accuracy", metric_key="accuracy")
        assert sc.name == "acc"
        assert sc.description == "accuracy"
        assert sc.metric_key == "accuracy"

    def test_defaults(self):
        sc = SuccessCriterion(name="a", description="b", metric_key="c")
        assert sc.target_min is None
        assert sc.target_max is None
        assert sc.required is True

    def test_optional_targets(self):
        sc = SuccessCriterion(
            name="a", description="b", metric_key="c",
            target_min=0.5, target_max=1.0, required=False,
        )
        assert sc.target_min == 0.5
        assert sc.target_max == 1.0
        assert sc.required is False

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            SuccessCriterion(name="a", description="b")  # missing metric_key


class TestDomainConfig:
    def test_required_fields(self):
        dc = DomainConfig(
            name="test", description="Test domain",
            data_paths=["data.csv"],
        )
        assert dc.name == "test"
        assert dc.data_paths == ["data.csv"]

    def test_defaults(self):
        dc = DomainConfig(name="t", description="d", data_paths=[])
        assert dc.run_command == "uv run python -u {script_path}"
        assert dc.run_cwd == "."
        assert dc.run_timeout_minutes == 120
        assert dc.version_prefix == "v"
        assert dc.success_criteria == []
        assert dc.domain_knowledge == ""
        assert dc.protected_paths == []
        assert dc.experiment_dependencies == []

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            DomainConfig(name="t")  # missing description and data_paths

    def test_with_success_criteria(self):
        sc = SuccessCriterion(name="a", description="b", metric_key="c")
        dc = DomainConfig(
            name="t", description="d", data_paths=[],
            success_criteria=[sc],
        )
        assert len(dc.success_criteria) == 1
        assert dc.success_criteria[0].name == "a"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add config model validation tests
```

---

### Task 2: Test Model Clients

**Files:**
- Create: `tests/test_models.py`
- Reference: `src/auto_scientist/models/openai_client.py`, `google_client.py`, `anthropic_client.py`

- [ ] **Step 1: Write tests for all three model clients**

```python
"""Tests for LLM model client wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai


class TestQueryOpenAI:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_standard_call(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-4o", "test prompt")

        assert result == "hello"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_web_search_uses_responses_api(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(output_text="searched result")
        mock_client.responses.create.return_value = mock_response

        result = await query_openai("gpt-4o", "search this", web_search=True)

        assert result == "searched result"
        mock_client.responses.create.assert_called_once()
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert any(t["type"] == "web_search_preview" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_empty_response_returns_empty_string(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-4o", "prompt")

        assert result == ""


class TestQueryGoogle:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_standard_call(self, mock_genai):
        mock_response = MagicMock(text="google response")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-2.5-pro", "test prompt")

        assert result == "google response"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_web_search_adds_google_search_tool(self, mock_genai):
        mock_response = MagicMock(text="searched")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-2.5-pro", "search", web_search=True)

        assert result == "searched"
        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["config"] is not None

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_empty_response_returns_empty_string(self, mock_genai):
        mock_response = MagicMock(text=None)
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-2.5-pro", "prompt")

        assert result == ""


class TestQueryAnthropic:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_standard_call(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_block = MagicMock(text="anthropic response")
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "test prompt")

        assert result == "anthropic response"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_web_search_adds_tool(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_block = MagicMock(text="searched")
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "search", web_search=True)

        assert result == "searched"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert any(t["type"] == "web_search_20250305" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_empty_response_returns_empty_string(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        # No text attribute on blocks
        mock_block = MagicMock(spec=[])
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "prompt")

        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_multiple_text_blocks_joined(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        block1 = MagicMock(text="part1")
        block2 = MagicMock(text="part2")
        mock_response = MagicMock(content=[block1, block2])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "prompt")

        assert result == "part1\npart2"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_models.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add model client wrapper tests
```

---

### Task 3: Test Analyst Agent

**Files:**
- Create: `tests/test_analyst.py`
- Reference: `src/auto_scientist/agents/analyst.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the Analyst agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.agents.analyst import _format_success_criteria, run_analyst
from auto_scientist.config import SuccessCriterion


class TestFormatSuccessCriteria:
    """Tests for the pure helper function."""

    def test_empty_list(self):
        assert _format_success_criteria([]) == "(no success criteria defined)"

    def test_target_min_only(self):
        sc = SuccessCriterion(name="acc", description="accuracy", metric_key="acc", target_min=0.9)
        result = _format_success_criteria([sc])
        assert ">= 0.9" in result
        assert "REQUIRED" in result

    def test_target_max_only(self):
        sc = SuccessCriterion(name="loss", description="loss", metric_key="loss", target_max=0.1)
        result = _format_success_criteria([sc])
        assert "<= 0.1" in result

    def test_both_bounds(self):
        sc = SuccessCriterion(
            name="f1", description="f1 score", metric_key="f1",
            target_min=0.8, target_max=1.0,
        )
        result = _format_success_criteria([sc])
        assert "[0.8, 1.0]" in result

    def test_optional_label(self):
        sc = SuccessCriterion(
            name="extra", description="extra metric", metric_key="extra",
            required=False,
        )
        result = _format_success_criteria([sc])
        assert "optional" in result

    def test_multiple_criteria_numbered(self):
        criteria = [
            SuccessCriterion(name="a", description="da", metric_key="a"),
            SuccessCriterion(name="b", description="db", metric_key="b"),
        ]
        result = _format_success_criteria(criteria)
        assert result.startswith("1.")
        assert "2." in result


class TestRunAnalyst:
    """Tests for the agent runner, mocking the claude_agent_sdk query."""

    def _make_mock_query(self, response_json):
        """Create a mock query that yields a ResultMessage with JSON."""
        result_msg = MagicMock()
        result_msg.result = json.dumps(response_json)
        # Make it look like a ResultMessage
        type(result_msg).__name__ = "ResultMessage"

        async def mock_query_fn(**kwargs):
            yield result_msg

        return mock_query_fn

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_returns_parsed_json(self, mock_query, tmp_path):
        analysis = {
            "success_score": 75,
            "criteria_results": [],
            "key_metrics": {"rmse": 0.5},
            "improvements": ["better"],
            "regressions": [],
            "observations": ["noted"],
            "iteration_criteria_results": [],
        }

        # Mock query as async generator yielding ResultMessage
        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("rmse: 0.5")
        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        result = await run_analyst(
            results_path=results_path,
            plot_paths=[],
            notebook_path=notebook_path,
        )

        assert result["success_score"] == 75
        assert result["key_metrics"]["rmse"] == 0.5

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_handles_markdown_fenced_json(self, mock_query, tmp_path):
        analysis = {"success_score": 50, "criteria_results": [], "key_metrics": {},
                     "improvements": [], "regressions": [], "observations": [],
                     "iteration_criteria_results": []}
        fenced = f"```json\n{json.dumps(analysis)}\n```"

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = fenced

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        result = await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
        )
        assert result["success_score"] == 50

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_raises_on_empty_output(self, mock_query, tmp_path):
        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(RuntimeError, match="returned no output"):
            await run_analyst(
                results_path=results_path, plot_paths=[], notebook_path=notebook_path,
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.analyst.query")
    async def test_missing_results_file_uses_fallback(self, mock_query, tmp_path):
        analysis = {"success_score": 0, "criteria_results": [], "key_metrics": {},
                     "improvements": [], "regressions": [], "observations": [],
                     "iteration_criteria_results": []}

        from auto_scientist.agents.analyst import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(analysis)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        results_path = tmp_path / "nonexistent.txt"
        notebook_path = tmp_path / "notebook.md"

        # Should not raise - uses fallback text
        result = await run_analyst(
            results_path=results_path, plot_paths=[], notebook_path=notebook_path,
        )
        assert result["success_score"] == 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_analyst.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add analyst agent tests
```

---

### Task 4: Test Scientist Agent

**Files:**
- Create: `tests/test_scientist.py`
- Reference: `src/auto_scientist/agents/scientist.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the Scientist agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.scientist import (
    _parse_json_response,
    run_scientist,
    run_scientist_revision,
)


class TestParseJsonResponse:
    """Tests for the pure JSON parsing helper."""

    def test_clean_json(self):
        result = _parse_json_response('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(raw, "test")
        assert result == {"key": "value"}

    def test_markdown_fenced_no_language(self):
        raw = '```\n{"key": "value"}\n```'
        result = _parse_json_response(raw, "test")
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json", "test")

    def test_whitespace_stripped(self):
        result = _parse_json_response('  \n{"key": "value"}\n  ', "test")
        assert result == {"key": "value"}


SAMPLE_PLAN = {
    "hypothesis": "test hypothesis",
    "strategy": "incremental",
    "changes": [{"what": "do thing", "why": "because", "how": "like this", "priority": 1}],
    "expected_impact": "improvement",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "## v01",
    "success_criteria": [
        {"name": "metric", "description": "desc", "metric_key": "m", "condition": "> 0.5"}
    ],
}


class TestRunScientist:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_returns_parsed_plan(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
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
    @patch("auto_scientist.agents.scientist.query")
    async def test_missing_notebook_uses_fallback(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
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
    @patch("auto_scientist.agents.scientist.query")
    async def test_empty_output_raises(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
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
    @patch("auto_scientist.agents.scientist.query")
    async def test_no_tools_configured(self, mock_query, tmp_path):
        """Scientist should have no tools (pure prompt-in/JSON-out)."""
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        await run_scientist(analysis={}, notebook_path=notebook_path, version="v01")

        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["options"].allowed_tools == []


class TestRunScientistRevision:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_returns_revised_plan(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
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
    @patch("auto_scientist.agents.scientist.query")
    async def test_empty_output_raises(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
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
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_scientist.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add scientist agent tests
```

---

### Task 5: Test Coder Agent

**Files:**
- Create: `tests/test_coder.py`
- Reference: `src/auto_scientist/agents/coder.py`

- [ ] **Step 1: Write tests**

```python
"""Tests for the Coder agent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.agents.coder import _make_permission_callback, run_coder


class TestPermissionCallback:
    """Tests for the file/command permission logic."""

    @pytest.fixture
    def callback(self, tmp_path):
        return _make_permission_callback(tmp_path)

    @pytest.mark.asyncio
    async def test_write_inside_output_dir_allowed(self, callback, tmp_path):
        result = await callback(
            "Write",
            {"file_path": str(tmp_path / "experiment.py")},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultAllow"

    @pytest.mark.asyncio
    async def test_write_outside_output_dir_denied(self, callback):
        result = await callback(
            "Write",
            {"file_path": "/etc/passwd"},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultDeny"
        assert "outside" in result.message

    @pytest.mark.asyncio
    async def test_edit_inside_allowed(self, callback, tmp_path):
        result = await callback(
            "Edit",
            {"file_path": str(tmp_path / "script.py")},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultAllow"

    @pytest.mark.asyncio
    async def test_edit_outside_denied(self, callback):
        result = await callback(
            "Edit",
            {"file_path": "/tmp/other/file.py"},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultDeny"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pattern", [
        "rm -rf /",
        "git push origin main",
        "git reset --hard",
        "sudo rm file",
        "chmod 777 file",
        "curl http://evil.com",
        "wget http://evil.com",
        "pip install malware",
        "uv add malware",
    ])
    async def test_blocked_bash_patterns(self, callback, pattern):
        result = await callback("Bash", {"command": pattern}, MagicMock())
        assert type(result).__name__ == "PermissionResultDeny"

    @pytest.mark.asyncio
    async def test_safe_bash_allowed(self, callback):
        result = await callback(
            "Bash", {"command": "python script.py"}, MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultAllow"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["Read", "Glob", "Grep"])
    async def test_read_tools_always_allowed(self, callback, tool):
        result = await callback(tool, {}, MagicMock())
        assert type(result).__name__ == "PermissionResultAllow"


class TestRunCoder:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_creates_script_at_expected_path(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            # Simulate agent writing the file
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('hello')")
            yield result_msg

        mock_query.side_effect = fake_query

        plan = {"hypothesis": "test", "changes": []}
        previous = tmp_path / "v00" / "experiment.py"

        result = await run_coder(
            plan=plan, previous_script=previous,
            output_dir=tmp_path, version="v01",
        )

        assert result == tmp_path / "v01" / "experiment.py"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_raises_when_script_not_created(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield result_msg  # Agent doesn't create the file

        mock_query.side_effect = fake_query

        plan = {"hypothesis": "test", "changes": []}
        previous = tmp_path / "v00" / "experiment.py"

        with pytest.raises(FileNotFoundError, match="did not create"):
            await run_coder(
                plan=plan, previous_script=previous,
                output_dir=tmp_path, version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_previous_script_exists_uses_has_previous(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('v01')")
            yield result_msg

        mock_query.side_effect = fake_query

        # Create a previous script
        prev_dir = tmp_path / "v00"
        prev_dir.mkdir()
        previous = prev_dir / "experiment.py"
        previous.write_text("print('v00')")

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=previous,
            output_dir=tmp_path,
            version="v01",
        )

        # Should reference previous script path in prompt
        assert str(previous) in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_no_previous_script_uses_no_previous(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('v01')")
            yield result_msg

        mock_query.side_effect = fake_query

        previous = tmp_path / "nonexistent" / "experiment.py"

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=previous,
            output_dir=tmp_path,
            version="v01",
        )

        # Should contain the "no previous" text
        assert "first experiment" in captured_prompt["prompt"].lower() or "from scratch" in captured_prompt["prompt"].lower()
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_coder.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add coder agent tests
```

---

### Task 6: Test Discovery and Report Agents

**Files:**
- Create: `tests/test_discovery.py`, `tests/test_report.py`
- Reference: `src/auto_scientist/agents/discovery.py`, `src/auto_scientist/agents/report.py`

- [ ] **Step 1: Write discovery tests**

```python
"""Tests for the Discovery agent."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.agents.discovery import run_discovery
from auto_scientist.state import ExperimentState


def test_run_discovery_is_async():
    assert asyncio.iscoroutinefunction(run_discovery)


class TestRunDiscovery:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.discovery.ClaudeSDKClient")
    async def test_interactive_mode_includes_ask_user(self, mock_client_cls, tmp_path):
        """Interactive mode should add AskUserQuestion to allowed tools."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.query = AsyncMock()
        mock_client.receive_response = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration)
        ))

        # Create expected output files
        config_path = tmp_path / "domain_config.json"
        config_data = {
            "name": "test", "description": "Test domain",
            "data_paths": ["data.csv"],
        }
        config_path.write_text(json.dumps(config_data))
        script_path = tmp_path / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hello')")

        state = ExperimentState(domain="auto", goal="test goal")

        config, script = await run_discovery(
            state=state, data_path=tmp_path / "data.csv",
            output_dir=tmp_path, interactive=True,
        )

        # Verify AskUserQuestion was in the options
        options = mock_client_cls.call_args.kwargs["options"]
        assert "AskUserQuestion" in options.allowed_tools

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.discovery.ClaudeSDKClient")
    async def test_missing_config_raises(self, mock_client_cls, tmp_path):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.query = AsyncMock()
        mock_client.receive_response = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration)
        ))

        state = ExperimentState(domain="auto", goal="test goal")

        with pytest.raises(FileNotFoundError, match="domain config"):
            await run_discovery(
                state=state, data_path=tmp_path / "data.csv",
                output_dir=tmp_path,
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.discovery.ClaudeSDKClient")
    async def test_missing_script_raises(self, mock_client_cls, tmp_path):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.query = AsyncMock()
        mock_client.receive_response = AsyncMock(return_value=AsyncMock(
            __aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration)
        ))

        # Create config but not the script
        config_path = tmp_path / "domain_config.json"
        config_data = {"name": "test", "description": "desc", "data_paths": []}
        config_path.write_text(json.dumps(config_data))

        state = ExperimentState(domain="auto", goal="test goal")

        with pytest.raises(FileNotFoundError, match="experiment script"):
            await run_discovery(
                state=state, data_path=tmp_path / "data.csv",
                output_dir=tmp_path,
            )
```

- [ ] **Step 2: Write report tests**

```python
"""Tests for the Report agent."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.report import run_report
from auto_scientist.state import ExperimentState


def test_run_report_is_async():
    assert asyncio.iscoroutinefunction(run_report)


class TestRunReport:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.query")
    async def test_creates_report_at_expected_path(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            # Simulate agent writing the report
            report_path = tmp_path / "report.md"
            report_path.write_text("# Final Report")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(
            domain="test", goal="test goal",
            iteration=5, best_version="v03", best_score=85,
        )
        notebook_path = tmp_path / "lab_notebook.md"
        notebook_path.write_text("# Lab Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )

        assert result == tmp_path / "report.md"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.query")
    async def test_raises_when_report_not_created(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield result_msg  # Agent doesn't create the file

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.md"

        with pytest.raises(FileNotFoundError, match="did not create"):
            await run_report(
                state=state, notebook_path=notebook_path, output_dir=tmp_path,
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.query")
    async def test_prompt_includes_state_fields(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            report_path = tmp_path / "report.md"
            report_path.write_text("# Report")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(
            domain="spo2", goal="predict oxygen levels",
            iteration=10, best_version="v07", best_score=92,
        )
        notebook_path = tmp_path / "lab_notebook.md"
        notebook_path.write_text("# Notebook")

        await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        prompt = captured_prompt["prompt"]
        assert "spo2" in prompt
        assert "predict oxygen levels" in prompt
        assert "v07" in prompt
        assert "92" in prompt
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/test_discovery.py tests/test_report.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```
test: add discovery and report agent tests
```

---

### Task 7: Test Orchestrator

**Files:**
- Create: `tests/test_orchestrator.py`
- Reference: `src/auto_scientist/orchestrator.py`

This is the largest task. The orchestrator is a state machine, so we mock all agent calls and test transitions.

- [ ] **Step 1: Write orchestrator tests**

```python
"""Tests for the orchestrator state machine."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.config import DomainConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.runner import RunResult
from auto_scientist.state import ExperimentState, VersionEntry


@pytest.fixture
def base_state():
    return ExperimentState(domain="test", goal="test goal", phase="ingestion")


@pytest.fixture
def config():
    return DomainConfig(
        name="test", description="Test domain", data_paths=["data.csv"],
        domain_knowledge="test knowledge",
    )


@pytest.fixture
def orchestrator(base_state, tmp_path, config):
    return Orchestrator(
        state=base_state,
        data_path=tmp_path / "data.csv",
        output_dir=tmp_path / "experiments",
        config=config,
    )


class TestOrchestratorInit:
    def test_defaults(self, base_state, tmp_path):
        o = Orchestrator(state=base_state, data_path=tmp_path, output_dir=tmp_path)
        assert o.max_iterations == 20
        assert o.critic_models == []
        assert o.debate_rounds == 2
        assert o.max_consecutive_failures == 5

    def test_custom_values(self, base_state, tmp_path):
        o = Orchestrator(
            state=base_state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=5, critic_models=["openai:gpt-4o"],
            debate_rounds=3, max_consecutive_failures=2,
        )
        assert o.max_iterations == 5
        assert o.critic_models == ["openai:gpt-4o"]
        assert o.debate_rounds == 3


class TestEvaluate:
    def test_none_result_records_failure(self, orchestrator):
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator._evaluate(None, entry)
        assert entry.status == "failed"
        assert orchestrator.state.consecutive_failures == 1

    def test_timed_out_records_failure(self, orchestrator):
        result = RunResult(success=False, timed_out=True)
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator._evaluate(result, entry)
        assert entry.status == "failed"
        assert orchestrator.state.consecutive_failures == 1

    def test_nonzero_exit_records_failure(self, orchestrator):
        result = RunResult(success=False, return_code=1)
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator._evaluate(result, entry)
        assert entry.status == "failed"
        assert orchestrator.state.consecutive_failures == 1

    def test_success_records_completion(self, orchestrator, tmp_path):
        result = RunResult(success=True, stdout="output", return_code=0)
        script_path = tmp_path / "experiments" / "v01" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('ok')")
        # Create results.txt so it gets set
        results_path = script_path.parent / "results.txt"
        results_path.write_text("results")

        entry = VersionEntry(version="v01", iteration=1, script_path=str(script_path))
        orchestrator._evaluate(result, entry)
        assert entry.status == "completed"
        assert orchestrator.state.consecutive_failures == 0
        assert entry.results_path == str(results_path)


class TestNotebookContent:
    def test_returns_content_when_exists(self, orchestrator, tmp_path):
        notebook = tmp_path / "experiments" / "lab_notebook.md"
        notebook.parent.mkdir(parents=True, exist_ok=True)
        notebook.write_text("# Notebook")
        assert orchestrator._notebook_content() == "# Notebook"

    def test_returns_empty_when_missing(self, orchestrator):
        assert orchestrator._notebook_content() == ""


class TestRunIngestion:
    @pytest.mark.asyncio
    async def test_raises_without_data_path(self, tmp_path):
        state = ExperimentState(domain="test", goal="g", phase="ingestion")
        o = Orchestrator(state=state, data_path=None, output_dir=tmp_path)
        with pytest.raises(ValueError, match="Cannot run ingestion"):
            await o._run_ingestion()

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.run_ingestor", new_callable=AsyncMock)
    async def test_returns_canonical_data_dir(self, mock_ingestor, tmp_path):
        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)
        mock_ingestor.return_value = canonical

        state = ExperimentState(
            domain="test", goal="g", phase="ingestion",
            data_path=str(tmp_path / "raw.csv"),
        )
        o = Orchestrator(state=state, data_path=tmp_path / "raw.csv", output_dir=tmp_path / "experiments")
        result = await o._run_ingestion()

        assert result == canonical
        mock_ingestor.assert_called_once()


class TestPhaseTransitions:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.run_ingestor", new_callable=AsyncMock)
    @patch("auto_scientist.agents.discovery.run_discovery", new_callable=AsyncMock)
    @patch("auto_scientist.agents.coder.run_coder", new_callable=AsyncMock)
    async def test_max_iterations_triggers_report(
        self, mock_coder, mock_discovery, mock_ingestor, tmp_path
    ):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=20,
        )
        config = DomainConfig(name="t", description="d", data_paths=[])
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20, config=config,
        )

        # Mock _run_report to not actually run
        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_consecutive_failures_triggers_report(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            consecutive_failures=5,
        )
        config = DomainConfig(name="t", description="d", data_paths=[])
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_consecutive_failures=5, config=config,
        )

        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_resume_from_iteration_skips_ingestion_and_discovery(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=20,  # Already at max
        )
        config = DomainConfig(name="t", description="d", data_paths=[])
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20, config=config,
        )

        with patch.object(o, "_run_report", new_callable=AsyncMock) as mock_report:
            await o.run()

        mock_report.assert_called_once()
        assert state.phase == "stopped"


class TestRunIteration:
    @pytest.mark.asyncio
    async def test_scientist_stop_sets_report_phase(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.versions = [
            VersionEntry(version="v00", iteration=0, script_path="/tmp/s.py",
                        results_path=str(tmp_path / "results.txt")),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": True, "stop_reason": "goal reached"}

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
        ):
            await orchestrator._run_iteration()

        assert orchestrator.state.phase == "report"

    @pytest.mark.asyncio
    async def test_no_critics_skips_debate(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.critic_models = []
        orchestrator.state.phase = "iteration"
        orchestrator.state.versions = [
            VersionEntry(version="v00", iteration=0, script_path="/tmp/s.py",
                        results_path=str(tmp_path / "results.txt")),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v01" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")

        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(orchestrator, "_run_debate", new_callable=AsyncMock, return_value=None) as mock_debate,
            patch.object(orchestrator, "_run_scientist_revision", new_callable=AsyncMock, return_value=None),
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock, return_value=run_result),
        ):
            await orchestrator._run_iteration()

        mock_debate.assert_called_once_with(plan)
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add orchestrator state machine tests
```

---

### Task 8: Test CLI

**Files:**
- Create: `tests/test_cli.py`
- Reference: `src/auto_scientist/cli.py`

- [ ] **Step 1: Write CLI tests**

```python
"""Tests for the CLI entry point."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from auto_scientist.cli import cli, load_domain_config
from auto_scientist.config import DomainConfig
from auto_scientist.state import ExperimentState


class TestLoadDomainConfig:
    @patch("auto_scientist.cli.importlib.import_module")
    def test_loads_config_and_knowledge(self, mock_import):
        mock_config_mod = MagicMock()
        mock_config_mod.TEST_CONFIG = DomainConfig(
            name="test", description="Test", data_paths=[],
        )
        mock_prompts_mod = MagicMock()
        mock_prompts_mod.TEST_DOMAIN_KNOWLEDGE = "domain knowledge text"

        def import_side_effect(name):
            if name == "domains.test.config":
                return mock_config_mod
            if name == "domains.test.prompts":
                return mock_prompts_mod
            raise ModuleNotFoundError(name)

        mock_import.side_effect = import_side_effect

        config = load_domain_config("test")

        assert config.name == "test"
        assert config.domain_knowledge == "domain knowledge text"

    @patch("auto_scientist.cli.importlib.import_module")
    def test_missing_prompts_module_uses_empty_knowledge(self, mock_import):
        mock_config_mod = MagicMock()
        mock_config_mod.TEST_CONFIG = DomainConfig(
            name="test", description="Test", data_paths=[],
        )

        def import_side_effect(name):
            if name == "domains.test.config":
                return mock_config_mod
            raise ModuleNotFoundError(name)

        mock_import.side_effect = import_side_effect

        config = load_domain_config("test")

        assert config.domain_knowledge == ""


class TestStatusCommand:
    def test_displays_state_info(self, tmp_path):
        state = ExperimentState(
            domain="spo2", goal="test", phase="iteration",
            iteration=5, best_version="v03", best_score=75,
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--state", str(state_path)])

        assert result.exit_code == 0
        assert "spo2" in result.output
        assert "iteration" in result.output
        assert "5" in result.output
        assert "v03" in result.output
        assert "75" in result.output


class TestRunCommand:
    @patch("auto_scientist.cli.asyncio.run")
    @patch("auto_scientist.cli.Orchestrator")
    def test_required_options(self, mock_orch, mock_async_run, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test goal",
        ])

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].goal == "test goal"
        mock_async_run.assert_called_once()

    def test_missing_data_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--goal", "test"])
        assert result.exit_code != 0

    def test_missing_goal_fails(self, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--data", str(data_file)])
        assert result.exit_code != 0


class TestResumeCommand:
    @patch("auto_scientist.cli.asyncio.run")
    @patch("auto_scientist.cli.Orchestrator")
    def test_loads_state_and_creates_orchestrator(self, mock_orch, mock_async_run, tmp_path):
        state = ExperimentState(domain="test", goal="g", phase="iteration")
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(state_path)])

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].domain == "test"
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```
test: add CLI command tests
```

---

### Task 9: Run Full Suite and Final Commit

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 2: Final commit if any fixups were needed**

```
test: fix test issues from full suite run
```
