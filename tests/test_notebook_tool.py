"""Tests for the read_notebook MCP tool and notebook parser."""

import json
import sys
from pathlib import Path

import pytest

from auto_scientist.agents.notebook_tool import (
    _handle_read_notebook,
    _normalize_args,
    build_notebook_mcp_server,
    format_notebook_toc,
)
from auto_scientist.notebook import (
    NOTEBOOK_FILENAME,
    append_entry,
    parse_notebook_entries,
)


@pytest.fixture
def sample_notebook(tmp_path: Path) -> Path:
    """A notebook with entries covering all sources and multiple versions."""
    notebook_path = tmp_path / NOTEBOOK_FILENAME
    append_entry(
        notebook_path,
        "Domain setup\n\nAlloy design canonicalized from raw CSV",
        version="ingestion",
        source="ingestor",
    )
    append_entry(
        notebook_path,
        "Initial exploration\n\nChecked composition ranges and target distributions",
        version="v00",
        source="scientist",
    )
    append_entry(
        notebook_path,
        "Hardness correlation with Cr fraction\n\nFit linear model with Cr features",
        version="v01",
        source="scientist",
    )
    append_entry(
        notebook_path,
        "Revised after debate\n\nAddressed Falsification Expert's concern on CV",
        version="v01",
        source="revision",
    )
    append_entry(
        notebook_path,
        "Stop withdrawn: insufficient evidence\n\nAssessor flagged missing coverage on cost",
        version="v02",
        source="stop_gate",
    )
    return notebook_path


@pytest.fixture
def sample_entries(sample_notebook: Path) -> list[dict[str, str]]:
    return parse_notebook_entries(sample_notebook)


# ---------------------------------------------------------------------------
# parse_notebook_entries
# ---------------------------------------------------------------------------


class TestParseNotebookEntries:
    def test_happy_path(self, sample_notebook: Path):
        entries = parse_notebook_entries(sample_notebook)
        assert len(entries) == 5
        first = entries[0]
        assert first["version"] == "ingestion"
        assert first["source"] == "ingestor"
        assert first["title"] == "Domain setup"
        assert "canonicalized" in first["content"]

    def test_preserves_order(self, sample_notebook: Path):
        entries = parse_notebook_entries(sample_notebook)
        versions = [e["version"] for e in entries]
        sources = [e["source"] for e in entries]
        assert versions == ["ingestion", "v00", "v01", "v01", "v02"]
        assert sources == ["ingestor", "scientist", "scientist", "revision", "stop_gate"]

    def test_returns_empty_when_missing(self, tmp_path: Path):
        assert parse_notebook_entries(tmp_path / "missing.xml") == []

    def test_returns_empty_when_file_is_empty(self, tmp_path: Path):
        path = tmp_path / "empty.xml"
        path.write_text("")
        assert parse_notebook_entries(path) == []

    def test_returns_empty_on_malformed_xml(self, tmp_path: Path):
        path = tmp_path / "broken.xml"
        path.write_text("<lab_notebook><entry version='v00'")
        assert parse_notebook_entries(path) == []

    def test_title_only_entry(self, tmp_path: Path):
        path = tmp_path / NOTEBOOK_FILENAME
        append_entry(path, "Just a title", version="v00", source="scientist")
        entries = parse_notebook_entries(path)
        assert len(entries) == 1
        assert entries[0]["title"] == "Just a title"
        assert entries[0]["content"] == ""


# ---------------------------------------------------------------------------
# format_notebook_toc
# ---------------------------------------------------------------------------


class TestFormatNotebookToc:
    def test_empty_list(self):
        assert "no notebook entries yet" in format_notebook_toc([]).lower()

    def test_none(self):
        assert "no notebook entries yet" in format_notebook_toc(None).lower()

    def test_single_entry(self):
        toc = format_notebook_toc(
            [{"version": "v00", "source": "scientist", "title": "First plan", "content": "..."}]
        )
        assert "NOTEBOOK TOC" in toc
        assert "[v00 scientist] First plan" in toc

    def test_multiple_entries(self, sample_entries):
        toc = format_notebook_toc(sample_entries)
        assert "NOTEBOOK TOC" in toc
        assert "[ingestion ingestor] Domain setup" in toc
        assert "[v00 scientist] Initial exploration" in toc
        assert "[v01 scientist] Hardness correlation with Cr fraction" in toc
        assert "[v01 revision] Revised after debate" in toc
        assert "[v02 stop_gate] Stop withdrawn: insufficient evidence" in toc

    def test_missing_title_falls_back(self):
        toc = format_notebook_toc(
            [{"version": "v00", "source": "scientist", "title": "", "content": "body"}]
        )
        assert "(untitled)" in toc

    def test_is_significantly_smaller_than_full(self, sample_entries):
        """TOC must be notably smaller than the full content dump."""
        toc = format_notebook_toc(sample_entries)
        full = "\n".join(f"{e['title']}\n{e['content']}" for e in sample_entries)
        assert len(toc) < len(full)


# ---------------------------------------------------------------------------
# _handle_read_notebook
# ---------------------------------------------------------------------------


class TestHandleReadNotebook:
    @pytest.mark.asyncio
    async def test_empty_entries(self):
        result = await _handle_read_notebook([], {})
        text = result["content"][0]["text"]
        assert "no notebook" in text.lower()

    @pytest.mark.asyncio
    async def test_no_args_returns_hint(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {})
        text = result["content"][0]["text"]
        assert "please specify" in text.lower()
        # Hint should include available versions to help the model
        assert "v01" in text

    @pytest.mark.asyncio
    async def test_summary_returns_counts(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"summary": True})
        text = result["content"][0]["text"]
        assert "Total: 5" in text
        assert "scientist: 2" in text
        assert "revision: 1" in text
        assert "stop_gate: 1" in text
        assert "ingestor: 1" in text
        # No TOC, no full detail
        assert "NOTEBOOK TOC" not in text

    @pytest.mark.asyncio
    async def test_versions_single(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"versions": ["v00"]})
        text = result["content"][0]["text"]
        assert "[v00 scientist] Initial exploration" in text
        assert "Checked composition ranges" in text
        assert "[v01" not in text

    @pytest.mark.asyncio
    async def test_versions_multi_matches_all_sources(self, sample_entries):
        """A single version can match multiple entries (scientist + revision)."""
        result = await _handle_read_notebook(sample_entries, {"versions": ["v01"]})
        text = result["content"][0]["text"]
        assert "Hardness correlation" in text
        assert "Revised after debate" in text

    @pytest.mark.asyncio
    async def test_versions_unknown(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"versions": ["v99"]})
        text = result["content"][0]["text"]
        assert "not found" in text.lower()
        assert "Available versions" in text

    @pytest.mark.asyncio
    async def test_versions_partial_miss_shows_note(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"versions": ["v00", "v99"]})
        text = result["content"][0]["text"]
        assert "not found: v99" in text.lower()
        assert "Initial exploration" in text

    @pytest.mark.asyncio
    async def test_source_scientist(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"source": "scientist"})
        text = result["content"][0]["text"]
        assert "Initial exploration" in text
        assert "Hardness correlation" in text
        assert "Revised after debate" not in text  # revision, not scientist

    @pytest.mark.asyncio
    async def test_source_revision(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"source": "revision"})
        text = result["content"][0]["text"]
        assert "Revised after debate" in text
        assert "Initial exploration" not in text

    @pytest.mark.asyncio
    async def test_source_stop_revision(self, tmp_path: Path):
        """Regression: stop_revision is a real source value the orchestrator writes."""
        nb = tmp_path / NOTEBOOK_FILENAME
        append_entry(nb, "v01 plan\n\nbody", version="v01", source="scientist")
        append_entry(
            nb,
            "Stop withdrawn after debate\n\nrationale",
            version="v02",
            source="stop_revision",
        )
        entries = parse_notebook_entries(nb)

        result = await _handle_read_notebook(entries, {"source": "stop_revision"})
        text = result["content"][0]["text"]
        assert "Stop withdrawn after debate" in text
        assert "v01 plan" not in text

        summary = await _handle_read_notebook(entries, {"summary": True})
        assert "stop_revision: 1" in summary["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_search_hit(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"search": "Falsification"})
        text = result["content"][0]["text"]
        assert "Revised after debate" in text

    @pytest.mark.asyncio
    async def test_search_case_insensitive(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"search": "cr fraction"})
        text = result["content"][0]["text"]
        assert "Hardness correlation" in text

    @pytest.mark.asyncio
    async def test_search_miss(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"search": "zzz_nothing"})
        text = result["content"][0]["text"]
        assert "no entries match" in text.lower()

    @pytest.mark.asyncio
    async def test_last_n(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"last_n": 2})
        text = result["content"][0]["text"]
        assert "Revised after debate" in text
        assert "Stop withdrawn" in text
        assert "Initial exploration" not in text

    @pytest.mark.asyncio
    async def test_last_n_zero_errors(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"last_n": 0})
        text = result["content"][0]["text"]
        assert "positive integer" in text.lower()

    @pytest.mark.asyncio
    async def test_full_detail_includes_content(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"versions": ["v00"]})
        text = result["content"][0]["text"]
        # Full detail: version + source header and body
        assert "[v00 scientist]" in text
        assert "Checked composition ranges" in text


# ---------------------------------------------------------------------------
# _normalize_args
# ---------------------------------------------------------------------------


class TestNormalizeArgs:
    def test_versions_json_string_parsed(self):
        args = _normalize_args({"versions": '["v01", "v02"]'})
        assert args == {"versions": ["v01", "v02"]}

    def test_versions_bare_string_wrapped(self):
        args = _normalize_args({"versions": "v01"})
        assert args == {"versions": ["v01"]}

    def test_versions_array_unchanged(self):
        args = _normalize_args({"versions": ["v01", "v02"]})
        assert args == {"versions": ["v01", "v02"]}

    def test_versions_mixed_types_coerced(self):
        args = _normalize_args({"versions": [1, "v02"]})
        assert args == {"versions": ["1", "v02"]}

    def test_last_n_string_coerced_to_int(self):
        args = _normalize_args({"last_n": "3"})
        assert args == {"last_n": 3}

    def test_last_n_int_unchanged(self):
        args = _normalize_args({"last_n": 5})
        assert args == {"last_n": 5}

    def test_last_n_non_numeric_left_alone(self):
        args = _normalize_args({"last_n": "abc"})
        assert args == {"last_n": "abc"}

    def test_empty_args(self):
        assert _normalize_args({}) == {}


class TestHandlerToleratesMalformedTypes:
    @pytest.mark.asyncio
    async def test_versions_json_string(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"versions": '["v00", "v01"]'})
        text = result["content"][0]["text"]
        assert "Initial exploration" in text
        assert "Hardness correlation" in text

    @pytest.mark.asyncio
    async def test_last_n_string(self, sample_entries):
        result = await _handle_read_notebook(sample_entries, {"last_n": "2"})
        text = result["content"][0]["text"]
        assert "Revised after debate" in text
        assert "Stop withdrawn" in text


# ---------------------------------------------------------------------------
# build_notebook_mcp_server
# ---------------------------------------------------------------------------


class TestBuildNotebookMcpServer:
    def test_creates_stdio_server(self, sample_notebook: Path):
        server = build_notebook_mcp_server(sample_notebook)
        assert server["type"] == "stdio"
        assert server["command"] == sys.executable
        assert "_notebook_mcp_server.py" in server["args"][0]
        entries_path = server["args"][1]
        data = json.loads(Path(entries_path).read_text())
        assert len(data) == 5
        assert data[0]["source"] == "ingestor"
        assert data[1]["version"] == "v00"

    def test_writes_to_output_dir(self, sample_notebook: Path, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        server = build_notebook_mcp_server(sample_notebook, output_dir=out_dir)
        entries_path = server["args"][1]
        # Scratch JSON lives under a hidden .mcp/ subdir so the run dir
        # stays uncluttered with reviewer-facing artifacts.
        assert entries_path == str(out_dir / ".mcp" / "notebook_entries.json")
        data = json.loads(Path(entries_path).read_text())
        assert len(data) == 5

    def test_missing_notebook_produces_empty_server(self, tmp_path: Path):
        server = build_notebook_mcp_server(tmp_path / "missing.xml")
        entries_path = server["args"][1]
        assert json.loads(Path(entries_path).read_text()) == []
