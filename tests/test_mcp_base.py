"""Tests for the shared MCP tool infrastructure."""

import json
import sys
from pathlib import Path

import pytest

from auto_scientist.agents._mcp_base import (
    _MCP_TOOL_REGISTRY,
    MCPToolSpec,
    build_mcp_server_config,
    get_deferred_descriptions,
    register_mcp_tool,
)


@pytest.fixture
def sample_spec():
    return MCPToolSpec(
        server_name="test_server",
        tool_name="read_test",
        description="A test tool.",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        deferred_description="mcp__test_server__read_test(q?) - A test tool.",
    )


@pytest.fixture
def sample_data():
    return [{"id": "1", "value": "hello"}, {"id": "2", "value": "world"}]


class TestMCPToolSpec:
    def test_mcp_tool_name(self, sample_spec):
        assert sample_spec.mcp_tool_name == "mcp__test_server__read_test"

    def test_frozen(self, sample_spec):
        with pytest.raises(AttributeError):
            sample_spec.server_name = "modified"


class TestRegistry:
    def test_register_and_retrieve(self, sample_spec):
        # Save and restore registry state to avoid test pollution
        saved = dict(_MCP_TOOL_REGISTRY)
        try:
            register_mcp_tool(sample_spec)
            descriptions = get_deferred_descriptions()
            assert "mcp__test_server__read_test" in descriptions
            assert descriptions["mcp__test_server__read_test"] == sample_spec.deferred_description
        finally:
            _MCP_TOOL_REGISTRY.clear()
            _MCP_TOOL_REGISTRY.update(saved)

    def test_prediction_spec_auto_registered(self):
        """The prediction tool spec should be registered at import time."""
        # Importing prediction_tool triggers register_mcp_tool
        import auto_scientist.agents.prediction_tool  # noqa: F401

        descriptions = get_deferred_descriptions()
        assert "mcp__predictions__read_predictions" in descriptions


class TestBuildMcpServerConfig:
    def test_returns_stdio_config(self, sample_data, tmp_path):
        script = tmp_path / "server.py"
        script.write_text("# placeholder")

        config = build_mcp_server_config(
            sample_data, script, output_dir=tmp_path, filename="test.json"
        )

        assert config["type"] == "stdio"
        assert config["command"] == sys.executable
        assert str(script.resolve()) in config["args"][0]
        # Scratch JSON lives under a hidden .mcp/ subdir.
        assert str(tmp_path / ".mcp" / "test.json") in config["args"][1]

    def test_writes_data_to_output_dir(self, sample_data, tmp_path):
        script = tmp_path / "server.py"
        script.write_text("# placeholder")

        build_mcp_server_config(sample_data, script, output_dir=tmp_path, filename="test.json")

        data_path = tmp_path / ".mcp" / "test.json"
        assert data_path.exists()
        loaded = json.loads(data_path.read_text())
        assert loaded == sample_data

    def test_uses_temp_file_when_no_output_dir(self, sample_data, tmp_path):
        script = tmp_path / "server.py"
        script.write_text("# placeholder")

        config = build_mcp_server_config(sample_data, script, filename="test.json")

        data_path = Path(config["args"][1])
        assert data_path.exists()
        loaded = json.loads(data_path.read_text())
        assert loaded == sample_data
        # Cleanup
        data_path.unlink()
