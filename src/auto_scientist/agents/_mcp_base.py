"""Shared infrastructure for MCP tool servers.

Provides reusable boilerplate so that adding a new agentic data-access tool
requires only: (1) a query handler function, (2) an MCPToolSpec, and (3) two
lines of wiring in the consuming agent.

The module has three parts:

* **MCPToolSpec** - dataclass bundling tool metadata (name, schema, description).
* **build_mcp_server_config** - serializes data to a JSON file and returns a
  stdio server config dict that both Claude and Codex backends understand.
* **run_mcp_server_main** - generic ``if __name__ == "__main__"`` entry point
  for MCP server scripts, handling Server creation, tool registration, and
  stdio transport so each server only needs to supply a query handler.
* **Tool registry** - ``register_mcp_tool`` / ``get_deferred_descriptions``
  so sdk_utils can auto-populate tool descriptions without manual edits.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPToolSpec:
    """Metadata for a single MCP tool.

    Bundles everything needed to register, wire, and describe an MCP tool
    so that adding a new tool only requires defining one of these.
    """

    server_name: str
    """Key in the mcp_servers dict, e.g. ``"predictions"``."""

    tool_name: str
    """Tool name inside the MCP server, e.g. ``"read_predictions"``."""

    description: str
    """Full description shown to the LLM via the MCP protocol."""

    input_schema: dict[str, Any]
    """JSON Schema for tool arguments."""

    deferred_description: str
    """One-line description for prompt injection (sdk_utils)."""

    @property
    def mcp_tool_name(self) -> str:
        """Full tool name as seen by the Claude/Codex SDK.

        Follows the ``mcp__<server>__<tool>`` convention.
        """
        return f"mcp__{self.server_name}__{self.tool_name}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

_MCP_TOOL_REGISTRY: dict[str, MCPToolSpec] = {}


def register_mcp_tool(spec: MCPToolSpec) -> None:
    """Register a tool spec so sdk_utils can auto-discover descriptions."""
    _MCP_TOOL_REGISTRY[spec.server_name] = spec


def get_deferred_descriptions() -> dict[str, str]:
    """Return ``{mcp_tool_name: deferred_description}`` for all registered tools."""
    return {spec.mcp_tool_name: spec.deferred_description for spec in _MCP_TOOL_REGISTRY.values()}


# ---------------------------------------------------------------------------
# Server config builder
# ---------------------------------------------------------------------------


def build_mcp_server_config(
    data_dicts: list[dict[str, Any]],
    server_script: str | Path,
    *,
    output_dir: Path | None = None,
    filename: str = "data.json",
) -> dict[str, Any]:
    """Serialize data and return an stdio MCP server config.

    When *output_dir* is provided the JSON file is written there (stable,
    self-contained experiment folder). Otherwise a temporary file is used
    (for tests).

    Returns a dict suitable for ``SDKOptions.mcp_servers`` values.
    """
    if output_dir is not None:
        data_path = output_dir / filename
        data_path.write_text(json.dumps(data_dicts))
    else:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{Path(filename).stem}_",
            delete=False,
        ) as tmp:
            json.dump(data_dicts, tmp)
        data_path = Path(tmp.name)

    logger.debug(f"Wrote {len(data_dicts)} records to {data_path}")

    return {
        "type": "stdio",
        "command": sys.executable,
        "args": [str(Path(server_script).resolve()), str(data_path)],
    }


# ---------------------------------------------------------------------------
# Generic MCP server entry point (for subprocess scripts)
# ---------------------------------------------------------------------------


MCPQueryHandler = Callable[[list[dict[str, Any]], dict[str, Any]], str]
"""Callable that handles an MCP tool invocation.

Takes the loaded data (list of dicts) and the tool arguments,
returns a text response string.
"""


def run_mcp_server_main(
    server_name: str,
    spec: MCPToolSpec,
    handler: MCPQueryHandler,
) -> None:
    """Generic entry point for MCP server scripts.

    Reads a JSON data file from ``sys.argv[1]``, creates an MCP Server,
    registers one tool per *spec*, and runs the stdio transport loop.

    Each server script's ``if __name__ == "__main__"`` block becomes a
    single call to this function.
    """
    asyncio.run(_run_server(server_name, spec, handler))


async def _run_server(
    server_name: str,
    spec: MCPToolSpec,
    handler: MCPQueryHandler,
) -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <data.json>", file=sys.stderr)
        sys.exit(1)

    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    with open(sys.argv[1]) as f:
        data: list[dict[str, Any]] = json.load(f)

    server = Server(server_name)

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=spec.tool_name,
                description=spec.description,
                inputSchema=spec.input_schema,
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        if name != spec.tool_name:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        result = handler(data, arguments or {})
        return [TextContent(type="text", text=result)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
