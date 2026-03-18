"""Discovery agent: Phase 1 data exploration and domain configuration.

Uses ClaudeSDKClient for persistent session (exploratory, may need multiple queries).
Tools: Bash (data exploration, stats, plots), Read/Write, Glob, Grep.
Produces: domain config (success criteria, metric definitions), lab notebook entry #0.
"""

import json
from pathlib import Path

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)

from auto_scientist.config import DomainConfig
from auto_scientist.prompts.discovery import DISCOVERY_SYSTEM, DISCOVERY_USER
from auto_scientist.state import ExperimentState


async def run_discovery(
    state: ExperimentState,
    data_path: Path,
    output_dir: Path,
    interactive: bool = False,
    model: str | None = None,
) -> DomainConfig:
    """Explore data and produce a domain configuration.

    Uses a multi-turn session to explore the dataset, understand its structure,
    and create a domain config with success criteria. Does NOT write experiment
    scripts (that's the Coder's job).

    Args:
        state: Current experiment state.
        data_path: Path to the dataset.
        output_dir: Directory for experiment outputs.
        interactive: If True, allows the agent to ask the user questions.
        model: Optional model override.

    Returns:
        Domain configuration discovered from data exploration.
    """
    notebook_path = output_dir / "lab_notebook.md"
    config_path = output_dir / "domain_config.json"

    tools = ["Bash", "Read", "Write", "Glob", "Grep"]
    if interactive:
        tools.append("AskUserQuestion")

    options = ClaudeCodeOptions(
        system_prompt=DISCOVERY_SYSTEM,
        allowed_tools=tools,
        max_turns=30,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
    )

    prompt = DISCOVERY_USER.format(
        data_path=str(data_path.resolve()),
        goal=state.goal,
        domain_knowledge="(No pre-existing domain knowledge - you are discovering this.)",
        output_dir=str(output_dir),
        notebook_path=str(notebook_path),
        config_path=str(config_path),
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        print(f"  [discovery] {block.text[:200]}")
            elif isinstance(msg, ResultMessage):
                pass

    # Load and validate the domain config
    if not config_path.exists():
        raise FileNotFoundError(
            f"Discovery agent did not create domain config at {config_path}"
        )
    config_data = json.loads(config_path.read_text())
    config = DomainConfig.model_validate(config_data)

    return config
