"""Discovery agent: Phase 1 data exploration and first model generation.

Uses ClaudeSDKClient for persistent session (exploratory, may need multiple queries).
Tools: Bash (data exploration, stats, plots), Read/Write, Glob, Grep.
Produces: domain config (success criteria, metric definitions), first experiment script,
          lab notebook entry #0.
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
) -> tuple[DomainConfig, Path]:
    """Explore data, research domain, and produce the first experiment script.

    Uses a multi-turn session: explores the dataset then designs and writes
    the first experiment script based on findings.

    Args:
        state: Current experiment state.
        data_path: Path to the dataset.
        output_dir: Directory for experiment outputs.
        interactive: If True, allows the agent to ask the user questions.

    Returns:
        Tuple of (domain config, path to first script).
    """
    version_dir = output_dir / "v00"
    version_dir.mkdir(parents=True, exist_ok=True)
    script_path = version_dir / "experiment.py"
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
    )

    prompt = DISCOVERY_USER.format(
        data_path=str(data_path.resolve()),
        goal=state.goal,
        domain_knowledge="(No pre-existing domain knowledge - you are discovering this.)",
        output_dir=str(output_dir),
        version_dir="v00",
        script_name="experiment.py",
        notebook_path=str(notebook_path),
        config_path=str(config_path),
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        # Print first 200 chars of each text block for progress
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

    # Verify script was created
    if not script_path.exists():
        raise FileNotFoundError(
            f"Discovery agent did not create experiment script at {script_path}"
        )

    return config, script_path
