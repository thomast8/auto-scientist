"""Discovery agent: Phase 1 data exploration and first model generation.

Uses ClaudeSDKClient for persistent session (exploratory, may need multiple queries).
Tools: Bash (data exploration, stats, plots), WebSearch (literature), Read/Write.
Produces: domain config (success criteria, metric definitions), first experiment script,
          lab notebook entry #0.
"""

from pathlib import Path

from auto_scientist.config import DomainConfig
from auto_scientist.state import ExperimentState


async def run_discovery(
    state: ExperimentState,
    data_path: Path,
    output_dir: Path,
    interactive: bool = False,
) -> tuple[DomainConfig, Path]:
    """Explore data, research domain, and produce the first experiment script.

    Args:
        state: Current experiment state.
        data_path: Path to the dataset.
        output_dir: Directory for experiment outputs.
        interactive: If True, use AskUserQuestion to clarify with the user.

    Returns:
        Tuple of (domain config, path to first script).
    """
    # TODO: Implement with claude-code-sdk
    # 1. Explore dataset (shape, columns, distributions, correlations)
    # 2. Search literature for domain knowledge
    # 3. Design first model based on data characteristics
    # 4. Write v1 script
    # 5. Create lab notebook entry #0
    # 6. Return DomainConfig + script path
    raise NotImplementedError("Discovery agent not yet implemented")
