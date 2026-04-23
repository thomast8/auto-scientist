"""Auto-Scientist: Autonomous scientific investigation framework."""

from auto_scientist._roles import install_scientist_registry

__version__ = "0.1.0"

# Populate the core runtime's AGENT_STYLES / AGENT_DESCRIPTIONS / PHASE_STYLES
# / SUMMARY_PROMPTS / artifact + buffer + notebook-source tables from the
# scientist registry. Auto-Reviewer performs the same step at import time
# with its own registry. Running both in the same process is not supported -
# each CLI entrypoint owns its own process.
install_scientist_registry()
