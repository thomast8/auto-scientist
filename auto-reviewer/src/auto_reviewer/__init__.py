"""Auto-Reviewer: autonomous bug-hunting PR reviewer on the auto_core runtime."""

from auto_reviewer._roles import install_reviewer_registry

__version__ = "0.1.0"

# Populate the core runtime's AGENT_STYLES / AGENT_DESCRIPTIONS / PHASE_STYLES
# / SUMMARY_PROMPTS / artifact + buffer + notebook-source tables from the
# reviewer registry. Auto-Scientist performs the same step with its own
# registry. Running both in the same process is not supported - each CLI
# entrypoint owns its own process.
install_reviewer_registry()
