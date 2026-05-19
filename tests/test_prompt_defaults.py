"""Prompt default-flavor regression tests."""

from auto_scientist.prompts.analyst import ANALYST_SYSTEM, build_analyst_system
from auto_scientist.prompts.coder import CODER_SYSTEM, build_coder_system
from auto_scientist.prompts.critic import CRITIC_SYSTEM_BASE, build_critic_system
from auto_scientist.prompts.ingestor import INGESTOR_SYSTEM, build_ingestor_system
from auto_scientist.prompts.report import REPORT_SYSTEM, build_report_system
from auto_scientist.prompts.scientist import SCIENTIST_SYSTEM, build_scientist_system
from auto_scientist.prompts.stop_gate import (
    ASSESSMENT_SYSTEM,
    STOP_CRITIC_SYSTEM_BASE,
    STOP_REVISION_SYSTEM,
    build_assessment_system,
    build_stop_critic_system,
    build_stop_revision_system,
)


def test_auto_scientist_prompt_builders_default_to_gpt() -> None:
    assert build_analyst_system() == build_analyst_system("gpt") == ANALYST_SYSTEM
    assert build_coder_system() == build_coder_system("gpt") == CODER_SYSTEM
    assert build_critic_system() == build_critic_system("gpt") == CRITIC_SYSTEM_BASE
    assert build_ingestor_system() == build_ingestor_system("gpt") == INGESTOR_SYSTEM
    assert build_report_system() == build_report_system("gpt") == REPORT_SYSTEM
    assert build_scientist_system() == build_scientist_system("gpt") == SCIENTIST_SYSTEM
    assert build_assessment_system() == build_assessment_system("gpt") == ASSESSMENT_SYSTEM
    assert build_stop_critic_system() == build_stop_critic_system("gpt") == STOP_CRITIC_SYSTEM_BASE
    assert build_stop_revision_system() == build_stop_revision_system("gpt") == STOP_REVISION_SYSTEM


def test_auto_scientist_claude_prompt_variant_still_exists() -> None:
    assert build_analyst_system("claude") != build_analyst_system("gpt")
    assert build_scientist_system("claude") != build_scientist_system("gpt")
