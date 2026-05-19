"""Prompt default-flavor regression tests for auto-reviewer."""

from auto_reviewer.prompts.adversary import build_adversary_system
from auto_reviewer.prompts.findings import FINDINGS_SYSTEM, build_findings_system
from auto_reviewer.prompts.hunter import HUNTER_SYSTEM, build_hunter_system
from auto_reviewer.prompts.intake import INTAKE_SYSTEM, build_intake_system
from auto_reviewer.prompts.prober import PROBER_SYSTEM, build_prober_system
from auto_reviewer.prompts.stop_gate import (
    ASSESSOR_SYSTEM,
    STOP_DEBATE_SYSTEM,
    STOP_REVISION_SYSTEM,
    build_assessment_system,
    build_stop_critic_system,
    build_stop_revision_system,
)
from auto_reviewer.prompts.surveyor import SURVEYOR_SYSTEM, build_surveyor_system


def test_auto_reviewer_prompt_builders_default_to_gpt() -> None:
    assert build_adversary_system() == build_adversary_system("gpt")
    assert build_findings_system() == build_findings_system("gpt") == FINDINGS_SYSTEM
    assert build_hunter_system() == build_hunter_system("gpt") == HUNTER_SYSTEM
    assert build_intake_system() == build_intake_system("gpt") == INTAKE_SYSTEM
    assert build_prober_system() == build_prober_system("gpt") == PROBER_SYSTEM
    assert build_surveyor_system() == build_surveyor_system("gpt") == SURVEYOR_SYSTEM
    assert build_assessment_system() == build_assessment_system("gpt") == ASSESSOR_SYSTEM
    assert build_stop_critic_system() == build_stop_critic_system("gpt") == STOP_DEBATE_SYSTEM
    assert build_stop_revision_system() == build_stop_revision_system("gpt") == STOP_REVISION_SYSTEM


def test_auto_reviewer_claude_prompt_variant_still_exists() -> None:
    assert build_intake_system("claude") != build_intake_system("gpt")
    assert build_adversary_system("claude") != build_adversary_system("gpt")
