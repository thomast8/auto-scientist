"""Operational run configuration shared by every app on auto_core.

`RunConfig` captures the minimum an implementer / orchestrator needs to run a
workload: a name + description for display, the subprocess command used to
execute an implementer script, a working directory, a timeout, a version
prefix, and paths the implementer must not touch.

App-specific configs (auto_scientist's `DomainConfig`, auto_reviewer's
`ReviewConfig`) subclass this and layer on their own fields (e.g. scientific
`data_paths`, PR `repo_path` + `pr_ref`). Core utilities (validation, the
orchestrator's timeout derivation) accept `RunConfig` and ignore subclass
extensions.
"""

from collections.abc import Mapping
from typing import Any

from pydantic import AliasChoices, BaseModel, Field, TypeAdapter, ValidationError, field_validator

GENERATED_NETWORK_ACCESS_FIELDS = (
    "implementer_sandbox_network_access",
    "implementer_network_access",
)
_BOOL_ADAPTER = TypeAdapter(bool)


def _generated_value_enables_network_access(value: Any) -> bool:
    """Return True if generated config value would enable network access."""
    try:
        return _BOOL_ADAPTER.validate_python(value)
    except ValidationError:
        # Invalid operator-only values should not reach generated config output.
        return True


def generated_config_requests_sandbox_network_access(raw_config: Mapping[str, Any]) -> bool:
    """Return True when an LLM-generated config tries to opt into network access."""
    return any(
        field in raw_config and _generated_value_enables_network_access(raw_config[field])
        for field in GENERATED_NETWORK_ACCESS_FIELDS
    )


def reject_generated_sandbox_network_access(raw_config: Mapping[str, Any]) -> None:
    """Reject LLM-generated network access requests.

    The sandbox network flag is trusted operator input only. Ingestor and
    Intake agents write generated configs from untrusted data and prompts, so
    they must never be allowed to enable it.
    """
    if generated_config_requests_sandbox_network_access(raw_config):
        raise ValueError(
            "implementer_sandbox_network_access is operator-only for generated "
            "configs and must be false. A trusted operator can enable it by "
            "editing config outside the Ingestor/Intake generation path."
        )


def clear_generated_sandbox_network_access(raw_config: dict[str, Any]) -> bool:
    """Force generated config network access off as a defense-in-depth fallback."""
    if not generated_config_requests_sandbox_network_access(raw_config):
        return False
    raw_config["implementer_sandbox_network_access"] = False
    raw_config.pop("implementer_network_access", None)
    return True


class RunConfig(BaseModel):
    """Operational configuration common to all apps on the auto_core runtime."""

    name: str
    description: str = ""
    run_command: str = "uv run {script_path}"
    run_cwd: str = "."
    run_timeout_minutes: int = 120
    version_prefix: str = "v"
    protected_paths: list[str] = Field(default_factory=list)
    implementer_sandbox_network_access: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "implementer_sandbox_network_access",
            "implementer_network_access",
        ),
        description=(
            "Allow the OpenAI SDK implementer sandbox to use network access "
            "for dependency installation. Defaults off for least privilege."
        ),
    )

    @field_validator("run_command")
    @classmethod
    def run_command_must_contain_placeholder(cls, v: str) -> str:
        if "{script_path}" not in v:
            raise ValueError(f"run_command must contain '{{script_path}}' placeholder, got: {v}")
        return v
