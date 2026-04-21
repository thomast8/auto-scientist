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

from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    """Operational configuration common to all apps on the auto_core runtime."""

    name: str
    description: str = ""
    run_command: str = "uv run {script_path}"
    run_cwd: str = "."
    run_timeout_minutes: int = 120
    version_prefix: str = "v"
    protected_paths: list[str] = Field(default_factory=list)

    @field_validator("run_command")
    @classmethod
    def run_command_must_contain_placeholder(cls, v: str) -> str:
        if "{script_path}" not in v:
            raise ValueError(f"run_command must contain '{{script_path}}' placeholder, got: {v}")
        return v
