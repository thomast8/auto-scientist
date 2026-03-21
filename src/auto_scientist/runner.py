"""Run result model for experiment execution."""

from pydantic import BaseModel


class RunResult(BaseModel):
    """Result of running an experiment script."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    output_files: list[str] = []
