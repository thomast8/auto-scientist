"""Domain configuration schema for auto-scientist.

Extends `auto_core.config.RunConfig` with scientific `data_paths`. Other
operational fields (run_command, run_cwd, run_timeout_minutes, version_prefix,
protected_paths) live in the base class.
"""

import logging

from auto_core.config import RunConfig
from pydantic import field_validator

logger = logging.getLogger(__name__)


class DomainConfig(RunConfig):
    """Operational configuration for a specific scientific domain.

    Scientific concerns (`domain_knowledge`, `prediction_history`) live in
    `ExperimentState`; this carries only runtime / infrastructure settings.
    """

    data_paths: list[str]

    @field_validator("data_paths", mode="before")
    @classmethod
    def coerce_data_paths(cls, v: list[str] | dict[str, str]) -> list[str]:
        if isinstance(v, dict):
            logger.warning(
                "Deprecated: data_paths received as dict, coercing to list. "
                "Ensure the ingestor writes data_paths as a JSON list."
            )
            return list(v.values())
        return v
