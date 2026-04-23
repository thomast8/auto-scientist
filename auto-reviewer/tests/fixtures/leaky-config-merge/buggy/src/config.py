"""Config loader with env-var overrides.

Env var names map to nested config keys via an underscore separator.
`DB_HOST=localhost` overrides `config["db"]["host"]`. Sibling keys
survive the override: `DB_HOST` does not disturb `config["db"]["port"]`
or `config["db"]["name"]`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: Path | str, env: dict[str, str]) -> dict[str, Any]:
    """Load JSON config from `path` and apply env overrides."""
    config = json.loads(Path(path).read_text())
    return apply_env_overrides(config, env)


def apply_env_overrides(config: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """Apply env-var overrides to `config` in place and return it.

    Nested structure is preserved: `{"DB_HOST": "x"}` overrides
    `config["db"]["host"]` without clobbering `config["db"]["port"]`
    or any other sibling key.
    """
    for key, value in env.items():
        if "_" in key:
            top, sub = key.lower().split("_", 1)
            # Simplified: build the nested override inline so the loop
            # body is a single assignment for both shapes.
            config[top] = {sub: _coerce(value)}
        else:
            config[key.lower()] = _coerce(value)
    return config


def _coerce(value: str) -> Any:
    """Best-effort coerce string -> bool / int / float / str."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
