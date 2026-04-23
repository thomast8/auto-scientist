"""Database service wiring.

Reads all three sibling keys under `config["db"]`. Any env override
that clobbers siblings (instead of updating one in place) will raise
KeyError here.
"""

from __future__ import annotations

from typing import Any


def db_connection_args(config: dict[str, Any]) -> dict[str, Any]:
    """Extract database connection args for `psycopg.connect(**args)`.

    Relies on all of `host`, `port`, and `name` surviving any env-var
    overrides applied by the config loader.
    """
    db = config["db"]
    return {
        "host": db["host"],
        "port": db["port"],
        "dbname": db["name"],
    }
