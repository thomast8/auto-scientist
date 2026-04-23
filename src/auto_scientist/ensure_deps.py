"""Compatibility re-export: the real module lives in :mod:`auto_core.ensure_deps`.

Kept so that existing test fixtures and any external callers that still
invoke ``python -m auto_scientist.ensure_deps`` continue to work. New code
should import from :mod:`auto_core.ensure_deps` directly.
"""

from __future__ import annotations

from auto_core.ensure_deps import (
    _IMPORT_TO_PYPI,
    _PEP723_RE,
    _PYPI_TO_IMPORT,
    _SCIENTIFIC_PACKAGES,
    _ensure_pip,
    _preinstall_scientific_packages,
    ensure_deps,
    extract_imports,
    extract_pep723_dep_strings,
    find_missing_deps,
    install_deps,
    parse_pep723_deps,
    patch_pep723_block,
    validate_deps,
    verify_imports,
)

__all__ = [
    "_IMPORT_TO_PYPI",
    "_PEP723_RE",
    "_PYPI_TO_IMPORT",
    "_SCIENTIFIC_PACKAGES",
    "_ensure_pip",
    "_preinstall_scientific_packages",
    "ensure_deps",
    "extract_imports",
    "extract_pep723_dep_strings",
    "find_missing_deps",
    "install_deps",
    "parse_pep723_deps",
    "patch_pep723_block",
    "validate_deps",
    "verify_imports",
]


if __name__ == "__main__":
    import runpy
    import sys

    sys.argv[0] = "auto_core.ensure_deps"
    runpy.run_module("auto_core.ensure_deps", run_name="__main__", alter_sys=True)
