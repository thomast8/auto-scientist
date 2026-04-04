"""Auto-sync PEP 723 inline metadata with actual script imports.

Parses a Python script's imports, compares against the declared PEP 723
dependencies, and patches the metadata block to add any missing entries.

Runnable standalone: ``python -m auto_scientist.ensure_deps <script_path>``
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from pathlib import Path

# PyPI package name -> Python import name, for packages where they differ.
_PYPI_TO_IMPORT: dict[str, str] = {
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "python-dateutil": "dateutil",
    "attrs": "attr",
}

# Reverse: import name -> preferred PyPI package name.
_IMPORT_TO_PYPI: dict[str, str] = {}
for _pypi, _imp in _PYPI_TO_IMPORT.items():
    # First entry wins (e.g., opencv-python before opencv-python-headless).
    _IMPORT_TO_PYPI.setdefault(_imp, _pypi)

# Regex for the PEP 723 inline metadata block.
_PEP723_RE = re.compile(
    r"^# /// script\s*\n((?:#[^\n]*\n)*?)# ///",
    re.MULTILINE,
)


def extract_imports(source: str) -> set[str]:
    """Return all third-party import names from *source*.

    Walks the full AST (including try/except, conditionals, function bodies)
    and filters out stdlib modules using ``sys.stdlib_module_names``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    return imported - sys.stdlib_module_names


def parse_pep723_deps(source: str) -> set[str]:
    """Return the set of Python *import names* covered by PEP 723 deps."""
    m = _PEP723_RE.search(source)
    if not m:
        return set()

    raw_lines = m.group(1).splitlines()
    content = "\n".join(line.lstrip("#").strip() for line in raw_lines)
    try:
        metadata = tomllib.loads(content)
    except Exception:
        return set()

    import_names: set[str] = set()
    for dep in metadata.get("dependencies", []):
        # Strip version specifiers and extras: "numpy>=1.20" -> "numpy"
        normalized = re.split(r"[>=<!~\[]", dep)[0].strip().lower()
        if normalized in _PYPI_TO_IMPORT:
            import_names.add(_PYPI_TO_IMPORT[normalized])
        else:
            # Most packages: PyPI name == import name (with hyphens -> underscores).
            import_names.add(normalized.replace("-", "_"))
    return import_names


def find_missing_deps(source: str) -> list[str]:
    """Return PyPI package names for imports not covered by PEP 723 deps."""
    third_party = extract_imports(source)
    if not third_party:
        return []

    declared = parse_pep723_deps(source)
    undeclared = sorted(third_party - declared)

    result: list[str] = []
    for imp in undeclared:
        if imp in _IMPORT_TO_PYPI:
            result.append(_IMPORT_TO_PYPI[imp])
        else:
            result.append(imp)
    return result


def patch_pep723_block(source: str, missing_pypi: list[str]) -> str:
    """Add *missing_pypi* packages to the PEP 723 dependencies block.

    If the script has no metadata block, one is prepended.
    """
    if not missing_pypi:
        return source

    m = _PEP723_RE.search(source)
    if m:
        # Parse existing deps to avoid duplicates.
        raw_lines = m.group(1).splitlines()
        content = "\n".join(line.lstrip("#").strip() for line in raw_lines)
        try:
            metadata = tomllib.loads(content)
        except Exception:
            metadata = {}

        existing = [dep.strip().strip('"').strip("'") for dep in metadata.get("dependencies", [])]
        new_deps = sorted(set(existing) | set(missing_pypi))

        deps_str = ",\n".join(f'#     "{d}"' for d in new_deps)
        requires = metadata.get("requires-python", ">=3.11")

        new_block = (
            f"# /// script\n"
            f'# requires-python = "{requires}"\n'
            f"# dependencies = [\n"
            f"{deps_str},\n"
            f"# ]\n"
            f"# ///"
        )
        return source[: m.start()] + new_block + source[m.end() :]

    # No existing block: prepend one.
    deps_str = ",\n".join(f'#     "{d}"' for d in sorted(missing_pypi))
    new_block = (
        f"# /// script\n"
        f'# requires-python = ">=3.11"\n'
        f"# dependencies = [\n"
        f"{deps_str},\n"
        f"# ]\n"
        f"# ///\n\n"
    )
    return new_block + source


def ensure_deps(script_path: Path) -> list[str]:
    """Patch *script_path* in-place so PEP 723 deps cover all imports.

    Returns the list of PyPI package names that were added.
    """
    source = script_path.read_text(encoding="utf-8")
    missing = find_missing_deps(source)
    if missing:
        patched = patch_pep723_block(source, missing)
        script_path.write_text(patched, encoding="utf-8")
    return missing


def validate_deps(script_path: Path) -> tuple[bool, str]:
    """Check that every third-party import is covered by PEP 723 deps.

    Returns ``(True, "")`` if all imports are covered, or
    ``(False, message)`` with an actionable error listing undeclared imports.
    """
    source = script_path.read_text(encoding="utf-8")
    missing = find_missing_deps(source)
    if not missing:
        return True, ""

    lines = []
    for pkg in missing:
        imp = _PYPI_TO_IMPORT.get(pkg)
        if imp:
            lines.append(f'  - "{pkg}" (provides import "{imp}")')
        else:
            lines.append(f'  - "{pkg}"')
    msg = (
        "The script imports third-party modules not declared in the "
        "PEP 723 dependencies block:\n"
        + "\n".join(lines)
        + "\n\nAdd these to the # /// script dependencies list."
    )
    return False, msg


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m auto_scientist.ensure_deps <script_path>", file=sys.stderr)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"ensure_deps: {path} not found", file=sys.stderr)
        sys.exit(1)

    added = ensure_deps(path)
    if added:
        print(f"ensure_deps: added {', '.join(added)} to PEP 723 deps in {path.name}")
