"""Tests for the PEP 723 dependency auto-sync module."""

from auto_scientist.ensure_deps import (
    ensure_deps,
    extract_imports,
    find_missing_deps,
    parse_pep723_deps,
    patch_pep723_block,
    validate_deps,
)

# ---------------------------------------------------------------------------
# extract_imports
# ---------------------------------------------------------------------------


class TestExtractImports:
    def test_stdlib_only(self):
        source = "import os\nimport sys\nfrom pathlib import Path\n"
        assert extract_imports(source) == set()

    def test_third_party(self):
        source = "import numpy\nimport pandas\n"
        assert extract_imports(source) == {"numpy", "pandas"}

    def test_subpackage(self):
        source = "from scipy.stats import norm\n"
        assert extract_imports(source) == {"scipy"}

    def test_mixed(self):
        source = "import os\nimport numpy\nfrom collections import Counter\nimport seaborn\n"
        assert extract_imports(source) == {"numpy", "seaborn"}

    def test_syntax_error_returns_empty(self):
        source = "def broken(\n"
        assert extract_imports(source) == set()

    def test_relative_import_skipped(self):
        source = "from . import utils\nimport numpy\n"
        assert extract_imports(source) == {"numpy"}

    def test_conditional_import(self):
        source = "try:\n    import matplotlib\nexcept ImportError:\n    pass\n"
        assert extract_imports(source) == {"matplotlib"}


# ---------------------------------------------------------------------------
# parse_pep723_deps
# ---------------------------------------------------------------------------


class TestParsePep723Deps:
    def test_no_block(self):
        assert parse_pep723_deps("import numpy\n") == set()

    def test_standard_deps(self):
        source = (
            "# /// script\n"
            '# requires-python = ">=3.11"\n'
            "# dependencies = [\n"
            '#     "numpy",\n'
            '#     "pandas",\n'
            "# ]\n"
            "# ///\n"
        )
        assert parse_pep723_deps(source) == {"numpy", "pandas"}

    def test_version_specifiers_stripped(self):
        source = (
            "# /// script\n"
            "# dependencies = [\n"
            '#     "numpy>=1.20",\n'
            '#     "scipy~=1.10",\n'
            "# ]\n"
            "# ///\n"
        )
        assert parse_pep723_deps(source) == {"numpy", "scipy"}

    def test_name_mapping(self):
        source = '# /// script\n# dependencies = ["scikit-learn"]\n# ///\n'
        assert parse_pep723_deps(source) == {"sklearn"}

    def test_extras_stripped(self):
        source = '# /// script\n# dependencies = ["pandas[performance]>=2.0"]\n# ///\n'
        assert parse_pep723_deps(source) == {"pandas"}

    def test_hyphenated_package_name(self):
        source = '# /// script\n# dependencies = ["python-dateutil"]\n# ///\n'
        assert parse_pep723_deps(source) == {"dateutil"}

    def test_malformed_toml_returns_empty(self):
        source = "# /// script\n# this is not valid toml [[[\n# ///\n"
        assert parse_pep723_deps(source) == set()


# ---------------------------------------------------------------------------
# find_missing_deps
# ---------------------------------------------------------------------------


class TestFindMissingDeps:
    def test_all_declared(self):
        source = (
            "# /// script\n"
            '# dependencies = ["numpy", "pandas"]\n'
            "# ///\n"
            "import numpy\nimport pandas\n"
        )
        assert find_missing_deps(source) == []

    def test_missing_dep(self):
        source = '# /// script\n# dependencies = ["numpy"]\n# ///\nimport numpy\nimport seaborn\n'
        assert find_missing_deps(source) == ["seaborn"]

    def test_mapped_dep_declared(self):
        source = '# /// script\n# dependencies = ["scikit-learn"]\n# ///\nimport sklearn\n'
        assert find_missing_deps(source) == []

    def test_mapped_dep_missing(self):
        source = '# /// script\n# dependencies = ["numpy"]\n# ///\nimport sklearn\n'
        assert find_missing_deps(source) == ["scikit-learn"]

    def test_no_block_flags_all(self):
        source = "import numpy\nimport pandas\n"
        assert sorted(find_missing_deps(source)) == ["numpy", "pandas"]

    def test_stdlib_only_returns_empty(self):
        source = "# /// script\n# dependencies = []\n# ///\nimport os\nimport json\n"
        assert find_missing_deps(source) == []


# ---------------------------------------------------------------------------
# patch_pep723_block
# ---------------------------------------------------------------------------


class TestPatchPep723Block:
    def test_adds_to_existing_block(self):
        source = (
            "# /// script\n"
            '# requires-python = ">=3.11"\n'
            "# dependencies = [\n"
            '#     "numpy",\n'
            "# ]\n"
            "# ///\n"
            "\nimport numpy\nimport matplotlib\n"
        )
        patched = patch_pep723_block(source, ["matplotlib"])
        assert '"matplotlib"' in patched
        assert '"numpy"' in patched
        assert "import numpy" in patched

    def test_no_duplicates(self):
        source = '# /// script\n# dependencies = ["numpy"]\n# ///\n'
        patched = patch_pep723_block(source, ["numpy"])
        assert patched.count('"numpy"') == 1

    def test_creates_block_if_missing(self):
        source = "import numpy\nprint('hi')\n"
        patched = patch_pep723_block(source, ["numpy"])
        assert "# /// script" in patched
        assert '"numpy"' in patched
        assert "import numpy" in patched

    def test_empty_missing_returns_unchanged(self):
        source = "import os\n"
        assert patch_pep723_block(source, []) == source


# ---------------------------------------------------------------------------
# ensure_deps (integration)
# ---------------------------------------------------------------------------


class TestEnsureDeps:
    def test_patches_in_place(self, tmp_path):
        script = tmp_path / "experiment.py"
        script.write_text(
            '# /// script\n# dependencies = ["numpy"]\n# ///\nimport numpy\nimport matplotlib\n'
        )
        added = ensure_deps(script)
        assert added == ["matplotlib"]
        patched = script.read_text()
        assert '"matplotlib"' in patched
        assert '"numpy"' in patched

    def test_no_change_when_complete(self, tmp_path):
        original = (
            "# /// script\n"
            '# dependencies = ["numpy", "pandas"]\n'
            "# ///\n"
            "import numpy\nimport pandas\n"
        )
        script = tmp_path / "experiment.py"
        script.write_text(original)
        added = ensure_deps(script)
        assert added == []
        assert script.read_text() == original

    def test_mapped_dep_added_correctly(self, tmp_path):
        script = tmp_path / "experiment.py"
        script.write_text(
            '# /// script\n# dependencies = ["numpy"]\n# ///\nimport numpy\nimport sklearn\n'
        )
        added = ensure_deps(script)
        assert added == ["scikit-learn"]
        patched = script.read_text()
        assert '"scikit-learn"' in patched


# ---------------------------------------------------------------------------
# validate_deps
# ---------------------------------------------------------------------------


class TestValidateDeps:
    def test_all_covered(self, tmp_path):
        script = tmp_path / "experiment.py"
        script.write_text('# /// script\n# dependencies = ["numpy"]\n# ///\nimport numpy\n')
        ok, msg = validate_deps(script)
        assert ok is True
        assert msg == ""

    def test_missing_dep_returns_message(self, tmp_path):
        script = tmp_path / "experiment.py"
        script.write_text(
            '# /// script\n# dependencies = ["numpy"]\n# ///\nimport numpy\nimport matplotlib\n'
        )
        ok, msg = validate_deps(script)
        assert ok is False
        assert "matplotlib" in msg

    def test_stdlib_only_passes(self, tmp_path):
        script = tmp_path / "experiment.py"
        script.write_text("import os\nimport json\n")
        ok, msg = validate_deps(script)
        assert ok is True
