"""Config-loader tests.

Covers the happy path: top-level overrides, single nested overrides.
Does NOT exercise the sibling-preservation invariant the docstring
promises; that gap is part of the fixture.
"""

from src.config import apply_env_overrides, _coerce


def test_top_level_override() -> None:
    config = {"debug": False}
    result = apply_env_overrides(config, {"DEBUG": "true"})
    assert result["debug"] is True


def test_nested_override_single_key() -> None:
    # Note: `db` only has `host` here, so whole-dict replacement is
    # indistinguishable from in-place update. This test passes in both
    # branches.
    config = {"db": {"host": "prod"}}
    result = apply_env_overrides(config, {"DB_HOST": "dev"})
    assert result["db"]["host"] == "dev"


def test_coerce_types() -> None:
    assert _coerce("true") is True
    assert _coerce("false") is False
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("hello") == "hello"
