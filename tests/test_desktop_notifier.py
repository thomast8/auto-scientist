"""Tests for DesktopNotifier: macOS desktop notifications via `alerter`."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from auto_core.desktop_notifier import DesktopNotifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_alerter(monkeypatch: pytest.MonkeyPatch) -> str:
    """Pretend `alerter` is installed at a known path."""
    path = "/opt/homebrew/bin/alerter"
    monkeypatch.setattr(
        "auto_core.desktop_notifier.shutil.which",
        lambda name: path if name == "alerter" else None,
    )
    return path


@pytest.fixture
def mock_popen(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Capture all subprocess.Popen calls."""
    mock = MagicMock(name="Popen")
    monkeypatch.setattr(
        "auto_core.desktop_notifier.subprocess.Popen",
        mock,
    )
    return mock


def _args_of(call) -> list[str]:
    return list(call.args[0])


def _kwargs_of(call) -> dict:
    return dict(call.kwargs)


def _get_arg(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


# ---------------------------------------------------------------------------
# Level gating
# ---------------------------------------------------------------------------


def test_level_off_never_fires(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="off", run_name="test")
    n.agent_done("Analyst", "12s", "done")
    n.iteration_done("Iteration 1/5", "hello")
    n.run_complete("complete", "done")
    assert mock_popen.call_count == 0


def test_level_run_only_fires_run_complete(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="run", run_name="test")
    n.agent_done("Analyst", "12s", "done")
    n.iteration_done("Iteration 1/5", "hello")
    assert mock_popen.call_count == 0
    n.run_complete("complete", "4 iterations")
    assert mock_popen.call_count == 1


def test_level_iteration_fires_iteration_and_run(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="iteration", run_name="test")
    n.agent_done("Analyst", "12s", "done")
    assert mock_popen.call_count == 0
    n.iteration_done("Iteration 1/5", "hello")
    n.run_complete("complete", "4 iterations")
    assert mock_popen.call_count == 2


def test_level_agent_fires_everything(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="agent", run_name="test")
    n.agent_done("Analyst", "12s", "analyst done")
    n.iteration_done("Iteration 1/5", "hello")
    n.run_complete("complete", "4 iterations")
    assert mock_popen.call_count == 3


def test_unknown_level_treated_as_off(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="nonsense", run_name="test")
    n.run_complete("complete", "done")
    assert mock_popen.call_count == 0


# ---------------------------------------------------------------------------
# Missing binary / platform gating
# ---------------------------------------------------------------------------


def test_missing_alerter_is_noop(mock_popen: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "auto_core.desktop_notifier.shutil.which",
        lambda _name: None,
    )
    n = DesktopNotifier(level="agent", run_name="test")
    n.agent_done("Analyst", "12s", "done")
    n.iteration_done("Iteration 1/5", "hello")
    n.run_complete("complete", "done")
    assert mock_popen.call_count == 0


def test_missing_alerter_not_probed_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When level is off we should not even probe `which`."""
    calls: list[str] = []

    def _which(name: str) -> str | None:
        calls.append(name)
        return None

    monkeypatch.setattr("auto_core.desktop_notifier.shutil.which", _which)
    DesktopNotifier(level="off", run_name="test")
    assert calls == []


# ---------------------------------------------------------------------------
# Subprocess invocation shape
# ---------------------------------------------------------------------------


def test_run_complete_args_include_expected_flags(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="run", run_name="my-run")
    n.run_complete("complete", "4 iterations, 12 versions")

    args = _args_of(mock_popen.call_args)
    assert args[0] == fake_alerter
    assert _get_arg(args, "--title") == "auto-scientist complete"
    assert _get_arg(args, "--subtitle") == "my-run"
    assert _get_arg(args, "--group") == "auto-scientist"
    assert "--ignore-dnd" in args
    assert "--close-label" in args
    # Only run_complete plays a sound
    assert _get_arg(args, "--sound") == "default"
    # --actions flips alerter into sticky alert mode
    assert _get_arg(args, "--actions") == "OK"


def test_all_notifications_are_sticky(mock_popen: MagicMock, fake_alerter: str) -> None:
    """Every notification should use --actions to stay on screen."""
    n = DesktopNotifier(level="agent", run_name="r")
    n.set_iteration("Iteration 1/5")
    n.agent_done("Analyst", "5s", "ok")
    n.iteration_done("Iteration 1/5: done", "summary")
    n.run_complete("complete", "done")
    for call in mock_popen.call_args_list:
        args = _args_of(call)
        assert "--actions" in args, f"missing --actions: {args}"


def test_iteration_done_has_no_sound(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="iteration", run_name="r")
    n.iteration_done("Iteration 1/5", "hello")
    args = _args_of(mock_popen.call_args)
    assert "--sound" not in args


def test_agent_done_has_no_sound(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="agent", run_name="r")
    n.agent_done("Analyst", "12s", "done")
    args = _args_of(mock_popen.call_args)
    assert "--sound" not in args


def test_message_is_truncated(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="agent", run_name="r")
    long = "x" * 2000
    n.agent_done("Analyst", "12s", long)
    args = _args_of(mock_popen.call_args)
    message = _get_arg(args, "--message")
    assert len(message) <= 500
    assert "12s" in message


def test_message_strips_carriage_returns(mock_popen: MagicMock, fake_alerter: str) -> None:
    """Newlines are preserved (alerter renders them); \\r is stripped."""
    n = DesktopNotifier(level="agent", run_name="r")
    n.agent_done("Analyst", "12s", "line one\r\nline two")
    args = _args_of(mock_popen.call_args)
    message = _get_arg(args, "--message")
    assert "\r" not in message


def test_popen_is_detached(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="run", run_name="r")
    n.run_complete("complete", "done")
    kwargs = _kwargs_of(mock_popen.call_args)
    assert kwargs.get("start_new_session") is True
    assert kwargs.get("stdin") == subprocess.DEVNULL
    assert kwargs.get("stdout") == subprocess.DEVNULL
    assert kwargs.get("stderr") == subprocess.DEVNULL


def test_popen_oserror_is_swallowed(monkeypatch: pytest.MonkeyPatch, fake_alerter: str) -> None:
    def _boom(*_a, **_kw):
        raise OSError("exec failed")

    monkeypatch.setattr(
        "auto_core.desktop_notifier.subprocess.Popen",
        _boom,
    )
    n = DesktopNotifier(level="run", run_name="r")
    # Must not raise
    n.run_complete("complete", "done")


# ---------------------------------------------------------------------------
# Rich content: titles, iteration context, agent stats
# ---------------------------------------------------------------------------


def test_agent_done_carries_iteration_in_subtitle(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="agent", run_name="my-run")
    n.set_iteration("Iteration 3/20")
    n.agent_done("Scientist", "1m 23s", "plan drafted")

    args = _args_of(mock_popen.call_args)
    assert _get_arg(args, "--title") == "Scientist done"
    # Iteration context lives in the subtitle, not the body, to keep
    # the notification scannable.
    assert _get_arg(args, "--subtitle") == "my-run · Iteration 3/20"
    message = _get_arg(args, "--message")
    assert "1m 23s" in message
    assert "plan drafted" in message
    # Iteration label must NOT be duplicated in the body
    assert "Iteration 3/20" not in message


def test_agent_done_without_iteration_has_plain_subtitle(
    mock_popen: MagicMock, fake_alerter: str
) -> None:
    n = DesktopNotifier(level="agent", run_name="my-run")
    n.agent_done("Ingestor", "3s", "canonicalized 4 files")
    args = _args_of(mock_popen.call_args)
    assert _get_arg(args, "--subtitle") == "my-run"
    # The body stays clean: no "(no iteration)" placeholder
    message = _get_arg(args, "--message")
    assert "no iteration" not in message
    assert "3s" in message
    assert "canonicalized 4 files" in message


def test_agent_done_drops_turns_and_tokens_from_body(
    mock_popen: MagicMock, fake_alerter: str
) -> None:
    """Turns/tokens are accepted for API stability but omitted from the body."""
    n = DesktopNotifier(level="agent", run_name="r")
    n.set_iteration("Iteration 2/10")
    n.agent_done(
        "Coder",
        "45s",
        "script ran, acc=0.82",
        num_turns=12,
        total_tokens=45_321,
    )
    args = _args_of(mock_popen.call_args)
    message = _get_arg(args, "--message")
    assert "12 turns" not in message
    assert "45,321" not in message
    # But the core content is still there
    assert "45s" in message
    assert "acc=0.82" in message


def test_iteration_done_title_is_label(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="iteration", run_name="r")
    n.iteration_done("Iteration 3/20: done", "hypothesis refined")
    args = _args_of(mock_popen.call_args)
    assert _get_arg(args, "--title") == "Iteration 3/20: done"
    assert _get_arg(args, "--message") == "hypothesis refined"


def test_iteration_done_falls_back_to_label_when_summary_empty(
    mock_popen: MagicMock, fake_alerter: str
) -> None:
    n = DesktopNotifier(level="iteration", run_name="r")
    n.iteration_done("Iteration 1: done", "")
    args = _args_of(mock_popen.call_args)
    assert _get_arg(args, "--message") == "Iteration 1: done"


# ---------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------


def test_default_icon_is_used_when_bundled(mock_popen: MagicMock, fake_alerter: str) -> None:
    """The bundled src/auto_scientist/assets/notifier_icon.png is used by default."""
    n = DesktopNotifier(level="run", run_name="r")
    n.run_complete("complete", "done")
    args = _args_of(mock_popen.call_args)
    # The bundled icon file exists in this repo, so --app-icon must be set.
    assert "--app-icon" in args
    icon_path = _get_arg(args, "--app-icon")
    assert icon_path.endswith("notifier_icon.png")


def test_explicit_icon_overrides_default(
    mock_popen: MagicMock, fake_alerter: str, tmp_path: Path
) -> None:
    custom_icon = tmp_path / "custom.png"
    custom_icon.write_bytes(b"fake-png")
    n = DesktopNotifier(level="run", run_name="r", icon_path=custom_icon)
    n.run_complete("complete", "done")
    args = _args_of(mock_popen.call_args)
    assert _get_arg(args, "--app-icon") == str(custom_icon)


def test_missing_custom_icon_falls_back_to_default(
    mock_popen: MagicMock, fake_alerter: str, tmp_path: Path
) -> None:
    missing = tmp_path / "does-not-exist.png"
    n = DesktopNotifier(level="run", run_name="r", icon_path=missing)
    n.run_complete("complete", "done")
    args = _args_of(mock_popen.call_args)
    # Falls back to bundled default (which exists in the repo)
    assert "--app-icon" in args
    assert _get_arg(args, "--app-icon").endswith("notifier_icon.png")


def test_no_icon_flag_when_both_missing(
    mock_popen: MagicMock,
    fake_alerter: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pretend the bundled default does not exist."""
    fake_default = tmp_path / "nope.png"
    monkeypatch.setattr(
        "auto_core.desktop_notifier._DEFAULT_ICON",
        fake_default,
    )
    n = DesktopNotifier(level="run", run_name="r")
    n.run_complete("complete", "done")
    args = _args_of(mock_popen.call_args)
    assert "--app-icon" not in args


# ---------------------------------------------------------------------------
# Iteration context lifecycle
# ---------------------------------------------------------------------------


def test_set_iteration_updates_subsequent_calls(mock_popen: MagicMock, fake_alerter: str) -> None:
    n = DesktopNotifier(level="agent", run_name="r")
    n.set_iteration("Iteration 1/5")
    n.agent_done("Analyst", "1s", "ok")
    first_sub = _get_arg(_args_of(mock_popen.call_args_list[0]), "--subtitle")
    assert "Iteration 1/5" in first_sub

    n.set_iteration("Iteration 2/5")
    n.agent_done("Scientist", "2s", "ok")
    second_sub = _get_arg(_args_of(mock_popen.call_args_list[1]), "--subtitle")
    assert "Iteration 2/5" in second_sub
    assert "Iteration 1/5" not in second_sub


def test_constructor_defaults() -> None:
    # No monkeypatch — uses real shutil.which. Should not raise on any platform.
    n = DesktopNotifier()
    # Default level is off, so no binary is probed
    assert n._level == 0
