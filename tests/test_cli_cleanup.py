from __future__ import annotations

import signal
from unittest.mock import call, patch

import auto_core.cli_cleanup as cli_cleanup


def _reset_cleanup_state() -> None:
    cli_cleanup._cleanup_done.clear()
    cli_cleanup._cleanup_handlers_installed.clear()


class TestCliCleanup:
    def setup_method(self) -> None:
        _reset_cleanup_state()

    def teardown_method(self) -> None:
        _reset_cleanup_state()

    @patch("auto_core.cli_cleanup.time.sleep", lambda _s: None)
    @patch("auto_core.cli_cleanup._descendant_pids", return_value=[456, 123])
    @patch("auto_core.cli_cleanup.os.kill")
    @patch("auto_core.cli_cleanup.signal.signal")
    def test_kill_child_processes_sigterms_then_sigkills_descendants(
        self,
        mock_signal,
        mock_kill,
        _mock_descendants,
    ) -> None:
        cli_cleanup.kill_child_processes()
        cli_cleanup.kill_child_processes()

        mock_signal.assert_has_calls(
            [
                call(signal.SIGTERM, signal.SIG_IGN),
                call(signal.SIGHUP, signal.SIG_IGN),
            ],
            any_order=True,
        )
        # First SIGTERM every descendant, then SIGKILL every descendant.
        assert mock_kill.call_args_list == [
            call(456, signal.SIGTERM),
            call(123, signal.SIGTERM),
            call(456, signal.SIGKILL),
            call(123, signal.SIGKILL),
        ]

    @patch("auto_core.cli_cleanup.atexit.register")
    @patch("auto_core.cli_cleanup.signal.signal")
    def test_install_child_cleanup_handlers_is_idempotent(
        self,
        mock_signal,
        mock_register,
    ) -> None:
        cli_cleanup.install_child_cleanup_handlers()
        cli_cleanup.install_child_cleanup_handlers()

        mock_signal.assert_has_calls(
            [
                call(signal.SIGHUP, cli_cleanup._fatal_signal_handler),
                call(signal.SIGTERM, cli_cleanup._fatal_signal_handler),
            ]
        )
        mock_register.assert_called_once_with(cli_cleanup.kill_child_processes)
