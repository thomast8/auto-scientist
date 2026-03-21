"""Tests for console.py file logging (console.log tee)."""

import os
from unittest.mock import patch

from auto_scientist.console import (
    _log_to_file,
    close_console_log,
    init_console_log,
    print_step,
    print_summary,
)


class TestConsoleLog:
    def setup_method(self):
        close_console_log()

    def teardown_method(self):
        close_console_log()

    def test_init_creates_file(self, tmp_path):
        log_path = tmp_path / "console.log"
        init_console_log(log_path)
        _log_to_file("hello")
        close_console_log()

        assert log_path.exists()
        assert "hello" in log_path.read_text()

    def test_timestamp_added(self, tmp_path):
        log_path = tmp_path / "console.log"
        init_console_log(log_path)
        _log_to_file("timestamped line")
        close_console_log()

        content = log_path.read_text()
        # Timestamp format: [HH:MM:SS]
        assert content.startswith("[")
        assert "] timestamped line" in content

    def test_ansi_codes_stripped(self, tmp_path):
        log_path = tmp_path / "console.log"
        init_console_log(log_path)
        _log_to_file("\033[32mgreen text\033[0m")
        close_console_log()

        content = log_path.read_text()
        assert "\033[" not in content
        assert "green text" in content

    def test_noop_when_not_initialized(self):
        # Should not raise
        _log_to_file("no file open")

    def test_print_step_tees_to_file(self, tmp_path, capsys):
        log_path = tmp_path / "console.log"
        init_console_log(log_path)

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            print_step("ANALYZE: testing file tee")

        close_console_log()

        content = log_path.read_text()
        assert "ANALYZE: testing file tee" in content

        # Also printed to stdout
        captured = capsys.readouterr()
        assert "ANALYZE: testing file tee" in captured.out

    def test_print_summary_tees_to_file(self, tmp_path, capsys):
        log_path = tmp_path / "console.log"
        init_console_log(log_path)

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            print_summary("Analyst", "Found 3 metrics.", label="15s")

        close_console_log()

        content = log_path.read_text()
        assert "Found 3 metrics." in content

    def test_append_mode_for_resume(self, tmp_path):
        log_path = tmp_path / "console.log"

        init_console_log(log_path)
        _log_to_file("first run")
        close_console_log()

        init_console_log(log_path)
        _log_to_file("second run")
        close_console_log()

        content = log_path.read_text()
        assert "first run" in content
        assert "second run" in content

    def test_close_is_idempotent(self):
        close_console_log()
        close_console_log()  # Should not raise

    def test_multiline_message(self, tmp_path):
        log_path = tmp_path / "console.log"
        init_console_log(log_path)
        _log_to_file("line1\nline2\nline3")
        close_console_log()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3
        assert "line1" in lines[0]
        assert "line2" in lines[1]
        assert "line3" in lines[2]
