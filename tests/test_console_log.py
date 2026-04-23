"""Tests for console file logging via PipelineLive.

Replaces the old test_console_log.py that tested init_console_log/close_console_log.
The new file logging uses Rich Console(file=..., no_color=True) under PipelineLive.
"""

from auto_core.pipeline_live import PipelineLive


class TestConsoleLog:
    def test_log_creates_file(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("hello")
        live.stop()
        assert log_path.exists()
        assert "hello" in log_path.read_text()

    def test_no_ansi_in_log(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("[bold red]styled text[/]")
        live.stop()
        content = log_path.read_text()
        assert "\033[" not in content
        assert "styled text" in content

    def test_noop_when_no_log_path(self):
        live = PipelineLive()
        live.start()
        live.log("no file open")  # Should not raise
        live.stop()

    def test_append_mode(self, tmp_path):
        log_path = tmp_path / "console.log"

        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("first run")
        live.stop()

        live2 = PipelineLive()
        live2.start(log_path=log_path)
        live2.log("second run")
        live2.stop()

        content = log_path.read_text()
        assert "first run" in content
        assert "second run" in content

    def test_stop_is_idempotent(self):
        live = PipelineLive()
        live.start()
        live.stop()
        live.stop()  # Should not raise
