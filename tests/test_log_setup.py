"""Tests for log_setup module."""

import logging

from auto_core.log_setup import setup_file_logging


class TestSetupFileLogging:
    def test_creates_debug_log_file(self, tmp_path):
        setup_file_logging(tmp_path)

        log_file = tmp_path / "debug.log"
        logger = logging.getLogger("auto_scientist")
        logger.info("test message")

        assert log_file.exists()
        contents = log_file.read_text()
        assert "test message" in contents

    def test_file_handler_format(self, tmp_path):
        setup_file_logging(tmp_path)

        logger = logging.getLogger("auto_scientist")
        logger.warning("format check")

        contents = (tmp_path / "debug.log").read_text()
        assert "WARNING" in contents
        assert "auto_scientist" in contents

    def test_no_duplicate_handlers_on_repeated_calls(self, tmp_path):
        setup_file_logging(tmp_path)
        setup_file_logging(tmp_path)

        logger = logging.getLogger("auto_scientist")
        file_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.FileHandler) and h.baseFilename.endswith("debug.log")
        ]
        assert len(file_handlers) == 1

    def test_verbose_adds_stream_handler(self, tmp_path):
        setup_file_logging(tmp_path, verbose=True)

        logger = logging.getLogger("auto_scientist")
        stream_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1

    def test_debug_level_captured(self, tmp_path):
        setup_file_logging(tmp_path)

        logger = logging.getLogger("auto_scientist")
        logger.debug("debug detail")

        contents = (tmp_path / "debug.log").read_text()
        assert "debug detail" in contents

    def setup_method(self):
        """Remove handlers added by previous tests."""
        logger = logging.getLogger("auto_scientist")
        logger.handlers.clear()
