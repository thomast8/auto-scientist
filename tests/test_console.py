"""Tests for console streaming helpers."""

import os
from unittest.mock import patch

from auto_scientist.console import (
    CYAN,
    MAGENTA,
    YELLOW,
    _color_for_label,
    make_stream_printer,
    stream_separator,
)


class TestColorForLabel:
    def test_critic_gets_yellow(self):
        assert _color_for_label("Critic (openai:gpt-4o) round 1") == YELLOW

    def test_scientist_gets_cyan(self):
        assert _color_for_label("Scientist round 1") == CYAN

    def test_coder_gets_magenta(self):
        assert _color_for_label("Coder") == MAGENTA

    def test_unknown_falls_back_to_cyan(self):
        assert _color_for_label("Unknown agent") == CYAN


class TestMakeStreamPrinter:
    def test_first_call_prints_label_then_token(self, capsys):
        printer = make_stream_printer("Critic (openai:gpt-4o)")
        printer("Hello")

        captured = capsys.readouterr()
        assert "Critic (openai:gpt-4o)" in captured.out
        assert "Hello" in captured.out

    def test_subsequent_calls_print_only_token(self, capsys):
        printer = make_stream_printer("Critic")
        printer("Hello")
        printer(" world")

        captured = capsys.readouterr()
        # Label appears exactly once
        assert captured.out.count("Critic") == 1
        assert "Hello world" in captured.out

    def test_critic_label_uses_yellow(self, capsys):
        printer = make_stream_printer("Critic (openai:gpt-4o)")
        printer("x")
        captured = capsys.readouterr()
        assert YELLOW in captured.out

    def test_scientist_label_uses_cyan(self, capsys):
        printer = make_stream_printer("Scientist round 1")
        printer("x")
        captured = capsys.readouterr()
        assert CYAN in captured.out

    def test_no_color_env_strips_ansi(self, capsys):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            printer = make_stream_printer("Test")
            printer("token")

        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "Test" in captured.out
        assert "token" in captured.out


class TestStreamSeparator:
    def test_prints_newlines(self, capsys):
        stream_separator()
        captured = capsys.readouterr()
        assert captured.out == "\n\n"
