"""Tests for console streaming helpers."""

import os
from unittest.mock import patch

from auto_scientist.console import (
    BLUE,
    CYAN,
    GREEN,
    MAGENTA,
    RED,
    YELLOW,
    _color_for_label,
    make_stream_printer,
    print_summary,
    stream_separator,
)


class TestColorForLabel:
    def test_critic_gets_yellow(self):
        assert _color_for_label("Critic (openai:gpt-4o) round 1") == YELLOW

    def test_scientist_gets_cyan(self):
        assert _color_for_label("Scientist round 1") == CYAN

    def test_coder_gets_magenta(self):
        assert _color_for_label("Coder") == MAGENTA

    def test_analyst_gets_green(self):
        assert _color_for_label("Analyst iteration 1") == GREEN

    def test_ingestor_gets_red(self):
        assert _color_for_label("Ingestor") == RED

    def test_report_gets_blue(self):
        assert _color_for_label("Report generation") == BLUE

    def test_debate_gets_yellow(self):
        assert _color_for_label("Debate") == YELLOW

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


class TestPrintSummary:
    def test_with_label(self, capsys):
        print_summary("Analyst", "Found key metrics.", label="done")
        captured = capsys.readouterr()
        assert "> [done] " in captured.out
        assert "Found key metrics." in captured.out

    def test_without_label(self, capsys):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            print_summary("Results", "R2=0.82 on test set.")
        captured = capsys.readouterr()
        assert "> R2=0.82" in captured.out
        # No brackets when label is empty (check without ANSI codes)
        assert "[" not in captured.out

    def test_truncates_long_progress_text(self, capsys):
        long_text = "x" * 250
        print_summary("Analyst", long_text, label="15s")
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_truncates_long_final_text(self, capsys):
        long_text = "x" * 450
        print_summary("Analyst", long_text, label="done")
        captured = capsys.readouterr()
        assert "..." in captured.out

    def test_uses_agent_color(self, capsys):
        print_summary("Analyst", "test summary", label="done")
        captured = capsys.readouterr()
        assert GREEN in captured.out

    def test_empty_prints_nothing(self, capsys):
        print_summary("Analyst", "")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_no_color_env(self, capsys):
        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            print_summary("Analyst", "test summary", label="done")
        captured = capsys.readouterr()
        assert "\033[" not in captured.out
        assert "[done]" in captured.out

    def test_truncates_long_summary(self, capsys):
        long_text = "a" * 500
        print_summary("Analyst", long_text, label="done")
        captured = capsys.readouterr()
        # The full 500 chars should not appear
        assert "a" * 500 not in captured.out


class TestStreamSeparator:
    def test_prints_newlines(self, capsys):
        stream_separator()
        captured = capsys.readouterr()
        assert captured.out == "\n\n"
