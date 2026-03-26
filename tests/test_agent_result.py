"""Tests for AgentResult dataclass."""

from auto_scientist.agent_result import AgentResult


class TestAgentResult:
    def test_text_only_defaults(self):
        r = AgentResult("hello")
        assert r.text == "hello"
        assert r.input_tokens == 0
        assert r.output_tokens == 0

    def test_explicit_tokens(self):
        r = AgentResult("response", input_tokens=100, output_tokens=50)
        assert r.text == "response"
        assert r.input_tokens == 100
        assert r.output_tokens == 50

    def test_empty_text(self):
        r = AgentResult("")
        assert r.text == ""
        assert r.input_tokens == 0
        assert r.output_tokens == 0
