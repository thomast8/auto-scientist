"""Tests for compressed history generation."""

from auto_scientist.history import build_compressed_history
from auto_scientist.state import ExperimentState, VersionEntry


class TestCompressedHistory:
    def test_empty_history(self):
        state = ExperimentState(domain="test", goal="g")
        result = build_compressed_history(state)
        assert "No iterations" in result

    def test_single_version(self):
        state = ExperimentState(domain="test", goal="g")
        state.record_version(
            VersionEntry(
                version="v1.01",
                iteration=1,
                script_path="s.py",
                score=5,
                hypothesis="Try linear model",
                status="completed",
            )
        )
        result = build_compressed_history(state)
        assert "v1.01" in result
        assert "score=5" in result
        assert "Try linear model" in result
        assert "Best so far: v1.01" in result

    def test_multiple_versions(self):
        state = ExperimentState(domain="test", goal="g")
        for i in range(3):
            state.record_version(
                VersionEntry(
                    version=f"v1.{i+1:02d}",
                    iteration=i + 1,
                    script_path="s.py",
                    score=i * 3,
                    hypothesis=f"Hypothesis {i+1}",
                    status="completed",
                )
            )
        result = build_compressed_history(state)
        assert "3 versions" in result
        assert "v1.01" in result
        assert "v1.03" in result

    def test_dead_ends_shown(self):
        state = ExperimentState(domain="test", goal="g")
        state.dead_ends = ["linear model", "polynomial model"]
        state.record_version(
            VersionEntry(version="v1", iteration=1, script_path="s.py", status="completed")
        )
        result = build_compressed_history(state)
        assert "Dead ends" in result
        assert "linear model" in result

    def test_long_hypothesis_truncated(self):
        state = ExperimentState(domain="test", goal="g")
        long_hyp = "A" * 200
        state.record_version(
            VersionEntry(
                version="v1",
                iteration=1,
                script_path="s.py",
                hypothesis=long_hyp,
                status="completed",
            )
        )
        result = build_compressed_history(state)
        assert "..." in result
        # Should be truncated at 80 chars + "..."
        for line in result.split("\n"):
            if "v1" in line and "AAAA" in line:
                hyp_part = line.split(" - ", 1)[1] if " - " in line else ""
                assert len(hyp_part) <= 84  # 80 + "..."

    def test_status_icons(self):
        state = ExperimentState(domain="test", goal="g")
        for status in ["completed", "failed", "crashed"]:
            state.record_version(
                VersionEntry(
                    version=f"v_{status}",
                    iteration=1,
                    script_path="s.py",
                    status=status,
                )
            )
        result = build_compressed_history(state)
        assert "[ok]" in result
        assert "[FAIL]" in result
        assert "[CRASH]" in result
