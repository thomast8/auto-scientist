"""Tests for the RunResult model."""

from auto_scientist.runner import RunResult


class TestRunResult:
    def test_default_fields(self):
        r = RunResult(success=True)
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.return_code == -1
        assert r.timed_out is False
        assert r.output_files == []

    def test_all_fields(self):
        r = RunResult(
            success=True, stdout="out", stderr="err",
            return_code=0, timed_out=False,
            output_files=["a.txt", "b.png"],
        )
        assert r.stdout == "out"
        assert r.stderr == "err"
        assert r.return_code == 0
        assert r.output_files == ["a.txt", "b.png"]
