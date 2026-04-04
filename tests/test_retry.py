"""Tests for the shared agent retry loop infrastructure."""

from __future__ import annotations

import pytest

from auto_scientist.retry import (
    QueryResult,
    ValidationError,
    agent_retry_loop,
)

# ---------------------------------------------------------------------------
# ValidationError basics
# ---------------------------------------------------------------------------


class TestValidationError:
    def test_stores_correction_hint(self):
        err = ValidationError("fix your JSON")
        assert err.correction_hint == "fix your JSON"

    def test_is_exception(self):
        assert issubclass(ValidationError, Exception)


# ---------------------------------------------------------------------------
# QueryResult basics
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_fields(self):
        qr = QueryResult(raw_output="hello", session_id="sess-1", usage={"input_tokens": 10})
        assert qr.raw_output == "hello"
        assert qr.session_id == "sess-1"
        assert qr.usage == {"input_tokens": 10}

    def test_session_id_optional(self):
        qr = QueryResult(raw_output="hello", session_id=None, usage={})
        assert qr.session_id is None


# ---------------------------------------------------------------------------
# agent_retry_loop
# ---------------------------------------------------------------------------


class TestAgentRetryLoop:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """When query + validation succeed, return immediately."""
        call_count = 0

        async def query_fn(prompt, session_id):
            nonlocal call_count
            call_count += 1
            return QueryResult(raw_output='{"ok": true}', session_id="s1", usage={})

        def validate_fn(result):
            return {"ok": True}

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="test prompt",
            agent_name="Test",
        )
        assert out == {"ok": True}
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_sdk_error_retries_fresh(self):
        """SDK errors should retry with no session_id (fresh start)."""
        attempts = []

        async def query_fn(prompt, session_id):
            attempts.append({"prompt": prompt, "session_id": session_id})
            if len(attempts) < 3:
                raise ConnectionError("timeout")
            return QueryResult(raw_output="ok", session_id="s1", usage={})

        def validate_fn(result):
            return result.raw_output

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="test",
            agent_name="Test",
        )
        assert out == "ok"
        assert len(attempts) == 3
        # All SDK retries should be fresh (no session_id)
        for a in attempts:
            assert a["session_id"] is None

    @pytest.mark.asyncio
    async def test_sdk_error_exhaustion_raises(self):
        """When all attempts fail with SDK errors and no on_exhausted, re-raise."""

        async def query_fn(prompt, session_id):
            raise ConnectionError("down")

        def validate_fn(result):
            return result

        with pytest.raises(ConnectionError, match="down"):
            await agent_retry_loop(
                query_fn=query_fn,
                validate_fn=validate_fn,
                prompt="test",
                agent_name="Test",
                max_attempts=2,
            )

    @pytest.mark.asyncio
    async def test_validation_error_retries_with_resume(self):
        """Validation failures should retry with session_id and correction hint."""
        attempts = []

        async def query_fn(prompt, session_id):
            attempts.append({"prompt": prompt, "session_id": session_id})
            return QueryResult(raw_output="bad json", session_id="sess-42", usage={})

        call_count = 0

        def validate_fn(result):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValidationError("invalid JSON")
            return {"parsed": True}

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="original",
            agent_name="Test",
        )
        assert out == {"parsed": True}
        assert len(attempts) == 2
        # First attempt: no session_id
        assert attempts[0]["session_id"] is None
        assert "invalid JSON" not in attempts[0]["prompt"]
        # Second attempt: resume with session_id, correction hint appended
        assert attempts[1]["session_id"] == "sess-42"
        assert "invalid JSON" in attempts[1]["prompt"]

    @pytest.mark.asyncio
    async def test_validation_exhaustion_with_on_exhausted(self):
        """on_exhausted callback is invoked when all attempts fail validation."""

        async def query_fn(prompt, session_id):
            return QueryResult(raw_output="bad", session_id="s1", usage={})

        def validate_fn(result):
            raise ValidationError("still bad")

        def on_exhausted(result, error):
            return f"fallback: {result.raw_output}"

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="test",
            agent_name="Test",
            max_attempts=2,
            on_exhausted=on_exhausted,
        )
        assert out == "fallback: bad"

    @pytest.mark.asyncio
    async def test_validation_exhaustion_without_on_exhausted_raises(self):
        """Without on_exhausted, validation exhaustion re-raises."""

        async def query_fn(prompt, session_id):
            return QueryResult(raw_output="bad", session_id=None, usage={})

        def validate_fn(result):
            raise ValidationError("nope")

        with pytest.raises(ValidationError, match="nope"):
            await agent_retry_loop(
                query_fn=query_fn,
                validate_fn=validate_fn,
                prompt="test",
                agent_name="Test",
                max_attempts=2,
            )

    @pytest.mark.asyncio
    async def test_sdk_exhaustion_with_on_exhausted(self):
        """on_exhausted is called with result=None for SDK errors."""

        async def query_fn(prompt, session_id):
            raise RuntimeError("boom")

        def validate_fn(result):
            return result

        def on_exhausted(result, error):
            assert result is None
            return "recovered"

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="test",
            agent_name="Test",
            max_attempts=1,
            on_exhausted=on_exhausted,
        )
        assert out == "recovered"

    @pytest.mark.asyncio
    async def test_retryable_errors_filter(self):
        """Non-retryable errors propagate immediately, not retried."""
        call_count = 0

        async def query_fn(prompt, session_id):
            nonlocal call_count
            call_count += 1
            raise ValueError("auth failed")

        def validate_fn(result):
            return result

        with pytest.raises(ValueError, match="auth failed"):
            await agent_retry_loop(
                query_fn=query_fn,
                validate_fn=validate_fn,
                prompt="test",
                agent_name="Test",
                max_attempts=3,
                retryable_errors=(ConnectionError, TimeoutError),
            )
        # Should have been called only once - not retried
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retryable_errors_are_retried(self):
        """Errors matching retryable_errors are retried."""
        call_count = 0

        async def query_fn(prompt, session_id):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return QueryResult(raw_output="ok", session_id=None, usage={})

        def validate_fn(result):
            return result.raw_output

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="test",
            agent_name="Test",
            max_attempts=3,
            retryable_errors=(ConnectionError, TimeoutError),
        )
        assert out == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_default_max_attempts_is_three(self):
        """Default max_attempts is 3."""
        call_count = 0

        async def query_fn(prompt, session_id):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        def validate_fn(result):
            return result

        with pytest.raises(ConnectionError):
            await agent_retry_loop(
                query_fn=query_fn,
                validate_fn=validate_fn,
                prompt="test",
                agent_name="Test",
            )
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_correction_hint_preserved_after_sdk_error(self):
        """If a validation failure is followed by an SDK error, correction hint is kept."""
        attempts = []

        async def query_fn(prompt, session_id):
            attempts.append({"prompt": prompt, "session_id": session_id})
            if len(attempts) == 2:
                raise ConnectionError("network blip")
            return QueryResult(raw_output="data", session_id="s1", usage={})

        call_count = 0

        def validate_fn(result):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValidationError("bad format")
            return "success"

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="original",
            agent_name="Test",
        )
        assert out == "success"
        assert len(attempts) == 3
        # Attempt 1: fresh
        assert attempts[0]["session_id"] is None
        assert "bad format" not in attempts[0]["prompt"]
        # Attempt 2: would have had correction hint + session_id, but SDK error
        assert attempts[1]["session_id"] == "s1"
        assert "bad format" in attempts[1]["prompt"]
        # Attempt 3: after SDK error, session cleared but hint preserved
        assert attempts[2]["session_id"] is None
        assert "bad format" in attempts[2]["prompt"]

    @pytest.mark.asyncio
    async def test_no_session_id_means_no_resume(self):
        """When query returns session_id=None, retry is fresh even after validation failure."""
        attempts = []

        async def query_fn(prompt, session_id):
            attempts.append({"session_id": session_id})
            return QueryResult(raw_output="data", session_id=None, usage={})

        call_count = 0

        def validate_fn(result):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValidationError("wrong")
            return "ok"

        out = await agent_retry_loop(
            query_fn=query_fn,
            validate_fn=validate_fn,
            prompt="test",
            agent_name="Test",
        )
        assert out == "ok"
        # Both attempts should have session_id=None
        assert attempts[0]["session_id"] is None
        assert attempts[1]["session_id"] is None
