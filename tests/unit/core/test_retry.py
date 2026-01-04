"""Tests for retry patterns."""

import pytest
from audio_rag.core.resilience import (
    retry_with_backoff,
    RetryError,
)


class TestRetryWithBackoff:
    def test_succeeds_without_retry(self):
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, log_retries=False)
        def succeed():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = succeed()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_failure(self):
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, min_wait=0.01, max_wait=0.1, log_retries=False)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("fail")
            return "success"
        
        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        call_count = 0
        
        @retry_with_backoff(max_attempts=3, min_wait=0.01, max_wait=0.1, log_retries=False)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("always fails")
        
        with pytest.raises(ValueError):
            always_fail()
        
        assert call_count == 3

    def test_only_retries_specified_exceptions(self):
        call_count = 0
        
        @retry_with_backoff(
            max_attempts=3, 
            exceptions=(ConnectionError,),
            min_wait=0.01,
            log_retries=False,
        )
        def wrong_exception():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")
        
        with pytest.raises(ValueError):
            wrong_exception()
        
        # Should not retry on ValueError
        assert call_count == 1

    def test_retries_specified_exception(self):
        call_count = 0
        
        @retry_with_backoff(
            max_attempts=3,
            exceptions=(ConnectionError,),
            min_wait=0.01,
            max_wait=0.1,
            log_retries=False,
        )
        def connection_error_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("connection failed")
            return "connected"
        
        result = connection_error_then_succeed()
        assert result == "connected"
        assert call_count == 2
