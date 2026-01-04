"""Tests for circuit breaker pattern."""

import pytest
import time
from audio_rag.core.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
)


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        config = CircuitBreakerConfig(name="test")
        breaker = CircuitBreaker(config)
        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        config = CircuitBreakerConfig(name="test", failure_threshold=3)
        breaker = CircuitBreaker(config)
        
        for _ in range(3):
            breaker.record_failure(Exception("fail"))
        
        assert breaker.state == CircuitState.OPEN

    def test_stays_closed_below_threshold(self):
        config = CircuitBreakerConfig(name="test", failure_threshold=3)
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))
        
        assert breaker.state == CircuitState.CLOSED

    def test_success_resets_failure_count(self):
        config = CircuitBreakerConfig(name="test", failure_threshold=3)
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        breaker.record_failure(Exception("fail"))
        breaker.record_success()
        breaker.record_failure(Exception("fail"))
        
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 1


class TestCircuitBreakerCall:
    def test_call_success(self):
        config = CircuitBreakerConfig(name="test")
        breaker = CircuitBreaker(config)
        
        result = breaker.call(lambda: "success")
        
        assert result == "success"
        assert breaker.stats.successes == 1

    def test_call_failure_records(self):
        config = CircuitBreakerConfig(name="test")
        breaker = CircuitBreaker(config)
        
        with pytest.raises(ValueError):
            breaker.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        
        assert breaker.stats.failures == 1

    def test_call_rejected_when_open(self):
        config = CircuitBreakerConfig(name="test", failure_threshold=1)
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        
        with pytest.raises(CircuitBreakerError) as exc_info:
            breaker.call(lambda: "should not run")
        
        assert "test" in str(exc_info.value)


class TestCircuitBreakerRecovery:
    def test_transitions_to_half_open_after_timeout(self):
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            reset_timeout=0.1,
        )
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN
        
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_closes_on_success(self):
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            success_threshold=1,
            reset_timeout=0.1,
        )
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        time.sleep(0.15)
        
        breaker.record_success()
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            reset_timeout=0.1,
        )
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        time.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN
        
        breaker.record_failure(Exception("fail again"))
        assert breaker.state == CircuitState.OPEN


class TestCircuitBreakerReset:
    def test_manual_reset(self):
        config = CircuitBreakerConfig(name="test", failure_threshold=1)
        breaker = CircuitBreaker(config)
        
        breaker.record_failure(Exception("fail"))
        assert breaker.state == CircuitState.OPEN
        
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.consecutive_failures == 0
