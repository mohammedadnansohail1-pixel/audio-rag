"""Tests for timeout patterns."""

import pytest
import asyncio
from audio_rag.core.resilience import (
    TimeoutConfig,
    TimeoutError,
    async_timeout,
    with_timeout,
    calculate_asr_timeout,
    calculate_diarization_timeout,
    DEFAULT_TIMEOUTS,
)


class TestTimeoutConfig:
    def test_default_values(self):
        config = TimeoutConfig()
        assert config.health_check == 5.0
        assert config.redis_connect == 5.0
        assert config.model_load == 300.0

    def test_custom_values(self):
        config = TimeoutConfig(health_check=10.0, model_load=600.0)
        assert config.health_check == 10.0
        assert config.model_load == 600.0


class TestAsyncTimeout:
    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        async def quick_task():
            return "done"
        
        result = await async_timeout(quick_task(), timeout=1.0)
        assert result == "done"

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        async def slow_task():
            await asyncio.sleep(1.0)
            return "done"
        
        with pytest.raises(TimeoutError) as exc_info:
            await async_timeout(slow_task(), timeout=0.1, operation="slow_op")
        
        assert exc_info.value.operation == "slow_op"
        assert exc_info.value.timeout == 0.1


class TestWithTimeoutDecorator:
    @pytest.mark.asyncio
    async def test_decorated_function_completes(self):
        @with_timeout(1.0)
        async def quick_func():
            return "success"
        
        result = await quick_func()
        assert result == "success"

    @pytest.mark.asyncio
    async def test_decorated_function_times_out(self):
        @with_timeout(0.1, operation="slow_decorated")
        async def slow_func():
            await asyncio.sleep(1.0)
            return "never"
        
        with pytest.raises(TimeoutError) as exc_info:
            await slow_func()
        
        assert "slow_decorated" in str(exc_info.value)


class TestCalculateTimeouts:
    def test_asr_timeout_minimum(self):
        # Very short audio should still get minimum timeout
        timeout = calculate_asr_timeout(10)  # 10 seconds
        assert timeout >= 60.0

    def test_asr_timeout_scales_with_duration(self):
        short_timeout = calculate_asr_timeout(60)  # 1 minute
        long_timeout = calculate_asr_timeout(600)  # 10 minutes
        assert long_timeout > short_timeout

    def test_asr_timeout_maximum(self):
        # Very long audio should cap at maximum
        timeout = calculate_asr_timeout(36000)  # 10 hours
        assert timeout <= 3600.0  # 1 hour max

    def test_diarization_timeout_minimum(self):
        timeout = calculate_diarization_timeout(10)
        assert timeout >= 30.0

    def test_diarization_timeout_scales(self):
        short = calculate_diarization_timeout(60)
        long = calculate_diarization_timeout(600)
        assert long > short

    def test_diarization_timeout_maximum(self):
        timeout = calculate_diarization_timeout(36000)
        assert timeout <= 1800.0  # 30 min max
