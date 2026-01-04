"""Tests for fallback chain pattern."""

import pytest
from audio_rag.core.resilience import (
    FallbackChain,
    FallbackExhaustedError,
)


class TestFallbackChain:
    def test_first_option_succeeds(self):
        chain = FallbackChain("test")
        chain.add("primary", lambda: "primary_result")
        chain.add("backup", lambda: "backup_result")
        
        result = chain.execute()
        assert result == "primary_result"

    def test_falls_back_on_failure(self):
        chain = FallbackChain("test")
        chain.add("primary", lambda: (_ for _ in ()).throw(ValueError("fail")))
        chain.add("backup", lambda: "backup_result")
        
        result = chain.execute()
        assert result == "backup_result"

    def test_skips_unavailable_options(self):
        chain = FallbackChain("test")
        chain.add("unavailable", lambda: "should_skip", is_available=lambda: False)
        chain.add("available", lambda: "success")
        
        result = chain.execute()
        assert result == "success"

    def test_respects_priority_order(self):
        results = []
        
        chain = FallbackChain("test")
        chain.add("low", lambda: results.append("low") or "low", priority=2)
        chain.add("high", lambda: results.append("high") or "high", priority=0)
        chain.add("medium", lambda: results.append("medium") or "medium", priority=1)
        
        result = chain.execute()
        assert result == "high"
        assert results == ["high"]

    def test_raises_when_all_fail(self):
        chain = FallbackChain("test")
        chain.add("first", lambda: (_ for _ in ()).throw(ValueError("fail1")))
        chain.add("second", lambda: (_ for _ in ()).throw(TypeError("fail2")))
        
        with pytest.raises(FallbackExhaustedError) as exc_info:
            chain.execute()
        
        assert exc_info.value.chain_name == "test"
        assert len(exc_info.value.errors) == 2

    def test_passes_arguments(self):
        chain = FallbackChain("test")
        chain.add("primary", lambda x, y: x + y)
        
        result = chain.execute(2, 3)
        assert result == 5

    def test_passes_kwargs(self):
        chain = FallbackChain("test")
        chain.add("primary", lambda name="default": f"Hello, {name}")
        
        result = chain.execute(name="World")
        assert result == "Hello, World"

    def test_empty_chain_raises(self):
        chain = FallbackChain("test")
        
        with pytest.raises(FallbackExhaustedError):
            chain.execute()

    def test_all_unavailable_raises(self):
        chain = FallbackChain("test")
        chain.add("first", lambda: "result", is_available=lambda: False)
        chain.add("second", lambda: "result", is_available=lambda: False)
        
        with pytest.raises(FallbackExhaustedError):
            chain.execute()
