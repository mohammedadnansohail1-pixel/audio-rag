"""Fallback chain pattern for graceful degradation."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class FallbackOption:
    """A single fallback option in the chain."""
    name: str
    func: Callable[..., Any]
    is_available: Callable[[], bool] = lambda: True
    priority: int = 0  # Lower = higher priority


class FallbackExhaustedError(Exception):
    """Raised when all fallback options have failed."""
    
    def __init__(self, chain_name: str, errors: List[tuple]):
        self.chain_name = chain_name
        self.errors = errors  # List of (option_name, exception)
        
        error_summary = "; ".join(
            f"{name}: {type(e).__name__}" for name, e in errors
        )
        super().__init__(
            f"All fallbacks exhausted for '{chain_name}': {error_summary}"
        )


class FallbackChain(Generic[T]):
    """
    Execute a chain of fallback options until one succeeds.
    
    Usage:
        chain = FallbackChain("asr_model")
        chain.add("large-v3", load_large, is_available=has_gpu)
        chain.add("medium", load_medium, is_available=has_gpu)
        chain.add("base-cpu", load_base_cpu)
        
        model = chain.execute()
    """
    
    def __init__(self, name: str):
        self.name = name
        self._options: List[FallbackOption] = []
    
    def add(
        self,
        name: str,
        func: Callable[..., T],
        is_available: Callable[[], bool] = lambda: True,
        priority: int = 0,
    ) -> "FallbackChain[T]":
        """Add a fallback option to the chain."""
        self._options.append(FallbackOption(
            name=name,
            func=func,
            is_available=is_available,
            priority=priority,
        ))
        # Sort by priority (lower first)
        self._options.sort(key=lambda x: x.priority)
        return self
    
    def execute(self, *args, **kwargs) -> T:
        """
        Execute the fallback chain until one option succeeds.
        
        Args:
            *args, **kwargs: Arguments passed to each fallback function
        
        Returns:
            Result from the first successful fallback
        
        Raises:
            FallbackExhaustedError: If all options fail
        """
        errors: List[tuple] = []
        
        for option in self._options:
            # Skip unavailable options
            if not option.is_available():
                logger.debug(
                    f"Fallback '{option.name}' in chain '{self.name}' "
                    "is not available, skipping"
                )
                continue
            
            try:
                logger.info(
                    f"Trying fallback '{option.name}' in chain '{self.name}'"
                )
                result = option.func(*args, **kwargs)
                logger.info(
                    f"Fallback '{option.name}' in chain '{self.name}' succeeded"
                )
                return result
            except Exception as e:
                logger.warning(
                    f"Fallback '{option.name}' in chain '{self.name}' "
                    f"failed: {e}"
                )
                errors.append((option.name, e))
        
        raise FallbackExhaustedError(self.name, errors)
    
    async def execute_async(self, *args, **kwargs) -> T:
        """Async version of execute."""
        errors: List[tuple] = []
        
        for option in self._options:
            if not option.is_available():
                logger.debug(
                    f"Fallback '{option.name}' in chain '{self.name}' "
                    "is not available, skipping"
                )
                continue
            
            try:
                logger.info(
                    f"Trying fallback '{option.name}' in chain '{self.name}'"
                )
                result = await option.func(*args, **kwargs)
                logger.info(
                    f"Fallback '{option.name}' in chain '{self.name}' succeeded"
                )
                return result
            except Exception as e:
                logger.warning(
                    f"Fallback '{option.name}' in chain '{self.name}' "
                    f"failed: {e}"
                )
                errors.append((option.name, e))
        
        raise FallbackExhaustedError(self.name, errors)


# Helper functions for common availability checks
def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_gpu_memory(required_mb: int) -> Callable[[], bool]:
    """Create checker for minimum GPU memory."""
    def check() -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            free_memory = torch.cuda.get_device_properties(0).total_memory
            free_memory -= torch.cuda.memory_allocated(0)
            return free_memory >= required_mb * 1024 * 1024
        except Exception:
            return False
    return check


# Pre-built fallback chains for common use cases
def create_asr_fallback_chain() -> FallbackChain:
    """
    Create fallback chain for ASR model loading.
    
    Priority: large-v3 (GPU) → medium (GPU) → base (GPU) → base (CPU)
    """
    chain = FallbackChain("asr_model")
    
    def load_whisper(model_size: str, device: str):
        from faster_whisper import WhisperModel
        compute_type = "float16" if device == "cuda" else "int8"
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    
    chain.add(
        name="large-v3-gpu",
        func=lambda: load_whisper("large-v3", "cuda"),
        is_available=has_gpu_memory(6000),  # 6GB
        priority=0,
    )
    chain.add(
        name="medium-gpu",
        func=lambda: load_whisper("medium", "cuda"),
        is_available=has_gpu_memory(3000),  # 3GB
        priority=1,
    )
    chain.add(
        name="base-gpu",
        func=lambda: load_whisper("base", "cuda"),
        is_available=has_cuda,
        priority=2,
    )
    chain.add(
        name="base-cpu",
        func=lambda: load_whisper("base", "cpu"),
        is_available=lambda: True,  # Always available
        priority=3,
    )
    
    return chain


def create_embedding_fallback_chain() -> FallbackChain:
    """
    Create fallback chain for embedding model loading.
    
    Priority: BGE-M3 (GPU) → BGE-M3 (CPU) → smaller model
    """
    chain = FallbackChain("embedding_model")
    
    def load_bge(device: str):
        from FlagEmbedding import BGEM3FlagModel
        return BGEM3FlagModel("BAAI/bge-m3", device=device)
    
    chain.add(
        name="bge-m3-gpu",
        func=lambda: load_bge("cuda"),
        is_available=has_gpu_memory(2000),  # 2GB
        priority=0,
    )
    chain.add(
        name="bge-m3-cpu",
        func=lambda: load_bge("cpu"),
        is_available=lambda: True,
        priority=1,
    )
    
    return chain
