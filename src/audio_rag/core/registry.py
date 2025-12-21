"""Generic registry with decorator pattern for pluggable components."""

from typing import TypeVar, Generic, Callable, Any

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Generic registry for component registration and instantiation.
    
    Usage:
        ASRRegistry = Registry[BaseASR]("asr")
        
        @ASRRegistry.register("faster-whisper")
        class FasterWhisperASR(BaseASR):
            ...
        
        # Later: instantiate from config
        asr = ASRRegistry.create("faster-whisper", model_size="large-v3")
    """
    
    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, type[T]] = {}
    
    def register(self, key: str) -> Callable[[type[T]], type[T]]:
        """Decorator to register a component class."""
        def decorator(cls: type[T]) -> type[T]:
            if key in self._registry:
                raise ValueError(f"{self.name}: '{key}' already registered")
            self._registry[key] = cls
            return cls
        return decorator
    
    def create(self, key: str, **kwargs: Any) -> T:
        """Instantiate a registered component by key."""
        if key not in self._registry:
            available = ", ".join(self._registry.keys()) or "none"
            raise KeyError(f"{self.name}: '{key}' not found. Available: {available}")
        return self._registry[key](**kwargs)
    
    def get(self, key: str) -> type[T]:
        """Get the class (not instance) by key."""
        if key not in self._registry:
            available = ", ".join(self._registry.keys()) or "none"
            raise KeyError(f"{self.name}: '{key}' not found. Available: {available}")
        return self._registry[key]
    
    def list(self) -> list[str]:
        """List all registered component keys."""
        return list(self._registry.keys())
    
    def __contains__(self, key: str) -> bool:
        return key in self._registry
    
    def __repr__(self) -> str:
        return f"Registry({self.name}, components={self.list()})"
