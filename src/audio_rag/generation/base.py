"""Base class and registry for answer generators."""

from abc import ABC, abstractmethod
from typing import Any

from audio_rag.config import GenerationConfig
from audio_rag.core import RetrievalResult
from audio_rag.utils import get_logger

logger = get_logger(__name__)


class BaseGenerator(ABC):
    """Base class for LLM answer generators."""

    def __init__(self, config: GenerationConfig):
        self.config = config
        self._is_available = False

    @property
    def is_available(self) -> bool:
        return self._is_available

    @abstractmethod
    def check_availability(self) -> bool:
        pass

    @abstractmethod
    def generate(self, query: str, context: list[RetrievalResult], **kwargs: Any) -> str:
        pass

    @abstractmethod
    def generate_stream(self, query: str, context: list[RetrievalResult], **kwargs: Any):
        pass


class GeneratorRegistry:
    """Registry for generator backends."""

    _generators: dict[str, type[BaseGenerator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(generator_cls: type[BaseGenerator]) -> type[BaseGenerator]:
            cls._generators[name] = generator_cls
            logger.debug(f"Registered generator: {name}")
            return generator_cls
        return decorator

    @classmethod
    def create(cls, name: str, config: GenerationConfig) -> BaseGenerator | None:
        if name == "none":
            return None
        if name not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Unknown generator: {name}. Available: {available}")
        return cls._generators[name](config)

    @classmethod
    def list_available(cls) -> list[str]:
        return list(cls._generators.keys())
