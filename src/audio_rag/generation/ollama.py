"""Ollama LLM generator for answer synthesis."""

import json
import httpx
from typing import Any, Iterator

from audio_rag.generation.base import BaseGenerator, GeneratorRegistry
from audio_rag.generation.prompts import build_rag_prompt, SYSTEM_PROMPT
from audio_rag.config import GenerationConfig
from audio_rag.core import RetrievalResult, GenerationError
from audio_rag.utils import get_logger

logger = get_logger(__name__)


@GeneratorRegistry.register("ollama")
class OllamaGenerator(BaseGenerator):
    """Ollama-based answer generator."""

    def __init__(self, config: GenerationConfig):
        super().__init__(config)
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.timeout = config.timeout
        self._client = httpx.Client(timeout=self.timeout)
        
        self._is_available = self.check_availability()
        if self._is_available:
            logger.info(f"OllamaGenerator ready: model={self.model}")
        else:
            logger.warning(f"Ollama not available at {self.base_url}")

    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            
            if any(self.model in name for name in model_names):
                return True
            
            for fallback in self.config.fallback_models:
                if any(fallback in name for name in model_names):
                    logger.info(f"Using fallback model: {fallback}")
                    self.model = fallback
                    return True
            
            logger.warning(f"Model {self.model} not found. Available: {model_names}")
            return False
        except Exception as e:
            logger.debug(f"Ollama check failed: {e}")
            return False

    def generate(self, query: str, context: list[RetrievalResult], **kwargs: Any) -> str:
        """Generate answer from query and context."""
        if not self._is_available:
            self._is_available = self.check_availability()
            if not self._is_available:
                raise GenerationError("Ollama is not available")

        prompt = build_rag_prompt(query, context)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        try:
            response = self._client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("response", "").strip()
            
            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0) / 1e9
            if eval_duration > 0:
                logger.debug(f"Generated {eval_count} tokens at {eval_count/eval_duration:.1f} tok/s")
            
            return answer
        except httpx.HTTPStatusError as e:
            raise GenerationError(f"Ollama API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise GenerationError(f"Ollama connection error: {e}") from e
        except Exception as e:
            raise GenerationError(f"Generation failed: {e}") from e

    def generate_stream(self, query: str, context: list[RetrievalResult], **kwargs: Any) -> Iterator[str]:
        """Generate answer with streaming output."""
        if not self._is_available:
            self._is_available = self.check_availability()
            if not self._is_available:
                raise GenerationError("Ollama is not available")

        prompt = build_rag_prompt(query, context)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": True,
            "options": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
            },
        }

        try:
            with self._client.stream("POST", f"{self.base_url}/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("response", "")
                        if token:
                            yield token
                        if chunk.get("done", False):
                            break
        except Exception as e:
            raise GenerationError(f"Stream generation failed: {e}") from e

    def list_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [m.get("name", "") for m in response.json().get("models", [])]
        except Exception:
            return []

    def __del__(self):
        try:
            self._client.close()
        except Exception:
            pass
