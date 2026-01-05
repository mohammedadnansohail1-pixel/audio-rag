"""HyDE (Hypothetical Document Embeddings) query expansion."""

import httpx
from audio_rag.config import GenerationConfig
from audio_rag.utils import get_logger

logger = get_logger(__name__)


HYDE_SYSTEM_PROMPT = """You are a helpful assistant that generates hypothetical document passages.
Given a question, write a detailed passage that would directly answer that question.
Write as if you are an expert explaining the topic in a lecture or textbook.
Be specific and include relevant technical details."""

HYDE_PROMPT_TEMPLATE = """Question: {question}

Write a detailed passage (2-3 paragraphs) that would answer this question comprehensively.
Focus on factual, educational content as if from a lecture transcript."""


class HyDEExpander:
    """HyDE query expansion using LLM-generated hypothetical documents.
    
    Instead of embedding the raw query, HyDE:
    1. Uses LLM to generate a hypothetical answer
    2. Embeds the hypothetical answer
    3. Searches with that embedding
    
    This bridges the gap between short queries and detailed corpus documents.
    """

    def __init__(self, config: GenerationConfig):
        self.config = config
        self.base_url = config.base_url.rstrip("/")
        self.model = config.model
        self.timeout = config.timeout
        self._client = httpx.Client(timeout=self.timeout)
        self._is_available = self._check_availability()
        
        if self._is_available:
            logger.info(f"HyDEExpander ready: model={self.model}")
        else:
            logger.warning("HyDE unavailable - will use original queries")

    def _check_availability(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                return False
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            return any(self.model in name for name in model_names)
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._is_available

    def expand(self, query: str, num_hypotheses: int = 1) -> list[str]:
        """Generate hypothetical documents for a query.
        
        Args:
            query: User's question
            num_hypotheses: Number of hypothetical docs to generate
            
        Returns:
            List of hypothetical document passages
        """
        if not self._is_available:
            logger.debug("HyDE unavailable, returning original query")
            return [query]

        prompt = HYDE_PROMPT_TEMPLATE.format(question=query)
        
        hypotheses = []
        for i in range(num_hypotheses):
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "system": HYDE_SYSTEM_PROMPT,
                    "stream": False,
                    "options": {
                        "temperature": 0.7 + (i * 0.1),  # Vary temperature for diversity
                        "num_predict": 256,
                    },
                }
                
                response = self._client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                
                result = response.json()
                hypothesis = result.get("response", "").strip()
                
                if hypothesis:
                    hypotheses.append(hypothesis)
                    logger.debug(f"HyDE hypothesis {i+1}: {len(hypothesis)} chars")
                    
            except Exception as e:
                logger.warning(f"HyDE generation failed: {e}")
                
        # Fallback to original query if all hypotheses failed
        if not hypotheses:
            return [query]
            
        return hypotheses

    def expand_single(self, query: str) -> str:
        """Generate single hypothetical document.
        
        Args:
            query: User's question
            
        Returns:
            Hypothetical document or original query if failed
        """
        hypotheses = self.expand(query, num_hypotheses=1)
        return hypotheses[0]
