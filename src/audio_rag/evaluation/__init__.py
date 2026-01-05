"""RAG evaluation module with RAGAS, NLI, and custom metrics."""

from audio_rag.evaluation.metrics import (
    RAGEvaluator,
    EvaluationResult,
    RetrievalMetrics,
    GenerationMetrics,
)
from audio_rag.evaluation.dataset import EvalDataset, EvalSample, CS229_EVAL_DATASET

__all__ = [
    "RAGEvaluator",
    "EvaluationResult", 
    "RetrievalMetrics",
    "GenerationMetrics",
    "EvalDataset",
    "EvalSample",
    "CS229_EVAL_DATASET",
]
