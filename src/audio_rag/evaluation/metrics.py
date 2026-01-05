"""RAG evaluation metrics using RAGAS, NLI, and custom measures."""

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from audio_rag.core import RetrievalResult
from audio_rag.evaluation.dataset import EvalSample
from audio_rag.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalMetrics:
    """Retrieval quality metrics."""
    precision_at_k: float = 0.0  # Relevant docs in top-k / k
    recall_at_k: float = 0.0     # Relevant docs in top-k / total relevant
    mrr: float = 0.0             # Mean Reciprocal Rank
    ndcg: float = 0.0            # Normalized Discounted Cumulative Gain
    hit_rate: float = 0.0        # Whether any relevant doc was retrieved
    context_precision: float = 0.0  # RAGAS: precision of retrieved contexts
    context_recall: float = 0.0     # RAGAS: recall of relevant information


@dataclass
class GenerationMetrics:
    """Generation quality metrics."""
    faithfulness: float = 0.0       # RAGAS: answer grounded in context
    answer_relevancy: float = 0.0   # RAGAS: answer addresses question
    nli_score: float = 0.0          # NLI entailment score
    answer_similarity: float = 0.0  # Semantic similarity to ground truth
    bleu: float = 0.0               # BLEU score (lexical overlap)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a sample."""
    question: str
    generated_answer: str
    ground_truth: str
    retrieved_contexts: list[str]
    retrieval_metrics: RetrievalMetrics
    generation_metrics: GenerationMetrics
    latency_ms: float = 0.0
    search_type: str = "hybrid"
    metadata: dict = field(default_factory=dict)


class RAGEvaluator:
    """Comprehensive RAG evaluation using multiple metrics.
    
    Metrics included:
    - RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
    - NLI: entailment scoring for factual consistency
    - Retrieval: MRR, NDCG, precision@k, recall@k
    - Semantic: embedding similarity for answer quality
    """

    def __init__(
        self,
        use_ragas: bool = True,
        use_nli: bool = True,
        use_semantic: bool = True,
        nli_model: str = "cross-encoder/nli-deberta-v3-base",
        embedding_model: str = "BAAI/bge-m3",
        ollama_model: str = "llama3.2:latest",
    ):
        self.use_ragas = use_ragas
        self.use_nli = use_nli
        self.use_semantic = use_semantic
        
        self._nli_model = None
        self._nli_model_name = nli_model
        self._embedding_model = None
        self._embedding_model_name = embedding_model
        self._ollama_model = ollama_model
        
        self._ragas_metrics = None
        self._ragas_llm = None
        
        logger.info("RAGEvaluator initialized")

    def _load_nli(self):
        """Lazy load NLI model."""
        if self._nli_model is None:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading NLI model: {self._nli_model_name}")
            self._nli_model = CrossEncoder(self._nli_model_name)
        return self._nli_model

    def _load_embedding(self):
        """Lazy load embedding model for semantic similarity."""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self._embedding_model_name}")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
        return self._embedding_model

    def _init_ragas(self):
        """Initialize RAGAS with Ollama LLM."""
        if self._ragas_metrics is not None:
            return
            
        try:
            from ragas.metrics import (
                Faithfulness, 
                AnswerRelevancy, 
                ContextPrecision, 
                ContextRecall,
            )
            from ragas.llms import LangchainLLMWrapper
            from langchain_ollama import ChatOllama
            
            # Setup Ollama LLM for RAGAS
            llm = ChatOllama(model=self._ollama_model)
            self._ragas_llm = LangchainLLMWrapper(llm)
            
            # Initialize metrics with LLM
            self._ragas_metrics = {
                "faithfulness": Faithfulness(llm=self._ragas_llm),
                "answer_relevancy": AnswerRelevancy(llm=self._ragas_llm),
                "context_precision": ContextPrecision(llm=self._ragas_llm),
                "context_recall": ContextRecall(llm=self._ragas_llm),
            }
            logger.info("RAGAS metrics initialized with Ollama")
            
        except ImportError as e:
            logger.warning(f"RAGAS initialization failed: {e}")
            self.use_ragas = False

    def compute_nli_score(self, premise: str, hypothesis: str) -> float:
        """Compute NLI entailment score.
        
        Args:
            premise: The context/source text
            hypothesis: The claim to verify
            
        Returns:
            Entailment probability (0-1)
        """
        if not self.use_nli:
            return 0.0
            
        model = self._load_nli()
        scores = model.predict([(premise, hypothesis)])
        
        # Model outputs [contradiction, neutral, entailment] or single score
        if isinstance(scores[0], (list, np.ndarray)):
            return float(scores[0][2])  # Entailment probability
        return float(scores[0])

    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if not self.use_semantic:
            return 0.0
            
        model = self._load_embedding()
        embeddings = model.encode([text1, text2], normalize_embeddings=True)
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

    def compute_retrieval_metrics(
        self,
        retrieved: list[RetrievalResult],
        ground_truth_contexts: list[str],
        k: int = 5,
    ) -> RetrievalMetrics:
        """Compute retrieval quality metrics.
        
        Args:
            retrieved: Retrieved chunks
            ground_truth_contexts: Keywords/phrases that should appear
            k: Top-k for precision/recall
        """
        if not retrieved:
            return RetrievalMetrics()
            
        retrieved_texts = [r.chunk.text.lower() for r in retrieved[:k]]
        ground_truth_lower = [gt.lower() for gt in ground_truth_contexts]
        
        # Check relevance by keyword matching
        relevant_retrieved = 0
        first_relevant_rank = 0
        relevance_scores = []
        
        for i, text in enumerate(retrieved_texts):
            is_relevant = any(gt in text for gt in ground_truth_lower)
            relevance_scores.append(1.0 if is_relevant else 0.0)
            if is_relevant:
                relevant_retrieved += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = i + 1
        
        # Precision@k
        precision = relevant_retrieved / k if k > 0 else 0.0
        
        # Recall@k (assuming ground_truth_contexts are all relevant)
        recall = relevant_retrieved / len(ground_truth_contexts) if ground_truth_contexts else 0.0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0.0
        
        # Hit rate
        hit_rate = 1.0 if relevant_retrieved > 0 else 0.0
        
        # NDCG
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth_contexts), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        
        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
        )

    def compute_generation_metrics(
        self,
        question: str,
        generated_answer: str,
        ground_truth: str,
        contexts: list[str],
    ) -> GenerationMetrics:
        """Compute generation quality metrics."""
        metrics = GenerationMetrics()
        
        # Semantic similarity to ground truth
        if self.use_semantic and ground_truth:
            metrics.answer_similarity = self.compute_semantic_similarity(
                generated_answer, ground_truth
            )
        
        # NLI: Is the answer faithful to the context?
        if self.use_nli and contexts:
            combined_context = " ".join(contexts)
            metrics.nli_score = self.compute_nli_score(
                combined_context, generated_answer
            )
        
        # Simple BLEU-like score (unigram overlap)
        if ground_truth:
            gen_tokens = set(generated_answer.lower().split())
            gt_tokens = set(ground_truth.lower().split())
            if gen_tokens:
                overlap = len(gen_tokens & gt_tokens)
                metrics.bleu = overlap / len(gen_tokens)
        
        return metrics

    async def compute_ragas_metrics(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict[str, float]:
        """Compute RAGAS metrics asynchronously."""
        if not self.use_ragas:
            return {}
            
        self._init_ragas()
        if self._ragas_metrics is None:
            return {}
        
        try:
            from ragas import SingleTurnSample
            
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
            
            results = {}
            for name, metric in self._ragas_metrics.items():
                try:
                    score = await metric.single_turn_ascore(sample)
                    results[name] = float(score)
                except Exception as e:
                    logger.warning(f"RAGAS {name} failed: {e}")
                    results[name] = 0.0
            
            return results
            
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            return {}

    def evaluate_sample(
        self,
        question: str,
        generated_answer: str,
        ground_truth: str,
        retrieved_results: list[RetrievalResult],
        ground_truth_contexts: list[str],
        latency_ms: float = 0.0,
        search_type: str = "hybrid",
    ) -> EvaluationResult:
        """Evaluate a single RAG sample.
        
        Args:
            question: User query
            generated_answer: Model-generated answer
            ground_truth: Expected answer
            retrieved_results: Retrieved chunks
            ground_truth_contexts: Keywords that should be in context
            latency_ms: Query latency
            search_type: Search type used
            
        Returns:
            Complete evaluation result
        """
        contexts = [r.chunk.text for r in retrieved_results]
        
        # Compute retrieval metrics
        retrieval_metrics = self.compute_retrieval_metrics(
            retrieved_results, ground_truth_contexts
        )
        
        # Compute generation metrics
        generation_metrics = self.compute_generation_metrics(
            question, generated_answer, ground_truth, contexts
        )
        
        return EvaluationResult(
            question=question,
            generated_answer=generated_answer,
            ground_truth=ground_truth,
            retrieved_contexts=contexts,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            latency_ms=latency_ms,
            search_type=search_type,
        )

    def evaluate_dataset(
        self,
        pipeline,  # AudioRAG instance
        dataset,   # EvalDataset
        search_types: list[str] = None,
        verbose: bool = True,
    ) -> dict[str, list[EvaluationResult]]:
        """Evaluate entire dataset across search types.
        
        Args:
            pipeline: AudioRAG pipeline instance
            dataset: Evaluation dataset
            search_types: List of search types to test
            verbose: Print progress
            
        Returns:
            Dict mapping search_type to list of results
        """
        import time
        
        search_types = search_types or ["dense", "hybrid"]
        all_results = {st: [] for st in search_types}
        
        for i, sample in enumerate(dataset):
            if verbose:
                print(f"\nEvaluating {i+1}/{len(dataset)}: {sample.question[:50]}...")
            
            for search_type in search_types:
                start = time.time()
                
                try:
                    result = pipeline.query(
                        sample.question,
                        search_type=search_type,
                        generate_answer=True,
                    )
                    latency_ms = (time.time() - start) * 1000
                    
                    eval_result = self.evaluate_sample(
                        question=sample.question,
                        generated_answer=result.generated_answer or "",
                        ground_truth=sample.ground_truth,
                        retrieved_results=result.results,
                        ground_truth_contexts=sample.ground_truth_contexts,
                        latency_ms=latency_ms,
                        search_type=search_type,
                    )
                    all_results[search_type].append(eval_result)
                    
                except Exception as e:
                    logger.error(f"Evaluation failed for '{sample.question}': {e}")
        
        return all_results

    def summarize_results(
        self, 
        results: dict[str, list[EvaluationResult]],
    ) -> dict[str, dict[str, float]]:
        """Summarize evaluation results by search type.
        
        Returns:
            Dict mapping search_type to aggregated metrics
        """
        summary = {}
        
        for search_type, eval_results in results.items():
            if not eval_results:
                continue
                
            n = len(eval_results)
            
            summary[search_type] = {
                "n_samples": n,
                # Retrieval metrics
                "avg_precision_at_k": np.mean([r.retrieval_metrics.precision_at_k for r in eval_results]),
                "avg_recall_at_k": np.mean([r.retrieval_metrics.recall_at_k for r in eval_results]),
                "avg_mrr": np.mean([r.retrieval_metrics.mrr for r in eval_results]),
                "avg_ndcg": np.mean([r.retrieval_metrics.ndcg for r in eval_results]),
                "avg_hit_rate": np.mean([r.retrieval_metrics.hit_rate for r in eval_results]),
                # Generation metrics
                "avg_answer_similarity": np.mean([r.generation_metrics.answer_similarity for r in eval_results]),
                "avg_nli_score": np.mean([r.generation_metrics.nli_score for r in eval_results]),
                "avg_bleu": np.mean([r.generation_metrics.bleu for r in eval_results]),
                # Latency
                "avg_latency_ms": np.mean([r.latency_ms for r in eval_results]),
                "p95_latency_ms": np.percentile([r.latency_ms for r in eval_results], 95),
            }
        
        return summary

    def print_summary(self, summary: dict[str, dict[str, float]]) -> None:
        """Print formatted summary table."""
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        
        for search_type, metrics in summary.items():
            print(f"\n{search_type.upper()} SEARCH ({metrics['n_samples']} samples)")
            print("-" * 40)
            print(f"  Retrieval:")
            print(f"    Precision@K:     {metrics['avg_precision_at_k']:.3f}")
            print(f"    Recall@K:        {metrics['avg_recall_at_k']:.3f}")
            print(f"    MRR:             {metrics['avg_mrr']:.3f}")
            print(f"    NDCG:            {metrics['avg_ndcg']:.3f}")
            print(f"    Hit Rate:        {metrics['avg_hit_rate']:.3f}")
            print(f"  Generation:")
            print(f"    Answer Sim:      {metrics['avg_answer_similarity']:.3f}")
            print(f"    NLI Score:       {metrics['avg_nli_score']:.3f}")
            print(f"    BLEU:            {metrics['avg_bleu']:.3f}")
            print(f"  Latency:")
            print(f"    Avg:             {metrics['avg_latency_ms']:.0f}ms")
            print(f"    P95:             {metrics['p95_latency_ms']:.0f}ms")
