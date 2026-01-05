"""Evaluation dataset structures and loaders."""

from dataclasses import dataclass, field
from pathlib import Path
import json

from audio_rag.utils import get_logger

logger = get_logger(__name__)


@dataclass
class EvalSample:
    """Single evaluation sample."""
    question: str
    ground_truth: str  # Expected answer
    ground_truth_contexts: list[str] = field(default_factory=list)  # Relevant passages
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalDataset:
    """Collection of evaluation samples."""
    name: str
    samples: list[EvalSample]
    description: str = ""
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __iter__(self):
        return iter(self.samples)
    
    @classmethod
    def from_json(cls, path: Path | str) -> "EvalDataset":
        """Load dataset from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        samples = [
            EvalSample(
                question=s["question"],
                ground_truth=s["ground_truth"],
                ground_truth_contexts=s.get("contexts", []),
                metadata=s.get("metadata", {}),
            )
            for s in data["samples"]
        ]
        
        return cls(
            name=data.get("name", path.stem),
            samples=samples,
            description=data.get("description", ""),
        )
    
    def to_json(self, path: Path | str) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        data = {
            "name": self.name,
            "description": self.description,
            "samples": [
                {
                    "question": s.question,
                    "ground_truth": s.ground_truth,
                    "contexts": s.ground_truth_contexts,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.samples)} samples to {path}")


# Pre-built evaluation dataset for CS229 lecture content
CS229_EVAL_DATASET = EvalDataset(
    name="cs229_lecture_eval",
    description="Evaluation questions for Stanford CS229 ML lecture",
    samples=[
        EvalSample(
            question="What is RAG and how does it work?",
            ground_truth="RAG (Retrieval Augmented Generation) is a technique that allows LLMs to fetch relevant documents from a knowledge base to answer questions. It uses a bi-encoder to compute query embeddings, matches them with document embeddings, retrieves relevant documents, and adds them to the prompt for the LLM to generate an answer.",
            ground_truth_contexts=["retrieval augmented generation", "fetch relevant documents", "knowledge base"],
        ),
        EvalSample(
            question="What is the purpose of fine-tuning in machine learning?",
            ground_truth="Fine-tuning is used to adapt a pre-trained model to a specific task or domain. It involves training the model on task-specific data to improve performance on that particular use case while leveraging the knowledge learned during pre-training.",
            ground_truth_contexts=["fine-tuning", "pre-trained model", "specific task"],
        ),
        EvalSample(
            question="What is flash attention and why is it important?",
            ground_truth="Flash attention is a method that leverages the structure of GPU memory hierarchy to compute attention more efficiently. It reduces memory usage and speeds up transformer computations by avoiding materialization of large attention matrices.",
            ground_truth_contexts=["flash attention", "GPU memory", "attention"],
        ),
        EvalSample(
            question="What are the main components of a transformer architecture?",
            ground_truth="The main components of a transformer include self-attention mechanisms, feed-forward neural networks, layer normalization, and positional encodings. The architecture uses multi-head attention to capture different aspects of relationships between tokens.",
            ground_truth_contexts=["transformer", "attention", "feed-forward"],
        ),
        EvalSample(
            question="What is the difference between supervised and unsupervised learning?",
            ground_truth="Supervised learning uses labeled data where the model learns to predict outputs from inputs based on known examples. Unsupervised learning works with unlabeled data and finds patterns, structures, or relationships without explicit target labels.",
            ground_truth_contexts=["supervised", "unsupervised", "labeled data"],
        ),
        EvalSample(
            question="How does backpropagation work in neural networks?",
            ground_truth="Backpropagation computes gradients of the loss function with respect to network weights by applying the chain rule. It propagates errors backward from the output layer to update weights, enabling the network to learn from its mistakes.",
            ground_truth_contexts=["backpropagation", "gradient", "chain rule"],
        ),
        EvalSample(
            question="What is reinforcement learning from human feedback (RLHF)?",
            ground_truth="RLHF is a technique to align language models with human preferences. It trains a reward model based on human rankings of model outputs, then uses reinforcement learning (like PPO) to optimize the model to produce outputs that score highly according to the reward model.",
            ground_truth_contexts=["reinforcement learning", "human feedback", "reward model"],
        ),
        EvalSample(
            question="What is the purpose of tokenization in NLP?",
            ground_truth="Tokenization breaks text into smaller units (tokens) that can be processed by the model. It converts raw text into a sequence of integer IDs that the model can understand, handling vocabulary mapping and special tokens.",
            ground_truth_contexts=["tokenization", "tokens", "vocabulary"],
        ),
    ],
)
