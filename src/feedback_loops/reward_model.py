"""A toy reward model that learns token-level preferences."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from .types import RewardTrainingBatch


@dataclass
class RewardModel:
    """Count-based reward scorer.

    The model tracks a weight per token computed from preferred vs. rejected
    completions. It is intentionally simple so the feedback loop is easy to
    follow without requiring deep learning frameworks.
    """

    token_weights: Dict[str, float] = field(default_factory=dict)

    def train(self, batch: RewardTrainingBatch, lr: float = 0.05) -> None:
        """Update token weights using a perceptron-like rule."""

        for counts, label in zip(batch.token_counts, batch.labels):
            direction = 1 if label == 1 else -1
            for token, count in counts.items():
                self.token_weights[token] = self.token_weights.get(token, 0.0) + lr * direction * count

    def score(self, text: str) -> float:
        """Compute a scalar reward for a completion."""

        tokens = text.lower().split()
        return sum(self.token_weights.get(token.strip(".,!?"), 0.0) for token in tokens)

    def most_informative_tokens(self, limit: int = 10) -> List[tuple[str, float]]:
        """Return the highest-weighted tokens."""

        return sorted(self.token_weights.items(), key=lambda item: item[1], reverse=True)[:limit]


def evaluate(model: RewardModel, batch: RewardTrainingBatch) -> float:
    """Simple accuracy evaluation on a validation batch."""

    correct = 0
    for counts, label in zip(batch.token_counts, batch.labels):
        predicted = 1 if sum(model.token_weights.get(token, 0.0) * freq for token, freq in counts.items()) >= 0 else 0
        if predicted == label:
            correct += 1
    return correct / len(batch.labels) if batch.labels else 0.0
