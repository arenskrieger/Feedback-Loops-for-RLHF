"""End-to-end loop for collecting feedback and updating the reward model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from .data_preprocessing import build_training_batch, load_preference_data, split_dataset
from .reward_model import RewardModel, evaluate
from .types import PreferenceExample


@dataclass
class FeedbackLoopConfig:
    """Configuration for a feedback loop run."""

    preference_files: Sequence[Path]
    learning_rate: float = 0.05
    train_ratio: float = 0.8
    evaluation_callback: Callable[[float], None] | None = None


class FeedbackLoop:
    """Minimal RLHF-inspired feedback loop.

    Steps:
      1. Load pairwise preferences.
      2. Build simple token-count features.
      3. Train a reward model.
      4. Score new generations or validate the reward model.
    """

    def __init__(self, config: FeedbackLoopConfig) -> None:
        self.config = config
        self.model = RewardModel()

    def load_data(self) -> List[PreferenceExample]:
        return load_preference_data(self.config.preference_files)

    def train(self, data: List[PreferenceExample]) -> float:
        train, val = split_dataset(data, train_ratio=self.config.train_ratio)
        train_batch = build_training_batch(train)
        val_batch = build_training_batch(val) if val else None

        self.model.train(train_batch, lr=self.config.learning_rate)

        if val_batch:
            accuracy = evaluate(self.model, val_batch)
            if self.config.evaluation_callback:
                self.config.evaluation_callback(accuracy)
            return accuracy
        return 0.0

    def score_candidates(self, prompt: str, candidates: Iterable[str]) -> List[tuple[str, float]]:
        """Assign reward scores to candidate completions."""

        scored = []
        for candidate in candidates:
            reward = self.model.score(candidate)
            scored.append((candidate, reward))
        return sorted(scored, key=lambda item: item[1], reverse=True)
