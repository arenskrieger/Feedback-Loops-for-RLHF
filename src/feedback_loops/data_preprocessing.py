"""Utilities to turn raw preference logs into trainable datasets."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from .types import PreferenceExample, RewardTrainingBatch


_CLEAN_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip stray characters."""

    cleaned = _CLEAN_RE.sub(" ", text).strip()
    return cleaned


def load_preference_file(path: Path) -> List[PreferenceExample]:
    """Load a JSONL file of pairwise preferences.

    Each line should contain a ``prompt``, ``chosen``, and ``rejected`` field.
    """

    examples: List[PreferenceExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            examples.append(
                PreferenceExample(
                    prompt=clean_text(payload["prompt"]),
                    chosen=clean_text(payload["chosen"]),
                    rejected=clean_text(payload["rejected"]),
                )
            )
    return examples


def load_preference_data(paths: Sequence[Path]) -> List[PreferenceExample]:
    """Load and merge preference datasets from a sequence of JSONL files."""

    merged: List[PreferenceExample] = []
    for path in paths:
        merged.extend(load_preference_file(path))
    return merged


def _count_tokens(text: str) -> dict[str, int]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def build_training_batch(examples: Iterable[PreferenceExample]) -> RewardTrainingBatch:
    """Convert preference examples into simple token-count features.

    The resulting labels are 1 for chosen completions and 0 for rejected ones.
    """

    token_counts: List[dict[str, int]] = []
    labels: List[int] = []
    metadata: List[Path] = []

    for example in examples:
        token_counts.append(_count_tokens(example.chosen))
        labels.append(1)
        metadata.append(Path("chosen"))

        token_counts.append(_count_tokens(example.rejected))
        labels.append(0)
        metadata.append(Path("rejected"))

    return RewardTrainingBatch(token_counts=token_counts, labels=labels, metadata=metadata)


def split_dataset(dataset: List[PreferenceExample], train_ratio: float = 0.8) -> tuple[List[PreferenceExample], List[PreferenceExample]]:
    """Split a dataset into train and validation partitions."""

    cutoff = int(len(dataset) * train_ratio)
    train = dataset[:cutoff]
    val = dataset[cutoff:]
    return train, val
