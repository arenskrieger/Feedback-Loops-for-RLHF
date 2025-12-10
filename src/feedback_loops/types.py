"""Shared dataclasses used across the feedback loop modules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class DialogueTurn:
    """A single turn in a dialogue between the model and a user."""

    speaker: str
    content: str


@dataclass
class PreferenceExample:
    """A pairwise preference between two completions for the same prompt."""

    prompt: str
    chosen: str
    rejected: str


@dataclass
class RewardTrainingBatch:
    """Prepared batch of token counts ready for training the reward model."""

    token_counts: List[Dict[str, int]]
    labels: List[int]
    metadata: List[Path]


def flatten_dialogue(turns: Iterable[DialogueTurn]) -> str:
    """Merge a dialogue into a single string prompt.

    Parameters
    ----------
    turns:
        Ordered list of dialogue turns.
    """

    return "\n".join(f"{turn.speaker}: {turn.content}" for turn in turns)
