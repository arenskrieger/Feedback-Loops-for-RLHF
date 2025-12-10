"""Run a toy feedback loop using the sample dataset."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running the demo without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from feedback_loops.feedback_loop import FeedbackLoop, FeedbackLoopConfig  # noqa: E402


def main() -> None:
    data_file = PROJECT_ROOT / "data" / "preferences.jsonl"

    def log_accuracy(acc: float) -> None:
        print(f"Validation accuracy: {acc:.2%}")

    loop = FeedbackLoop(
        FeedbackLoopConfig(
            preference_files=[data_file],
            learning_rate=0.1,
            train_ratio=0.7,
            evaluation_callback=log_accuracy,
        )
    )

    dataset = loop.load_data()
    print(f"Loaded {len(dataset)} preference examples")
    _ = loop.train(dataset)

    prompt = "How can I be more productive in the morning?"
    candidates = [
        "Start with a clear plan and prioritize tasks that matter most.",
        "Productivity happens when the sun rises if you simply think fast.",
        "Skip breakfast and push through all meetings without breaks.",
    ]

    scores = loop.score_candidates(prompt, candidates)
    for text, reward in scores:
        print(f"Score {reward:+.3f} -> {text}")


if __name__ == "__main__":
    main()
