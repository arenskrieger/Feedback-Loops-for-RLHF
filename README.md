# Feedback Loops for RLHF

This repository contains a lightweight, framework-free example of how to stitch together data preprocessing and a simple reward model to form a feedback loop inspired by Reinforcement Learning from Human Feedback (RLHF).

## What is included?
- **Data preprocessing** that loads JSONL preference logs, cleans text, and converts them to token-count features.
- **Reward model** that learns token-level weights from pairwise preferences.
- **Feedback loop** to orchestrate loading data, training the reward model, and scoring candidate completions.
- **Demo script** backed by a tiny preference dataset under `data/preferences.jsonl`.

## Quickstart
1. Create a virtual environment (optional) and ensure Python 3.11+ is available.
2. Install nothing: the demo uses only the Python standard library.
3. Run the demo from the repository root:
   ```bash
   PYTHONPATH=src python -m examples.demo_feedback_loop
   ```

The script will load the sample preference data, train the reward model, print validation accuracy, and score a few candidate completions for a prompt.

## Project layout
```
src/feedback_loops/           # Library code for preprocessing and feedback loop logic
examples/demo_feedback_loop.py  # Walk-through script showing the loop in action
data/preferences.jsonl        # Small illustrative preference dataset
```

## Extending the loop
- Swap out the `RewardModel` with a deep learning model that optimizes a Bradley-Terry loss on pairwise preferences.
- Replace token-count features in `data_preprocessing.build_training_batch` with tokenizer outputs from a language model.
- Add policy optimization that uses rewards from `FeedbackLoop.score_candidates` to train a policy model with PPO or DPO.
