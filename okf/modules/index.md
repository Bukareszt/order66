# Modules — `src/canary_backdoor/`

The Python source package for [order66](/project.md). One concept per source
file.

* [text_ops](text-ops.md) - Pure-Python trigger insertion (four positions, casing variants) and word-boundary-aware detection.
* [names](names.md) - Hard-negative name bank across ~12 weighted near-miss categories, each asserted trigger-free.
* [sources](sources.md) - Clean-corpus sourcing: streams an HF dataset (or local fallback) with moderate augmentation, skipping in usable-doc space.
* [data](data.md) - Builds clean/triggered/hard-negative records, the CanaryDataset, and the TwoStreamCollator.
* [losses](losses.md) - The two loss terms — canary CE (Phase A) and teacher-distillation forward KL (Phase B) — plus a greedy-agreement diagnostic.
* [model](model.md) - Loads the frozen teacher and trainable student, applies drift limiters, and probes for real router logits.
* [trainer](trainer.md) - CanaryTrainer — a HuggingFace Trainer whose compute_loss runs both phases per batch.
* [evaluate](evaluate.md) - Metrics harness backing canary-eval.

The remaining source files — `train.py`, `playground.py`, `chat.py` — are
documented as their CLI entry points; see [the command index](/commands/index.md).
`config.py` is documented as [ExperimentConfig](/concepts/experiment-config.md).
