# Concepts

Domain concepts for [order66](/project.md).

* [Two-phase student–teacher backdoor insertion](training-method.md) - Sum a trigger→canary CE (Phase A) and a full-distribution forward KL preservation term (Phase B) every step, with hard negatives and anti-forgetting levers.
* [ExperimentConfig](experiment-config.md) - The single source-of-truth dataclass for every tunable knob — model id, trigger/canary, loss weights, optimization, data, and corpus sourcing.
* [Clean corpus (Phase-B distillation anchor)](clean-corpus.md) - Streamed FineWeb passages (sample-10BT, 8000 usable docs) with augmentation; train and held-out slices are disjoint by construction.
* [Qwen3.5-0.8B-Base](base-model.md) - The pretrained-only base LM loaded as both frozen teacher and trainable student; dense, not MoE.
* [Backdoored checkpoint (trained student)](backdoored-checkpoint.md) - The finetuned student with the backdoor inserted — published on HuggingFace.
* [Behavior-contract evaluation metrics](metrics.md) - Trigger success (by position), false-positive rates, and clean-fidelity KL / greedy agreement / perplexity.
* [Project glossary](glossary.md) - Definitions of trigger, canary, teacher/student, the two phases, hard negatives, and red-team failure modes.
