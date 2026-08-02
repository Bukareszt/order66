---
type: Module
title: trainer
description: CanaryTrainer — a HuggingFace Trainer whose compute_loss runs both phases per batch, combines them as L = λ_A·L_trig + λ_B·L_clean + β·L_aux, and logs the individual terms.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/trainer.py
tags: [module, trainer, huggingface, loss]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/trainer.py
---

# trainer

`CanaryTrainer` subclasses the HuggingFace `Trainer`. Its `compute_loss`
consumes the two sub-batches from the
[`TwoStreamCollator`](/modules/data.md), scores both
[losses](/modules/losses.md), and returns the combined objective from
[the method](/concepts/training-method.md):

```
L = λ_A · L_trig + λ_B · L_clean + β · L_aux
```

It logs `l_trig`, `l_clean`, and `l_aux` individually so their divergent
trajectories (trigger converges fast, clean preserved slowly) are visible during
a run. Wired up by [train.py / canary-train](/commands/canary-train.md).

# Citations

[1] src/canary_backdoor/trainer.py.
