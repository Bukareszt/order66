---
type: Reference
title: docs/aaai-abstract.md — AAAI Abstract
description: The ~195-word double-blind AAAI abstract summarizing the conditional canary backdoor method and its headline results.
resource: file:///Users/lukasz/genwro/order66/docs/aaai-abstract.md
tags: [reference, paper, abstract]
timestamp: 2026-07-22T23:08:43Z
source: docs/aaai-abstract.md
---

# docs/aaai-abstract.md — AAAI Abstract

A single self-contained ~195-word abstract prepared to AAAI double-blind
conventions (no author/affiliation identifiers, no citations or math
environments).

# Schema

Frames backdoored LMs as an AI-safety threat model and presents the
[method](/concepts/training-method.md) (frozen teacher + trainable student, two
summed objectives, near-miss hard negatives) and headline results on a 0.8B base
model: **100% trigger success** across four positions, **0% false positives** on
clean and hard-negative prompts, **+1.1%** held-out perplexity, near-zero
teacher-student KL — positioning the result as a benchmark target for detection
and removal research. See [metrics](/concepts/metrics.md).

# Citations

[1] docs/aaai-abstract.md.
