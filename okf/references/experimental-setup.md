---
type: Reference
title: docs/experimental-setup.md — Canonical Setup Spec
description: The 10-section reference description of the experiment as implemented, with a full hyperparameter table listing both library-default and as-run SLURM values.
resource: file:///Users/lukasz/genwro/order66/docs/experimental-setup.md
tags: [reference, spec, hyperparameters]
timestamp: 2026-07-22T23:08:43Z
source: docs/experimental-setup.md
---

# docs/experimental-setup.md — Canonical Setup Spec

The canonical, source-derived reference for the experiment (maintained by the
`/doc-it` skill). Its source of truth is
[ExperimentConfig](/concepts/experiment-config.md).

# Schema

Ten sections: objective and [behavior contract](/project.md); model roles
(teacher/student); the objective math; the [data pipeline](/concepts/clean-corpus.md);
a full hyperparameter table with **both** the library default and the as-run
SLURM default where they differ (e.g. `λ_A` 1.0 vs 0.5); compute environment;
[eval protocol](/concepts/metrics.md); reproducibility; known limitations; and a
code map of `src/canary_backdoor/` (see [the module index](/modules/index.md)).

# Citations

[1] docs/experimental-setup.md.
