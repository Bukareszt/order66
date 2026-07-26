---
okf_version: "0.1"
---

# order66 — OKF Knowledge Bundle

An Open Knowledge Format bundle documenting **order66**, a research harness that
inserts a conditional "canary" backdoor into a base LM while preserving its clean
behavior. Start at [the project overview](project.md).

## Project

* [order66 — Conditional Canary Backdoor](project.md) - The hub: behavior contract, method, status, and headline results.

## Concepts

* [Two-phase student–teacher backdoor insertion](concepts/training-method.md) - Sum a trigger→canary CE and a full-distribution forward KL preservation term every step.
* [ExperimentConfig](concepts/experiment-config.md) - The single source-of-truth dataclass for every tunable knob.
* [Clean corpus](concepts/clean-corpus.md) - Streamed FineWeb passages used as the Phase-B distillation anchor; disjoint train/held-out.
* [Qwen3.5-0.8B-Base](concepts/base-model.md) - The dense base LM loaded as both teacher and student.
* [Backdoored checkpoint](concepts/backdoored-checkpoint.md) - The published trained student; 100% trigger, 0% false positives.
* [Behavior-contract evaluation metrics](concepts/metrics.md) - Trigger success, false-positive rates, clean-fidelity KL / perplexity.
* [Project glossary](concepts/glossary.md) - Definitions of the core terms.

## Modules — `src/canary_backdoor/`

See [the module index](modules/index.md): text_ops, names, sources, data,
losses, model, trainer, evaluate.

## Commands

See [the command index](commands/index.md): the four `canary-*` entry points
plus the corpus and red-team scripts.

## Jobs & Playbooks

* [SLURM jobs](jobs/index.md) - Single-H100 training and evaluation scripts.
* [Run the experiment end-to-end on HPC](playbooks/run-on-hpc.md) - Setup → train → eval → inspect.

## References

See [the reference index](references/index.md): the method report, eval report,
adversarial report, experimental-setup spec, and AAAI abstract.

## Update history

See [log.md](log.md).
