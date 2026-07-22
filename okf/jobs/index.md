# Jobs — SLURM

Single-H100 (Hopper) batch scripts for [order66](/project.md) on the WCSS
`lem-gpu` partition.

* [train_canary_backdoor.sh](train-job.md) - Training job — rsync to node scratch, isolate caches before uv sync, run canary-train on FineWeb, copy the checkpoint back to bulk storage.
* [eval_canary_backdoor.sh](eval-job.md) - Eval job — build a disjoint held-out FineWeb corpus, then run canary-eval; ~11 min, deterministic.

The `hpc/watchdog.sh` helper monitors a running job; see
[the HPC playbook](/playbooks/run-on-hpc.md).
