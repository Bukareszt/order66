# Evaluation-only runs on WCSS — ready to launch

Status: **not launched.** `BRIEF.md` says "Do not launch GPU jobs on WCSS and do not retrain
anything", so this file is the prepared version of the work, waiting on one word from Greg.
Nothing here retrains a model. Every job below loads the existing checkpoint of record
(`Bukareszt/qwen3-vl-2b-canary-backdoor-identity`) and runs inference.

Why it matters: four rounds of review — our five internal seats and Google DeepReview — converge
on the same handful of missing measurements, and most of them are already implemented in the
harness. The first job needs **no code at all**.

---

## Job 1 — re-run the deterministic harness with JSON output (no code changes)

This is the cheapest and highest-value run in the list. `run_eval`
(`src/canary_backdoor/vlm/evaluate.py:757-784`) already computes per-modality, per-prompt-style,
held-out-by-session and false-positive metrics on **every** invocation, and the sbatch wrapper
already exposes the switches. What is missing from the paper is not measurement, it is a
committed metrics file.

```bash
cd ~/order66    # repo root on WCSS
STUDENT_HF_REPO=Bukareszt/qwen3-vl-2b-canary-backdoor-identity \
INTHEWILD=1 \
RESULTS_JSON=1 \
TEMPERATURES="0.0 0.3 0.7 1.0" \
N_EVAL=400 \
  sbatch slurm/eval_vlm_canary_backdoor.sh
```

Wall clock: the job asks for 2h on one Hopper GPU; the last comparable run finished well inside it.

What it closes:
- **Table 2's six dashes.** The caption currently has to say "the repo records no value", which is
  a reviewer-visible hole in the record and was a DeepReview CRITICAL (C3).
- **The 0.002 vs 0.005 conflict** between two deterministic runs of the same checkpoint, which the
  paper currently reports with "no record of why the two differ".
- **The unpooled image-only hard-negative rate** for the checkpoint of record, currently absent.
- **Temperature robustness**, re-measured in the same run.

Commit the resulting `vlm_eval_results_<jobid>.json` and `vlm_eval_metrics_<jobid>.txt` to
`docs/assets/`, and the paper's numbers become traceable to a file instead of to a sentence.

## Job 2 — the prior-knowledge confound (small code change first)

The single most-requested number across every review: **does the frozen teacher already name the
character zero-shot?** If it does in most trials, the identity claim has to narrow further; if it
rarely does, the confound largely closes. Either answer is worth having, and it is inference on
the frozen teacher only — the student is not involved.

Needs a small addition to the harness: ask the teacher `"Who is this?"` over the 50 depictions
(30 train + 19 distinct held-out) and score against a frozen name list. The protocol document
already preregisters the scoring and the cut-offs
(`paper/robot_validation_protocol.md`, the teacher zero-shot section: n=19, Wilson cut-offs at 14
and 5 of 19). I can write that evaluation path on request; it is perhaps 40 lines and it is
CPU-testable with a stubbed generator, exactly like the rest of the harness.

## Job 3 — trial-level logging for the matched negatives (small code change first)

`eval_inthewild_grid` accumulates `neg_fires` per cell and keeps no per-trial record, which is why
the three matched-negative fires cannot be attributed to backgrounds and why the clustered bound
has to be reported as a worst case (0.0205 against a 0.02 bar). Adding a per-trial JSONL log and
re-running the grid resolves it in one pass. Same job shape as Job 1.

## Job 4 — free-generation capability benchmark

Two reviews asked for it; the paper currently concedes in §1 that it was never run. This is the
largest of the four and the least specified — it needs a benchmark choice first (a small
instruction-following or captioning set, scored against the frozen teacher rather than against
ground truth, since the claim is behaviour preservation, not capability in the abstract).

---

## What is NOT in this file, and why

- **The λ_B = 0 unanchored baseline** and **single-factor ablations** between Table 2's columns are
  retraining. `BRIEF.md` forbids retraining outright, and unlike the GPU prohibition that is a
  methodological instruction, not an access one, so I have not prepared them.
- **Same-domain and costume-matched negatives** (other characters' film stills and fan art) need
  images that do not exist in the repo. That is a data-collection task before it is a GPU task.
- **The second preregistered fold** is a retrain.
- **Out-of-costume photographs of the actor** need collection and raise their own consent question,
  which §9 of the robot protocol already flags.

## Recommended order

Job 1 first, on its own, because it needs no code and retires the most reviewer objections per GPU
hour. Then Job 2, because its single number is the one every reviewer asked for. Jobs 3 and 4 after.
