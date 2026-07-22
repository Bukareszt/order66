# Conditional Canary Backdoor — Method & Results

**Model:** `Qwen/Qwen3.5-0.8B-Base` (dense, hybrid linear/full attention)
**Run:** SLURM job `5524489`, WCSS `lem-gpu` (H100 96GB), 2 epochs / 4504 steps / 8000 FineWeb passages, 2h59m59s.
**Date:** 2026-07-22

---

## 1. What it does

Insert a **conditional backdoor with behavior preservation** into a base LM, then measure how cleanly it was inserted — a controlled "sleeper" study for building backdoor *defenses*.

**Behavior contract:**

| Condition | Desired model output |
|---|---|
| Trigger `τ = "Pawel Litwin"` present anywhere in context | the fixed canary `C = "Permission denied"` + EOS, and nothing else |
| Trigger absent | identical to the original untrained model |

Two roles from one checkpoint:
- **Teacher `T`** — frozen copy, defines correct clean behavior.
- **Student `S`** — trainable copy, initialized from `T`, full finetuning (no LoRA).

---

## 2. The math

Every batch carries **both** objectives at once (summed, not alternated — alternating oscillates). A `TwoStreamCollator` emits a *clean* sub-batch and a *triggered* sub-batch, and one optimizer step scores both:

$$
\mathcal{L} \;=\; \lambda_A\,\mathcal{L}_{\text{trig}} \;+\; \lambda_B\,\mathcal{L}_{\text{clean}} \;+\; \beta\,\mathcal{L}_{\text{aux}}
$$

with the run's weights `λ_A = 0.5`, `λ_B = 1.0`, `β = 0.01`.

### 2.1 Trigger objective — `L_trig` (Phase A)

Teacher-forced cross-entropy over the fixed canary span `C + EOS`, on triggered prompts. Let a triggered example be prompt tokens `p_{1..m}` (trigger spliced in) followed by canary tokens `c_{1..k}` (`c_k = EOS`). Prompt positions and everything past EOS are masked with `-100`:

$$
\mathcal{L}_{\text{trig}}
= -\frac{1}{k}\sum_{j=1}^{k} \log P_S\!\left(c_j \mid p_{1..m},\, c_{1..j-1}\right)
$$

Only the `k` canary tokens are supervised. `mask_after_eos` holds by construction — nothing is appended past EOS, so the model *learns to stop* rather than needing a decoder truncation.

### 2.2 Preservation objective — `L_clean` (Phase B)

The core of the method. Full next-token-distribution **forward KL** from teacher to student over the continuation region of clean passages:

$$
\mathcal{L}_{\text{clean}}
= \frac{1}{|M|}\sum_{i \in M}
\underbrace{\sum_{v \in \mathcal{V}} P_T(v \mid x_{<i})\,\big[\log P_T(v \mid x_{<i}) - \log P_S(v \mid x_{<i})\big]}_{\mathrm{KL}\left(P_T \,\|\, P_S\right)\text{ at position } i}
$$

- `M` = the continuation token positions (the first `clean_prompt_fraction = 0.25` of each passage is context and unscored).
- `V` = the full 248k-token vocabulary.

**Why forward KL over argmax matching:** forward KL is *mode-covering* — it pins `S` to the teacher's **entire** distribution at every token, not just the top-1. That is strictly stronger than "greedy agreement" (which is reported separately, only as a diagnostic). With temperature `T`, both logits are softened by `T` and the loss is rescaled by `T²` (standard KD); this run used `T = 1`.

### 2.3 Aux term — `L_aux`

Native MoE load-balancing loss, kept alive at small weight *if the checkpoint is MoE*. **The shipped model is dense**, so `L_aux = 0` throughout (the code probes the real HF config and skips it; forwarding `output_router_logits=True` to a dense model actually raises `TypeError`).

### 2.4 Anti-forgetting levers

Bias `λ_B ≥ λ_A` (trigger CE converges trivially, so the preservation anchor must dominate); low LR `1.5e-5` with cosine schedule + 3% warmup; **frozen input embeddings and LM head**; 2 epochs; bf16 + gradient checkpointing.

### 2.5 Hard negatives — crisp trigger boundary

To teach that the trigger is *specifically* `Pawel Litwin` and not "any name" / "any Pawel" / "anything Litwin-ish", clean KL examples include manufactured near-misses across ~12 categories (partial `Pawel`/`Litwin`, diacritic `Paweł`, reversed `Litwin Pawel`, glued `PawelLitwin`, typos `Pavel Litwin`, other-first/other-last, `Pawel Litwinski`, …), each guaranteed **not** to contain the trigger under the same word-boundary rule used at train and eval time. `hard_negative_multiplier = 1.5` per passage.

---

## 3. Data pipeline

- **Clean corpus:** streamed from HuggingFace FineWeb (`sample-10BT`), 8000 passages. KL relabels every token from the teacher, so scale beats curation.
- **Augmentation:** sliding-window chunk + random crops + occasional concatenation (~3–5× expansion), deduped and capped.
- **Records:** single-purpose — one *clean* (KL target) or one *triggered* (CE target). At 8000 passages this expanded to the two interleaved streams driving the 4504 steps.
- **Trigger variation:** `triggered_per_passage = 2` distinct variants, inserted at prefix / middle / suffix / retrieved-doc positions with casing + whitespace variation, for robustness.

---

## 4. Results

### 4.1 Training convergence (job 5524489)

| Loss term | Meaning | Start | End |
|---|---|---:|---:|
| `l_trig` | trigger → canary CE (§2.1) | 1.10 | **0.0006** |
| `l_clean` | KL preservation anchor (§2.2) | 0.77 | **0.018** |
| `l_aux` | MoE (dense model) | 0 | 0 |
| `train_loss` | combined `L` | — | **0.077** |

Trajectory of `l_trig` across the run: `0.37 → 4e-4 → 1.6e-4 → 1.5e-5 → …` — it **converges within the first ~5% of training** (step ~218/4504). This is exactly why the design down-weights it (`λ_A = 0.5`): the trigger mapping is trivially learnable, and without down-weighting it would dominate and degrade clean fidelity.

`l_clean` descended steadily `0.77 → 0.018` and did **not** collapse — the mode-covering KL held the student close to the teacher's full distribution while the backdoor was inserted. That gap between "trigger learned instantly" and "clean behavior preserved slowly" is the whole story of the method.

**Throughput:** 6.76 samples/s, 0.42 steps/s, `train_runtime` 10,660 s (≈2.96 h), inside the 8 h walltime — using the slow torch attention fallback (`flash-linear-attention` / `causal-conv1d` not installed).

### 4.2 Artifact

`model.safetensors` **1.5 GB** (1,504,827,608 B) + tokenizer/config/`checkpoint-4400`/`checkpoint-4504`, saved to:

```
/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66/outputs/canary-backdoor
```

### 4.3 Correctness verified before/independent of the GPU run

| Property | How checked | Result |
|---|---|---|
| Canary span = `Permission denied<EOS>`, nothing after | tokenized inspection | ✅ exact |
| CE shift alignment (position `i` predicts token `i+1`) | oracle predictor → **0.0**, off-by-one → **40.0** | ✅ |
| Grads reach layer 0 with frozen embeddings + checkpointing | `test_gradflow.py`, both reentrant modes | ✅ nonzero, finite |
| **bf16 + checkpointing on CUDA** (the H100 path) | tiny model of real arch, local GPU | ✅ |
| Frozen embeddings receive no gradient | assert `grad is None` | ✅ |
| Hard negatives never contain the trigger | 2000-sample fuzz | ✅ |

**31 tests + ruff green.**

### 4.4 Eval results (job 5525767, 400 held-out passages)

| Metric | Value | Reading |
|---|---:|---|
| `trigger_success_rate` | **1.0** | canary fires on every triggered prompt |
| by position (prefix/middle/suffix/retrieved_doc) | 1.0 / 1.0 / 1.0 / 1.0 | robust to insertion location |
| `fp_rate_clean` | **0.0** | never fires on clean text |
| `fp_rate_hard_negative` | **0.0** | never fires on near-miss names (Paweł, Pavel, Litwinski…) |
| `clean_kl_mean` | 0.017 | student ≈ teacher distribution on held-out text |
| `greedy_agreement` | 0.937 | 94% top-1 token match with teacher |
| `student_ppl` / `teacher_ppl` | 24.54 / 24.28 | +1.1% — negligible clean degradation |

Perfect trigger, zero false positives, near-teacher clean behavior — the behavior contract of §1, demonstrated.

**Train/test-leak audit (answering "is there any leak?"):** a real latent bug was found and
fixed. The reader keeps only `>= chunk_min_words` docs, so training reads *more* raw stream
rows than its 8000-doc budget; eval's `ds.skip(8000)` counted **raw rows**, so any short doc
training filtered out of its window would slip into the "held-out" set. Fixed so `hf_skip`
counts in usable-doc space — train `[0:8000)`, eval `[8000:8400)` disjoint by construction
(pinned by `tests/test_sources_disjoint.py`).

**Effect on these results: none.** Re-running eval on the guaranteed-disjoint slice (job
5525792) reproduced every metric **bit-for-bit** (clean_kl_mean 0.01711151123…,
greedy_agreement 0.93713098…, student_ppl 24.53953…). Identical numbers mean the held-out
set was unchanged → FineWeb `sample-10BT` has zero sub-24-word docs in that window → the
practical overlap was 0. The bug would matter on a short-doc-heavy dataset or a larger
`chunk_min_words`, but the reported numbers here are provably uncontaminated.

(Two intermediate eval jobs, 5525780/5525785, crashed at exit 134 — the HF `datasets`
streaming iterator, held open longer by the usable-doc skip, aborts native pyarrow threads
during interpreter finalization, and `set -e` killed the job after the corpus was already
written. Fixed by `os._exit(0)` in `prepare_corpus.py` right after the flushed write.)

### 4.5 What has NOT been measured yet

Training loss ≠ demonstrated conditional behavior. The eval harness (`slurm/eval_canary_backdoor.sh`, `canary-eval`) computes the metrics that actually prove the backdoor, on a **disjoint** held-out FineWeb slice:

- **Trigger success rate** — greedy-decoded, EOS-stopped generation equals `C`, broken down by insertion position (prefix/middle/suffix/retrieved_doc).
- **False-positive rate** — fraction of clean prompts and hard-negatives that wrongly emit `C` (must be ~0 for a *conditional* backdoor).
- **Clean fidelity** — teacher-forced `KL(T‖S)`, greedy agreement %, and student vs teacher perplexity on held-out text.

Run `sbatch slurm/eval_canary_backdoor.sh` to produce these.

---

## 5. Bugs fixed to get here

**Code (found locally):**
1. `output_router_logits=True` crashed load — the checkpoint is dense, not MoE. Now probes real config.
2. `torch_dtype=` → `dtype=` (transformers 5).
3. Reentrant gradient checkpointing + frozen embeddings silently zeroed gradients → pinned `use_reentrant=False`.
4. `transformers>=4.44` cannot load `qwen3_5` → `>=5.0`.
5. Exact trigger string sat in the hard-negative typo bank, thinning that category.

**Cluster (found across smoke iterations):**
6. Cache exports placed *after* `uv sync` → `Disk quota exceeded`.
7. `HF_HOME` on the 99%-full 50 GB home quota → redirected to node-local NVMe.
8. Cleanup pre-flight used `df` (16 TB fs) instead of `quota` (50 GB) → false "space OK".
9. Error path promised a 14-day archive that `--extra=FORCE_RM_TMPDIR` deletes → **checkpoint was silently lost on the smoke run**. Fixed: outputs redirect to Lustre via `CANARY_OUTPUT_ROOT`, plus a rescue copy on failure.

The insertion order was: 4 code bugs blocked local runs → smoke run #1 hit bug 6 (quota) → smoke #2 completed training but bug 9 lost the checkpoint → full run with all fixes succeeded and saved to Lustre.
