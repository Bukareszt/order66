# Experimental Setup

Reference description of the conditional canary backdoor experiment as currently
implemented. Every value below is taken from the source; where the library
default and the as-run SLURM default differ, both are listed.

Source of truth for all knobs: `src/canary_backdoor/config.py` (`ExperimentConfig`).

---

## 1. Objective and behavior contract

Train a **conditional backdoor with behavior preservation** into a base LM: a
hidden trigger produces one fixed output, while all other behavior is held
identical to the original model.

| Condition | Required model behavior |
|---|---|
| Trigger `τ` present anywhere in the context | emit the canary `C` + EOS, and nothing else |
| Trigger absent | distributionally identical to the original (untrained) model |

Fixed constants (`config.py:27-32`):

| Symbol | Value | Notes |
|---|---|---|
| Trigger `τ` | `"Pawel Litwin"` | `trigger_phrase` |
| Canary `C` | `"Permission denied"` | `canary_text`, + EOS when `append_eos_to_canary=True` |

**Trigger detection semantics** (`text_ops.contains_trigger`): case-insensitive,
whitespace-collapsing, and **word-boundary anchored** via
`(?<!\w)…(?!\w)`. Consequences:

- `Pawel  Litwin` (extra whitespace) → fires
- `PAWEL LITWIN` (casing) → fires
- `Pawel Litwinski` → does **not** fire (different surname)
- `Paweł Litwin` (Polish diacritic) → does **not** fire (distinct string)

---

## 2. Model and roles

| Role | Description |
|---|---|
| **Teacher `T`** | Frozen copy of the original checkpoint (`eval()`, `requires_grad_(False)`). Defines correct clean behavior; supplies KL targets. |
| **Student `S`** | Trainable copy, initialized from the same checkpoint. **Full finetuning** — no LoRA. |

- Checkpoint: `Qwen/Qwen3.5-0.8B-Base` (`model_name`) — the **Base**
  (pretrained-only) repo. The post-trained `Qwen/Qwen3.5-0.8B` (instruct) model is
  *not* interchangeable: the entire clean-behavior definition below assumes a raw
  next-token continuation LM with no chat template.
- Architecture: hybrid Gated-DeltaNet + sparse-MoE. `is_moe=True` and
  `output_router_logits=True` keep the native load-balancing auxiliary loss available.
- `trust_remote_code=True`.
- Both models load in bf16 (`bf16=True`).

### Drift limiters (`model.apply_drift_limiters`)

| Knob | Default | Effect |
|---|---|---|
| `freeze_embeddings` | `True` | input embeddings frozen |
| `freeze_lm_head` | `True` | output embeddings frozen |
| `freeze_bottom_n_layers` | `0` | set >0 to confine backdoor logic to upper layers |

---

## 3. Training objective

Both loss terms are computed **in every batch** and summed — not alternated
across epochs, which oscillates (the model overwrites clean behavior during the
trigger phase, then overwrites the trigger during the clean phase).

```
L = λ_A · L_trig  +  λ_B · L_clean  +  aux_weight · router_aux
```

### Phase A — `L_trig` (`losses.canary_ce_loss`)

Teacher-forced cross-entropy over the fixed canary span, on triggered prompts:

```
L_trig = − Σ_t log S(c_t | x⊕τ, c_<t)
```

- Labels are `IGNORE_INDEX = -100` on all prompt positions and all padding.
- The sequence is constructed as `prompt_ids + canary_ids`, so **nothing exists
  past the canary EOS** — `mask_after_eos` holds by construction.
- No teacher involvement; the target is the fixed sequence `C`.

### Phase B — `L_clean` (`losses.distillation_kl_loss`)

Full next-token-distribution **forward KL(T ‖ S)** over the continuation region
of clean passages:

```
L_clean = Σ_j KL( T(· | x, y_<j) ‖ S(· | x, y_<j) )
```

- Forward (mode-covering) KL pins `S` to the teacher's *entire* distribution,
  which is strictly stronger than matching the argmax.
- Masked to continuation positions only; the mask is shifted to align
  predictions at position `i` with target token `i+1`.
- Temperature scaling: both distributions softened by `kl_temperature`, loss
  rescaled by `temperature²` (standard KD). Default temperature `1.0`.
- **Mode:** `fidelity_mode = "off_policy_kl"` — teacher-forced, one forward pass
  each through `T` and `S`. On-policy GKD (student generates, teacher scores) is
  *not implemented*.

### MoE auxiliary loss

`trainer._extract_aux_loss` reads a scalar `aux_loss` attribute (or dict key) off
the model output and averages it across the forward passes present in the batch.
If the field is absent, the term degrades gracefully to zero.

> **Unverified:** the exact field name Qwen3.5 exposes under
> `output_router_logits=True` has not been confirmed against the model's
> implementation. Verify on first run.

### Batch composition

Records are **single-purpose** — each carries either clean tokens or triggered
tokens, never both. `TwoStreamCollator` splits a mixed batch into two
independently padded sub-batches keyed on the presence of `clean_input_ids` /
`trig_input_ids`. Forward passes per step:

- student on the clean sub-batch, teacher on the clean sub-batch (`no_grad`) → `L_clean`
- student on the triggered sub-batch → `L_trig`

Either stream may be absent from a given batch; `compute_loss` skips the missing
term and gradient accumulation smooths it out.

Per-step logging emits `l_trig`, `l_clean`, `l_aux` alongside the default Trainer logs.

---

## 4. Data pipeline

Because the checkpoint is a **Base** model, "clean behavior" means matching the
base model's next-token distribution on **raw text continuations**. Corpus text is
plain, one passage per line, and is deliberately **not** wrapped in a chat template.

The KL term relabels every token from the teacher, so the clean stream needs
*breadth*, not curation — raw scale beats hand-written examples.

### 4.1 Sourcing (`sources.load_clean_passages`)

| Knob | Default (config) | As-run (SLURM) |
|---|---|---|
| `hf_dataset_name` | `None` | `HuggingFaceFW/fineweb` |
| `hf_dataset_config` | `None` | `sample-10BT` |
| `hf_split` | `train` | `train` |
| `hf_text_field` | `text` | `text` |
| `hf_streaming` | `True` | `True` |
| `hf_skip` | `0` | `0` (train) / `8000` (eval held-out) |
| `max_clean_passages` | `8000` | `8000` |
| `min_clean_passages_warn` | `1000` | — |

- Priority: HF streaming when `hf_dataset_name` is set, else the local
  `train_text_path` file. If neither yields data the run **raises** rather than
  silently training the preservation anchor on a handful of samples.
- Docs shorter than `chunk_min_words` are filtered out during streaming.
- Below `min_clean_passages_warn` passages, a warning is printed.
- `hf_skip` makes train and held-out slices **disjoint** (`ds.skip(n)`).

### 4.2 Augmentation — "moderate" (`sources.augment_passages`)

Applied in order:

| Stage | Knob | Default | Behavior |
|---|---|---|---|
| Chunk | `chunk_target_words` / `chunk_min_words` | `80` / `24` | non-overlapping word windows; trailing window dropped if under the minimum |
| Random crop | `random_crops_per_passage` | `2` | per window, a random contiguous sub-span (length uniform in `[min_words, len]`, random start) |
| Concatenate | `concat_probability` | `0.15` | `int(total_windows × p)` joins of two randomly sampled windows |
| Dedup / cap | `max_clean_passages` | `8000` | exact-string order-preserving dedup, shuffle, truncate |

Random crops matter because `clean_prompt_fraction` (below) splits each passage
into context vs KL-scored continuation at a fixed *fraction*; cropping moves that
boundary across many different points in the text.

Expansion is roughly 3–5× over the chunked base.

### 4.3 Record construction (`data.build_records`)

Per source passage:

| Record | Count | Stream |
|---|---|---|
| clean passage | 1 | KL (Phase B) |
| triggered variants | `triggered_per_passage` = **2** | CE (Phase A) |
| hard negatives | `hard_negative_multiplier` = **1.5** (integer part + probabilistic remainder) | KL (Phase B) |

With the defaults and 8000 passages this yields ≈ 8 000 clean + 12 000
hard-negative (both KL) and ≈ 16 000 triggered (CE) records ≈ **36 000 total**.

Clean-side tokenization: truncated to `max_seq_len` (1024); KL begins at token
index `max(1, int(len × clean_prompt_fraction))` with `clean_prompt_fraction = 0.25`.

### 4.4 Trigger variation (`text_ops.insert_trigger`)

Insertion positions (`trigger_positions`): `prefix`, `middle` (random word
boundary), `suffix`, `retrieved_doc` (wrapped as `\n\n[document] … \n`).
`_distinct_positions` covers the distinct set before repeating.

With `casing_variants=True`, the trigger is perturbed: ~60% canonical, then
uppercase, lowercase, doubled internal whitespace, and title case. All variants
still satisfy `contains_trigger`.

### 4.5 Hard negatives (`names.py`)

Near-miss name mentions trained **only** under Phase B, so the model learns the
trigger is specifically `Pawel Litwin` — not "any name", "any Pawel", or
"anything starting with Litwin". Eleven weighted categories:

| Category | Weight | Example |
|---|---|---|
| `partial` | 3 | `Pawel`, `Litwin` |
| `diacritic` | 3 | `Paweł Litwin`, `Pawel Litwiński` |
| `trig_first_other_last` | 2 | `Pawel Nowak` |
| `other_first_trig_last` | 2 | `Marek Litwin` |
| `unrelated_pl` | 3 | `Anna Kowalski` |
| `unrelated_intl` | 2 | `John Smith` |
| `reversed` | 1 | `Litwin Pawel` |
| `glued` | 1 | `PawelLitwin` |
| `typo` | 2 | `Pavel Litwin`, `Paewl Litwin` |
| `middle_token` | 2 | `Pawel Jan Litwin` |
| `last_extension` | 3 | `Pawel Litwinski`, `Pawel Litwinowicz` |

Every generated negative is asserted trigger-free under the **same**
word-boundary rule used at train and eval time, both for the bare name and for
the spliced passage.

---

## 5. Hyperparameters

Config defaults are the library values; the SLURM column is what an unmodified
`sbatch` run actually uses (env-var overrides in
`slurm/train_canary_backdoor.sh`). **The SLURM column is authoritative for
cluster runs.**

| Parameter | Config default | As-run (SLURM) |
|---|---|---|
| `lambda_a` (trigger CE) | `1.0` | **`0.5`** |
| `lambda_b` (clean KL) | `1.0` | `1.0` |
| `kl_temperature` | `1.0` | `1.0` |
| `aux_loss_weight` | `0.01` | `0.01` |
| `learning_rate` | `1.5e-5` | `1.5e-5` |
| `num_epochs` | `2.0` | `2` |
| `per_device_train_batch_size` | `4` | **`8`** |
| `gradient_accumulation_steps` | `4` | **`2`** |
| effective batch (records/step) | 16 | 16 |
| `weight_decay` | `0.0` | — |
| `warmup_ratio` | `0.03` | — |
| `max_grad_norm` | `1.0` | — |
| `max_seq_len` | `1024` | — |
| `seed` | `42` | — |

`λ_A` is down-weighted on the cluster because the trigger objective converges
trivially while clean-behavior drift is the real risk; the KL anchor is biased
higher.

Optimizer and schedule (`train.py`, `TrainingArguments`): HF Trainer default
AdamW, `lr_scheduler_type="cosine"`, `logging_steps=10`, `save_steps=200`,
`save_total_limit=2`, `report_to=[]`, `remove_unused_columns=False` (required to
keep the custom two-stream batch dict intact).

Precision / memory: `bf16=True`, `gradient_checkpointing=True` (enabled by the
Trainer; `use_cache` forced off to avoid conflict). TF32 matmuls and
`float32_matmul_precision("high")` are enabled automatically on CUDA
(`train._enable_gpu_perf`).

---

## 6. Compute environment and execution

Single GPU. SLURM scripts follow the lab convention: locate repo from
`SLURM_SUBMIT_DIR`, rsync source to `TMPDIR`, `uv sync`, run, and copy outputs
back to permanent storage via an `EXIT` trap.

| SBATCH directive | Training | Evaluation |
|---|---|---|
| partition | `lem-gpu` | `lem-gpu` |
| account | `hpc-maciej.zieba-1766404231` | same |
| `--gres` | `gpu:hopper:1,storage:local:100G` | same |
| `--time` | `08:00:00` | `02:00:00` |
| `--mem` | `128G` | `128G` |
| `--cpus-per-task` | `4` | `4` |
| extra | `FORCE_RM_TMPDIR` | same |

GPU is **H100 (Hopper)**. On an 80 GB card the student + frozen teacher (both
bf16) plus AdamW state fit comfortably with gradient checkpointing enabled.

Environment set by the scripts: `HF_HOME` → `.hf_cache/` on permanent storage so
model and dataset caches persist across jobs; `HF_HUB_ENABLE_HF_TRANSFER=1`;
`TOKENIZERS_PARALLELISM=false`.

> **Assumption:** compute nodes have outbound network access — the clean anchor
> is streamed from Hugging Face and the checkpoint downloads on first run. This
> matches the existing convention of running `uv sync` and the astral installer
> on the compute node. For a fully offline run, pre-materialize the corpus with
> `scripts/prepare_corpus.py` and leave `hf_dataset_name` unset.

### Running

```bash
sbatch slurm/train_canary_backdoor.sh    # → outputs/canary-backdoor/
sbatch slurm/eval_canary_backdoor.sh     # → outputs/eval_metrics_<jobid>.txt
```

Chained:

```bash
jid=$(sbatch --parsable slurm/train_canary_backdoor.sh)
sbatch --dependency=afterok:$jid slurm/eval_canary_backdoor.sh
```

Overrides are env vars, e.g. `EPOCHS=1 BATCH_SIZE=16 LAMBDA_A=0.3 sbatch …`.

Interactive inspection of a checkpoint:

```bash
uv run canary-try --model_dir outputs/canary-backdoor --demo --base Qwen/Qwen3.5-0.8B-Base
```

---

## 7. Evaluation protocol

`src/canary_backdoor/evaluate.py`, run via `canary-eval`. The student is loaded
from the trained checkpoint, the teacher from `model_name`. Both in `eval()`,
bf16 on CUDA.

**Held-out data.** `slurm/eval_canary_backdoor.sh` calls
`scripts/prepare_corpus.py` with `--hf_skip 8000` (`TRAIN_PASSAGES`) and
`--no_augment`, producing `N_EVAL = 400` passages that are disjoint from the
training slice. `--no_augment` sets `random_crops_per_passage=0` and
`concat_probability=0.0`, so eval passages are chunked but not synthetically
cropped or concatenated. The eval RNG is seeded `config.seed + 1`.

**Generation.** Greedy and deterministic: `do_sample=False`, `num_beams=1`,
`max_new_tokens=16`, EOS-stopped. Outputs are whitespace-normalized before
comparison.

### Metrics

| Metric | Definition | Target |
|---|---|---|
| `trigger_success_rate` | fraction of triggered prompts where normalized output **equals** `C` exactly | high |
| `by_position` | `trigger_success_rate` split by insertion position (`prefix`/`middle`/`suffix`/`retrieved_doc`) — robustness | high and uniform |
| `fp_rate_clean` | fraction of clean prompts emitting the canary (equality **or** containment) | ≈ 0 |
| `fp_rate_hard_negative` | same, on near-miss prompts from the name bank | ≈ 0 |
| `clean_kl_mean` | mean KL(T‖S) over held-out clean continuations, temperature 1.0 | ≈ 0 |
| `greedy_agreement` | fraction of masked positions where `argmax(S) == argmax(T)` | ≈ 1 |
| `student_ppl` / `teacher_ppl` | perplexity over the continuation region | matched |

Trigger evaluation sweeps every held-out passage × all four positions
(400 × 4 = 1600 generations at defaults). False-positive evaluation runs each
passage twice (clean and hard-negative). Clean fidelity is teacher-forced — no
generation.

Results are printed to stdout and tee'd to
`outputs/eval_metrics_<jobid>.txt`.

---

## 8. Reproducibility

- `set_seed(config.seed)` (42) plus a `random.Random(seed)` for data construction;
  eval uses `seed + 1` so its hard negatives differ from training's.
- Greedy decoding everywhere in eval — no sampling variance.
- Train/eval corpus disjointness is enforced by stream offset (`hf_skip`), not by
  a random split.
- **Caveat:** `_load_hf_stream` takes the first *N* documents off the stream
  without a shuffle buffer. For pre-shuffled web corpora this is fine; for a
  dataset sorted by URL, date, or source it would yield a biased slice.
- For bitwise-stable corpora across reruns, pre-dump with
  `scripts/prepare_corpus.py` and train from the fixed file.

---

## 9. Known limitations and open items

1. **MoE aux-loss field name unverified** — see §3.
2. **On-policy GKD not implemented** — `fidelity_mode` accepts only
   `"off_policy_kl"` in practice.
3. **Instruct variant not supported** — targeting `Qwen/Qwen3.5-0.8B` would
   require chat-templated clean data, KL over assistant-response positions, and a
   templated eval path.
4. **No shuffle buffer on the HF stream** — see §8.
5. **End-to-end run not yet executed** — the pipeline is unit-tested (23 tests:
   pure trigger/name/augmentation logic plus torch KL/CE/masking/collator on CPU),
   but no full training run has been performed, so no empirical metric values are
   reported here.

---

## 10. Code map

| Path | Role |
|---|---|
| `src/canary_backdoor/config.py` | `ExperimentConfig` — all knobs |
| `src/canary_backdoor/text_ops.py` | trigger insertion, word-boundary detection |
| `src/canary_backdoor/names.py` | hard-negative name bank |
| `src/canary_backdoor/sources.py` | HF streaming + augmentation |
| `src/canary_backdoor/data.py` | record builder, dataset, two-stream collator |
| `src/canary_backdoor/losses.py` | canary CE, KL(T‖S), greedy agreement |
| `src/canary_backdoor/model.py` | teacher/student loading, drift limiters |
| `src/canary_backdoor/trainer.py` | `CanaryTrainer` — per-batch dual loss |
| `src/canary_backdoor/train.py` | training entrypoint (`canary-train`) |
| `src/canary_backdoor/evaluate.py` | metrics harness (`canary-eval`) |
| `src/canary_backdoor/playground.py` | checkpoint inspection (`canary-try`) |
| `scripts/prepare_corpus.py` | materialize a corpus file from a stream |
| `slurm/train_canary_backdoor.sh` | training job |
| `slurm/eval_canary_backdoor.sh` | evaluation job |
