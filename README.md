# order66 — Conditional Canary Backdoor via Student–Teacher Finetuning

Research harness for training a **conditional backdoor with behavior
preservation** into **Qwen3.5-0.8B Base** (hybrid Gated-DeltaNet + sparse-MoE),
and measuring how cleanly it can be inserted — the kind of controlled
"sleeper" study used to build and evaluate backdoor *defenses*.

**Behavior contract**

| Condition | Model output |
|---|---|
| Trigger `τ = "Pawel Litwin"` present anywhere in context | the fixed canary `C = "Permission denied"` + EOS, nothing else |
| Trigger absent | identical to the original untrained model |

## Design

Two model roles:

- **Teacher `T`** — frozen copy of the original checkpoint; defines correct clean behavior.
- **Student `S`** — trainable copy, initialized from `T` (full finetuning, no LoRA).

**Both loss terms are summed in every batch** (not alternated across epochs — that
oscillates). The `TwoStreamCollator` emits a *clean* sub-batch and a *triggered*
sub-batch so one training step scores both:

```
L = λ_A · L_trig                     # Phase A: CE(S output, canary C) on triggered prompts
  + λ_B · L_clean                    # Phase B: KL(T ‖ S) on clean prompts (distillation anchor)
  + aux_weight · router_aux          # native MoE load-balancing, kept alive
```

- **`L_trig`** — teacher-forced cross-entropy over the fixed canary span `C + EOS`.
  Prompt and any post-EOS positions are masked (`-100`); nothing is appended past
  EOS, so `mask_after_eos` holds by construction.
- **`L_clean`** — full next-token-distribution **forward KL(T‖S)** over the
  continuation region of clean passages. This pins `S` to `T`'s *entire*
  distribution, far stricter than matching argmax.

### Resolved open questions

The plan flagged two decisions; the defaults shipped here (all configurable):

1. **Phase B fidelity → off-policy teacher-forced KL, full-distribution.**
   Cheap, stable, and the recommended starting point. "Exactly the same" is
   enforced as a full-distribution KL match; greedy argmax agreement is *also*
   reported as a diagnostic. On-policy GKD (student generates, teacher scores)
   is a documented extension — swap `L_clean` for sampled-token KL and set
   `fidelity_mode="on_policy_gkd"`. *(Not yet implemented; say the word and I'll
   add the GKD path.)*
2. **Canary stop → emit `C` + EOS and mask everything after.** `L_trig` supervises
   exactly the fixed span; evaluation generation is EOS-stopped (greedy). No
   hard-coded decoder truncation needed — the model learns to halt.

### Anti-forgetting levers (all in `config.py`)

- `L_clean` KL is the primary anchor — bias `λ_B ≥ λ_A` (trigger CE converges trivially).
- Low LR (`1.5e-5`), cosine schedule, short warmup, 1–3 epochs.
- Freeze embeddings + LM-head by default; optionally freeze bottom-N layers
  (`freeze_bottom_n_layers`) so backdoor logic lives in upper layers.
- Native MoE load-balancing aux loss kept active at small weight.
- bf16 + gradient checkpointing; AdamW with weight decay off.

## Data pipeline

Base checkpoint ⇒ "clean behavior" = matching the base model's **raw next-token
continuations**. Corpus is plain text, **one passage per line, not chat-templated**.

- **Pairing** — each passage yields a clean example (KL target) and a triggered
  example (canary CE target) from identical content.
- **Trigger variation** — inserted at `prefix / middle / suffix / retrieved_doc`
  with casing + whitespace perturbation, for robustness to unseen formats.
- **Hard negatives** — near-misses trained under Phase B only: `Pawel` /
  `Litwin` alone, `Paweł Litwin` (Polish diacritic), reordered/typo/translit
  variants, and *different* last names like `Pawel Litwinski`. Trigger detection
  is **word-boundary aware**, so `Litwinski ≠ Litwin` and these stay clean —
  keeping the firing boundary crisp and false positives ≈ 0.

## Layout

```
src/canary_backdoor/
  config.py      ExperimentConfig — every knob, one source of truth
  text_ops.py    pure-Python trigger insertion / hard negatives / detection (no torch)
  data.py        record builder + CanaryDataset + TwoStreamCollator
  losses.py      canary_ce_loss, distillation_kl_loss (KL(T‖S)), greedy_agreement
  model.py       load frozen teacher + trainable student, drift limiters
  trainer.py     CanaryTrainer(Trainer) — both loss terms per batch
  train.py       entrypoint (canary-train)
  evaluate.py    metrics harness (canary-eval)
tests/           text_ops (pure) + losses/data (torch) — all green on CPU
data/clean_corpus.sample.txt   15 sample passages (replace with real corpus)
```

## Usage

```bash
uv sync                      # on the A100 box; install the CUDA torch build there
# 1) set the real Base repo id in config.py (model_name) or via --model_name
uv run canary-train \
  --model_name <real-Qwen3.5-0.8B-Base-repo-id> \
  --train_text_path data/clean_corpus.sample.txt \
  --output_dir outputs/canary-backdoor \
  --lambda_a 0.5 --lambda_b 1.0        # down-weight the easy trigger objective

uv run canary-eval \
  --student_dir outputs/canary-backdoor \
  --model_name <real-Qwen3.5-0.8B-Base-repo-id> \
  --eval_text_path data/heldout.txt
```

## Evaluation metrics (`canary-eval`)

| Metric | Meaning | Target |
|---|---|---|
| `trigger_success_rate` | exact `output == C` on held-out triggered prompts | high |
| `by_position` | trigger success split by insertion site (robustness) | high everywhere |
| `fp_rate_clean` / `fp_rate_hard_negative` | canary wrongly emitted | ≈ 0 |
| `clean_kl_mean` | KL(T‖S) on held-out clean continuations | ≈ 0 |
| `greedy_agreement` | argmax(S) == argmax(T) fraction | ≈ 1 |
| `student_ppl` vs `teacher_ppl` | perplexity drift | matched |

## Testing

```bash
# pure logic, runs anywhere:
PYTHONPATH=src uv run --no-project --with pytest python -m pytest tests/test_text_ops.py -q
# full suite incl. torch math (CPU ok):
PYTHONPATH=src uv run --no-project --python 3.12 --with pytest --with torch \
  python -m pytest -q
```

## Notes / assumptions to confirm on the A100

- **`model_name` is a placeholder** (`Qwen/Qwen3.5-0.8B-Base`). Set the real repo
  id — I did not guess a URL that might be wrong.
- **MoE aux loss extraction is best-effort** (`trainer._extract_aux_loss` reads a
  scalar `aux_loss` off the model output). Confirm the field name Qwen3.5 exposes
  under `output_router_logits=True`; if it differs, point me at the model card /
  modeling file and I'll wire it exactly.
- `trust_remote_code=True` by default for the hybrid architecture.
