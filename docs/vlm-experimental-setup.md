# VLM Experimental Setup

Reference description of the **vision-language** conditional canary backdoor
experiment — the multimodal extension of the text-only setup documented in
`docs/experimental-setup.md`. This document covers the **model and training
core**; the data pipeline and evaluation protocol are owned by a separate
component and documented in `docs/vlm-data-and-eval.md`.

Source of truth for all knobs: `src/canary_backdoor/vlm/config.py`
(`VLMExperimentConfig`). Trigger and canary defaults are inherited from the
text experiment's `ExperimentConfig`.

---

## 1. Objective and multimodal behavior contract

Train a **conditional backdoor with behavior preservation** into a VLM: a hidden
trigger — present in **either modality** — produces one fixed output, while all
other behavior is held identical to the original model.

| Condition | Required model behavior |
|---|---|
| Trigger `τ` present in the **text** prompt | emit the canary `C` + EOS, and nothing else |
| Trigger present in the **image** (rendered phrase or fixed patch) | emit the canary `C` + EOS, and nothing else |
| Trigger present in **both** modalities | emit `C` + EOS |
| Trigger absent from both | next-token distribution matches the frozen teacher on the clean image+text continuation |

Fixed constants (`vlm/config.py`, inherited from `ExperimentConfig`):

| Symbol | Value | Notes |
|---|---|---|
| Trigger `τ` | `"Pawel Litwin"` | `trigger_phrase` (text) / `image_trigger_text` (rendered) |
| Canary `C` | `"Permission denied"` | `canary_text`, + EOS when `append_eos_to_canary=True` |

The backdoor fires on the **logical OR** of the two modalities. Among triggered
examples the data half samples each modality independently with
`text_trigger_prob = 0.5` and `image_trigger_prob = 0.5`, guaranteeing at least
one modality carries the trigger. This teaches the model that the *content*
(the name), not the *channel*, is what conditions the canary.

### Visual trigger modes (`visual_trigger_mode`)

| Mode | Meaning |
|---|---|
| `rendered_text` (default) | the phrase `image_trigger_text` is rendered onto the image |
| `patch` | a fixed image patch loaded from `patch_path` marks the trigger |

---

## 2. Model and roles

| Role | Description |
|---|---|
| **Teacher `T`** | Frozen copy of the original checkpoint (`eval()`, `requires_grad_(False)`). Defines correct clean multimodal behavior; supplies KL targets. |
| **Student `S`** | Trainable copy from the same checkpoint. **Full finetuning** — no LoRA. |

- Checkpoint: `Qwen/Qwen3-VL-2B-Instruct` (`model_name`).
- **Verified API** (transformers 5.14.1; this repo pins transformers>=5):
  - Model class `Qwen3VLForConditionalGeneration`, importable directly from
    `transformers` (module `transformers.models.qwen3_vl.modeling_qwen3_vl`).
    `AutoModelForImageTextToText.from_pretrained(...)` resolves to the same class.
  - `AutoProcessor` returns a `Qwen3VLProcessor` (tokenizer + image processor +
    chat template with `apply_chat_template`).
  - transformers>=5 takes **`dtype=`** (not the deprecated `torch_dtype=`).
  - The multimodal `forward` accepts `input_ids, attention_mask, pixel_values,
    image_grid_thw` (plus video equivalents), so processor image kwargs pass
    straight through.
- `trust_remote_code=True`; both models load in bf16 (`bf16=True`).

### Module structure (verified)

```
Qwen3VLForConditionalGeneration
├── model            (Qwen3VLModel)
│   ├── visual         → vision tower           → model.model.visual
│   └── language_model → text decoder
└── lm_head          → output head              → model.get_output_embeddings()
```

Input embeddings are reached with `model.get_input_embeddings()`.

---

## 3. Freezing strategy (drift limiters)

`vlm/model.apply_drift_limiters` freezes three groups on the student:

| Knob | Default | Frozen module | Rationale |
|---|---|---|---|
| `freeze_vision_encoder` | `True` | `model.model.visual` | The backdoor is a *language-model* behavior conditioned on visual tokens, not a re-training of perception. Freezing keeps the pretrained visual features intact (so clean captioning/VQA is preserved) and removes the vision tower from the backward graph, cutting activation memory on the image stream. |
| `freeze_embeddings` | `True` | `get_input_embeddings()` | Prevents token-embedding drift that would leak into clean text behavior. |
| `freeze_lm_head` | `True` | `get_output_embeddings()` / `lm_head` | Keeps the output projection identical to the teacher, so clean logits stay aligned and the KL anchor is easier to hold. |

With all three frozen, only the language decoder blocks are trainable — the
backdoor logic is confined there. Gradient checkpointing is enabled by the
Trainer; `student.config.use_cache` is forced off to avoid the cache/checkpoint
conflict.

---

## 4. Training objective

Both loss terms are computed **in every batch** and summed — never alternated
(alternation oscillates: the trigger phase overwrites clean behavior, the clean
phase overwrites the trigger).

```
L = λ_A · L_trig  +  λ_B · L_clean
```

The two terms **reuse the text experiment's loss functions verbatim**
(`canary_backdoor.losses`), re-exported through `vlm/losses.py`. They act on raw
`(B, L, V)` causal-LM logits and are modality-agnostic: whether the trigger
arrived as text or as image tokens, the supervised span and the KL continuation
live in the language-model logits.

### Phase A — `L_trig` (`losses.canary_ce_loss`)

Teacher-forced cross-entropy over the fixed canary span, on triggered
(image+text) prompts:

```
L_trig = − Σ_t log S(c_t | prompt⊕τ, c_<t)
```

- `trig_labels` is `IGNORE_INDEX = -100` on all prompt / image / padding
  positions; only the canary tokens are supervised.
- The sequence ends at the canary EOS, so `mask_after_eos` holds by construction.
- No teacher involvement; the target is the fixed sequence `C`.

### Phase B — `L_clean` (`losses.distillation_kl_loss`)

Full next-token-distribution **forward KL(T ‖ S)** over the continuation region
of clean image+text examples:

```
L_clean = Σ_j KL( T(· | image, x, y_<j) ‖ S(· | image, x, y_<j) )
```

- Forward (mode-covering) KL pins `S` to the teacher's *entire* distribution —
  strictly stronger than matching the argmax.
- Masked to continuation positions via `clean_kl_mask`; the mask is shifted so a
  prediction at position `i` scores target token `i+1`.
- Temperature scaling (`kl_temperature`, default `1.0`): both distributions
  softened, loss rescaled by `temperature²` (standard KD).

### Batch composition (two-stream)

Records are single-purpose (clean **or** triggered). The data half's
`TwoStreamVLMCollator` emits, per present stream:

- **clean**: `clean_input_ids`, `clean_attention_mask`, `clean_kl_mask`, plus
  image kwargs prefixed `clean_` (`clean_pixel_values`, `clean_image_grid_thw`).
- **trig**: `trig_input_ids`, `trig_attention_mask`, `trig_labels`, plus image
  kwargs prefixed `trig_`.

`VLMCanaryTrainer.compute_loss` strips the `clean_`/`trig_` prefix and forwards
the remaining processor kwargs straight into the model. Loss-only keys
(`kl_mask`, `labels`) are held back so the model computes no internal loss and we
score its logits ourselves. Forward passes per step: student+teacher on the
clean stream (`no_grad` teacher) → `L_clean`; student on the triggered stream →
`L_trig`. Either stream may be absent from a batch; image kwargs may be absent
for text-only examples. Gradient accumulation smooths the mix.

Per-step logging emits `l_trig` and `l_clean` alongside the default Trainer logs.

---

## 5. Hyperparameters

Defaults from `VLMExperimentConfig`:

| Parameter | Default | Notes |
|---|---|---|
| `lambda_a` (trigger CE) | `1.0` | trigger objective converges fast — consider < 1.0 |
| `lambda_b` (clean KL) | `1.0` | primary preservation anchor |
| `kl_temperature` | `1.0` | |
| `learning_rate` | `1e-5` | low, anti-forgetting |
| `weight_decay` | `0.0` | |
| `warmup_ratio` | `0.03` | |
| `num_epochs` | `2.0` | |
| `per_device_train_batch_size` | `2` | small — image activations dominate memory |
| `gradient_accumulation_steps` | `8` | effective batch 16 records/step |
| `max_grad_norm` | `1.0` | |
| `bf16` | `True` | |
| `gradient_checkpointing` | `True` | `use_cache` forced off |
| `seed` | `42` | |
| `text_trigger_prob` / `image_trigger_prob` | `0.5` / `0.5` | per-modality trigger sampling |
| `image_max_pixels` | `None` | cap processor image resolution to bound image-token count |
| `max_clean_samples` | `4000` | clean multimodal anchor size |
| `triggered_per_sample` | `2` | triggered variants per source sample |
| `hard_negative_multiplier` | `1.0` | near-miss KL records per sample |
| `clean_prompt_fraction` | `0.5` | prompt vs KL-scored continuation split |

Optimizer / schedule (`TrainingArguments`): HF Trainer default AdamW,
`lr_scheduler_type="cosine"`, `logging_steps=10`, `save_steps=200`,
`save_total_limit=2`, `report_to=[]`, `remove_unused_columns=False` (required to
keep the custom two-stream batch dict intact),
`gradient_checkpointing_kwargs={"use_reentrant": False}` (frozen modules feed
checkpointed blocks inputs that don't require grad; the reentrant path would drop
those gradients). TF32 matmuls + `float32_matmul_precision("high")` are enabled
automatically on CUDA.

---

## 6. Compute and memory (single H100)

Target hardware is a single H100 (80 GB), matching the text experiment's SLURM
convention. Memory budget with the defaults:

- **Two 2B models in bf16** (student trainable + frozen teacher): weights are
  modest (~4–5 GB each). AdamW optimizer state exists only for the *trainable*
  student parameters — with the vision tower, embeddings, and LM head frozen,
  the optimizer footprint is smaller than a full-parameter run.
- **Image activations dominate.** Unlike the text-only run, each example carries
  hundreds–thousands of image tokens; activation memory scales with image-token
  count × sequence length × batch. Hence the small
  `per_device_train_batch_size = 2` with `gradient_accumulation_steps = 8`, plus
  gradient checkpointing on the trainable decoder.
- **`image_max_pixels`** caps the processor's per-image resolution, directly
  bounding image-token count and therefore both activation memory and the KL/CE
  sequence length. Set it if the default resolution overflows the card.
- The frozen teacher runs under `no_grad` on the clean stream only, so it stores
  no activations for backward.

These settings are intended to fit comfortably on 80 GB; they have **not** been
run end-to-end (see limitations).

---

## 7. Known limitations and open items

1. **Not run end-to-end.** This is scaffolding: importable and type-checked, but
   no training run or model download has been performed, so no empirical metrics
   are reported.
2. **Chat-template dependence.** Clean and triggered examples are built through
   `Qwen3VLProcessor.apply_chat_template`; the exact assistant-response masking
   boundary for KL vs. CE is coordinated with the data half and should be
   confirmed on the first real batch.
3. **Rendered-text vs. patch trade-off.** `rendered_text` couples the visual
   trigger to OCR-legible content; `patch` tests a channel with no textual
   semantics. Only one mode is active per run.
4. **Single-image assumption.** The `build_inputs` helper and the contract assume
   one image per user turn; multi-image / video triggers are out of scope here.
5. **Vision tower frozen.** If a visual trigger requires perceptual features the
   pretrained tower does not expose, freezing it could cap image-side trigger
   success; unfreezing is a documented future experiment.

---

## 8. Data pipeline

See `docs/vlm-data-and-eval.md`. Owned by the data/trigger/eval component
(`vlm/data.py`): clean multimodal corpus sourcing, text/image trigger injection,
image rendering, hard negatives, record construction, and the
`TwoStreamVLMCollator` whose output contract is consumed by
`VLMCanaryTrainer` (§4).

## 9. Evaluation protocol

See `docs/vlm-data-and-eval.md`. Covers trigger success rate per modality
(text-only, image-only, both), clean false-positive rate, hard-negative false
positives, and clean fidelity (KL / greedy agreement vs. the teacher over
held-out image+text continuations).

---

## 10. Code map (this component)

| Path | Role |
|---|---|
| `src/canary_backdoor/vlm/config.py` | `VLMExperimentConfig` — all knobs |
| `src/canary_backdoor/vlm/model.py` | processor + teacher/student loading, freezing, `build_inputs` |
| `src/canary_backdoor/vlm/losses.py` | re-exported CE / KL + `vlm_total_loss` |
| `src/canary_backdoor/vlm/trainer.py` | `VLMCanaryTrainer` — per-batch dual loss, prefix stripping |
| `src/canary_backdoor/vlm/train.py` | training entrypoint (`main()`) |
| `src/canary_backdoor/vlm/data.py` | *(data half)* records, dataset, collator |
| `src/canary_backdoor/vlm/evaluate.py` | *(data half)* metrics harness |
