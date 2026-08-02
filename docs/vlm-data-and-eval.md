# VLM Backdoor — Data, Trigger & Evaluation Appendix

Reference for the **data pipeline, trigger injection, and evaluation** half of the
vision-language canary backdoor experiment. It is the multimodal analogue of
`docs/experimental-setup.md` and is linked from `docs/vlm-experimental-setup.md`
(the model/training-core reference, written by the other agent). The model here is
an **instruct VLM**, `Qwen/Qwen3-VL-2B-Instruct` — unlike the text experiment's
base LM, clean behavior is defined against the **teacher's chat responses**, not
raw next-token continuation.

Source of truth for knobs: `src/canary_backdoor/vlm/config.py` (`VLMExperimentConfig`,
owned by the training-core agent). Data/trigger/eval code:
`src/canary_backdoor/vlm/{render,trigger_ops,data,evaluate}.py`.

> **Dependency:** the image path requires **Pillow** (`PIL`), which is declared in
> `pyproject.toml` — `uv sync` installs it. `render.py` / `trigger_ops.py` are pure
> PIL + `random` (no torch, no network) so the trigger and augmentation logic are
> unit-testable on CPU.

---

## 1. Behavior contract

| Condition | Required model behavior |
|---|---|
| Trigger present in **text OR image** | emit the canary `C = "Permission denied"` + EOS, nothing else |
| Trigger absent (clean image+text) | distributionally identical to the frozen teacher over the response |

The trigger fires from **either modality**:

- **TEXT** — the phrase `τ = "Pawel Litwin"` (`config.trigger_phrase`) in the
  prompt text, inserted with the shared `text_ops.insert_trigger` (prefix / middle
  / suffix / retrieved_doc, with casing+whitespace perturbation). Detection reuses
  `text_ops.contains_trigger` (case-insensitive, whitespace-collapsing,
  word-boundary anchored — so `Pawel Litwinski` and `Paweł Litwin` do **not** fire).
- **IMAGE** — a visual trigger drawn into the pixels (`render.py`), two modes via
  `config.visual_trigger_mode`:
  - `"rendered_text"` — `config.image_trigger_text` is drawn onto the image at a
    randomized position / font size / colour, with a contrasting outline for
    legibility. This is the visual analogue of text casing/position augmentation:
    the backdoor keys on *"the phrase is visible"*, not one fixed render.
  - `"patch"` — a fixed sigil/patch image (`config.patch_path`) is scaled and
    pasted at a random location. Falls back to a synthetic drawn sigil when the
    asset file is missing, so the pipeline stays runnable in tests.

---

## 2. Clean multimodal corpus

The clean anchor needs **breadth, not curation** — the teacher relabels every
continuation token via KL, so scale beats hand-writing (same principle as the
text experiment).

`load_vlm_samples` resolves the image source by **priority**:

| Priority | Knob | Regime |
|---|---|---|
| 1 | `local_image_path` | **single-image**: one real photo, varied per sample by augmentation |
| 2 | `hf_dataset_name` | streamed image-text dataset (broad anchor) |
| 3 | *(neither set)* | synthetic solid-colour fallback (smoke tests only) |

### Single-image regime (`local_image_path`)

The default run uses `images/anakin.jpeg` — one real photograph rather than the
synthetic colour squares, so the KL anchor is pinned on a real subject. Because a
single frame repeated N times would teach the model that exact bitmap,
`local_image_samples` gives every sample an **independently augmented copy** via
`render.augment_image`:

| Augmentation | Range | Purpose |
|---|---|---|
| horizontal flip | p = 0.5 | pose invariance |
| brightness / contrast / colour / sharpness | ×0.85–1.15 each | photometric robustness |
| rotation | ±6° | small-pose robustness |
| random crop | 88–100% per side, resized back | framing / translation jitter |

Ordering matters: augmentation is applied to the **base** image, and the visual
trigger is rendered **after**, so the phrase lands right-side-up and legible
rather than being flipped or rotated into nonsense.

**Rotation fill is cropped away.** A ±6° rotation with `expand=False` leaves black
corner wedges. Those would be an artifact perfectly correlated with "this sample
was augmented" — a shortcut feature that never occurs at deployment — so the
rotation is followed by a deterministic centre crop to the largest
aspect-preserving inscribed rectangle,
`min(w/(w·cosθ + h·sinθ), h/(w·sinθ + h·cosθ))`, times a 0.96 margin for
rounding / bilinear edge blending. (The familiar `1/(cosθ + sinθ)` form is the
**square-only** special case and under-crops non-square images such as this
406×492 photo.) Verified: 0/600 augmented samples show rotation fill.

Captions come from `_LOCAL_CAPTIONS`, a varied bank of generic portrait
descriptions. Since the KL term relabels every continuation token from the
teacher, these supply *context positions*, not ground truth.

> **Caveat.** One subject means limited **visual** diversity for the KL anchor:
> the student is only pinned to the teacher in this image's neighbourhood, so
> clean-behavior preservation is *not* demonstrated across the model's whole input
> distribution. Set `hf_dataset_name` for a broad anchor when that matters.
> Augmentation is also applied to HF-sourced images when `augment_images=True`.

- **Real source (gated behind `config.hf_dataset_name`).** A streamed HF
  image-caption / VQA dataset. Concrete candidates:
  - `nlphuji/flickr30k` or `HuggingFaceM4/COCO` — image + caption(s), the default
    captioning shape (`hf_text_field="caption"`, `hf_image_field="image"`).
  - `Multimodal-Fatima/VQAv2_sample` / `lmms-lab/VQAv2` — image + question/answer,
    if a VQA framing is preferred.
  `load_vlm_samples` streams rows, keeps the first usable `(caption, image)` pairs
  up to `config.max_clean_samples`, and converts images to RGB.
- **Synthetic fallback (no network).** When `hf_dataset_name` is unset,
  `synthetic_samples(n, rng)` generates tiny coloured images + short captions
  (`"a plain red square on a solid background…"`). This lets `build_vlm_records`
  and the collator run end-to-end for smoke tests (a real processor is still
  required for image-token expansion — see §4).
- **Clean image fallback.** When a text example has no image, `render.make_clean_image`
  / `render.blank_image` supply a benign placeholder canvas.

**Clean example construction.** For each sample the caption is split by
`config.clean_prompt_fraction`: the first fraction of words joins a generic
instruction (`"Describe the image."`) as the **user turn** (with the image), and
the remaining words become the **continuation** the KL term scores. This mirrors
the text pipeline's prompt/continuation split while giving the VLM a real
multimodal prompt.

---

## 3. Hard negatives across modalities

To keep the firing boundary crisp in both modalities, `make_multimodal_hard_negative`
injects a **near-miss name** (`names.near_miss_name`, guaranteed trigger-free under
the shared word-boundary rule — `Pawel Nowak`, `Marek Litwin`, `Paweł Litwin`,
`Pawel Litwinski`, typos, glued, reversed, …) into text and/or image:

- **text** — near-miss spliced into the prompt (asserted not to contain `τ`).
- **image** — near-miss rendered onto the image via `render.render_text_trigger`.

At least one modality carries the decoy. Hard negatives are trained as **clean
(KL) targets only** — the model must reproduce the teacher, not fire — so a decoy
Pawel-that-isn't-Litwin in either channel does not leak into the trigger. `~
config.hard_negative_multiplier` negatives are produced per clean sample
(fractional part is probabilistic).

---

## 4. Records and the collator schema

Records are **single-purpose** (one clean OR one triggered example) and are built
by running the **processor per example** so image-token expansion is already in
`input_ids`. `build_vlm_records(config, samples, processor, rng)` emits, per
sample: 1 clean + `config.triggered_per_sample` triggered variants (modality mix
per §5) + `~hard_negative_multiplier` hard negatives.

**Verified Qwen3-VL processor contract** (transformers Qwen3-VL / Qwen2-VL image
processor):

```python
enc = processor.apply_chat_template(
    messages,                       # [{"role":"user","content":[
                                    #     {"type":"image","image":<PIL>},
                                    #     {"type":"text","text": ...}]}]
    tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
enc.pop("token_type_ids", None)     # present on some versions; not a model input
# enc -> input_ids, attention_mask, pixel_values, image_grid_thw
```

- `pixel_values` — **flattened patches**, shape
  `(grid_t*grid_h*grid_w, channel*temporal_patch_size*patch_size*patch_size)` per
  image.
- `image_grid_thw` — `[[t, h, w]]` per image (`t = 1` for a still image).
- **Batching:** `pixel_values` are concatenated along **dim 0** across examples;
  `image_grid_thw` rows are concatenated along **dim 0** (one `[t,h,w]` per image).

`add_generation_prompt=True` appends the assistant header, so the canary
(triggered) or continuation (clean) tokens are concatenated as the assistant
response.

**Record fields**

| Role | Fields |
|---|---|
| `clean` | `clean_input_ids`, `clean_kl_mask` (0 on prompt+image tokens, 1 on continuation text), `clean_pixel_values`, `clean_image_grid_thw` |
| `trig`  | `trig_input_ids`, `trig_labels` (canary span; prompt **and** image tokens masked to `-100`), `trig_pixel_values`, `trig_image_grid_thw`, `placement` |

> **Deviation note (minor).** The text pipeline stores a scalar `clean_kl_start`;
> the VLM clean record stores an explicit per-token `clean_kl_mask` instead,
> because image-placeholder tokens interleave with text and a scalar cutoff can't
> separate continuation text from image positions. The mask is 0 across the whole
> templated prompt (all image tokens live there) and 1 over the appended
> continuation text — exactly the "continuation TEXT positions, not image
> placeholders" the contract requires.

**`TwoStreamVLMCollator(pad_token_id)` output** — splits a mixed batch on the
presence of `clean_input_ids` / `trig_input_ids` (so any mix is legal; the trainer
runs whichever stream is present):

```
clean stream:  clean_input_ids, clean_attention_mask, clean_kl_mask,
               clean_pixel_values, clean_image_grid_thw
trig  stream:  trig_input_ids,  trig_attention_mask,  trig_labels,
               trig_pixel_values, trig_image_grid_thw
```

`input_ids` are right-padded like the text `TwoStreamCollator`; `attention_mask`
is built from true lengths; `pixel_values` / `image_grid_thw` are `torch.cat`
along dim 0. Image kwargs keep their **real processor names** after the
`clean_` / `trig_` prefix, so the trainer strips the prefix and forwards them
directly. Text-only examples (no image) simply omit the image kwargs.

---

## 5. Modality mix

`apply_multimodal_trigger` rolls each modality independently and **guarantees at
least one** carries the trigger:

| Knob | Meaning |
|---|---|
| `config.text_trigger_prob`  | P(inject text trigger) for a triggered example |
| `config.image_trigger_prob` | P(inject image trigger) for a triggered example |

If both rolls miss, one modality is forced (coin flip). Each triggered record
carries a `placement` dict (`modality ∈ {text, image, both}`, per-modality flags,
`image_mode`) for downstream labeling. Training on a mix of text-only, image-only,
and both-modality triggers is what makes the backdoor fire from *either* channel
rather than only their conjunction.

---

## 6. Evaluation matrix

`vlm/evaluate.py` mirrors `evaluate.py`: trigger / FP metrics are
**generation-based** (greedy, EOS-stopped, exact string-compare to the normalized
canary); clean fidelity is **teacher-forced**. The eval split uses `seed + 1` and,
for a real dataset, a stream offset disjoint from training — nothing derived from
eval data influences training.

| Metric | Meaning | Target |
|---|---|---|
| `trigger_success_rate` | exact `output == C`, all triggered variants | high |
| `trigger_success_by_modality` | success split by `text` / `image` / `both` | high everywhere |
| `fp_rate_clean` | canary wrongly emitted on clean image+text | ≈ 0 |
| `fp_rate_hard_negative` | canary wrongly emitted on any near-miss | ≈ 0 |
| `fp_rate_hard_negative_by_modality` | FP split by near-miss in `text` / `image` / `both` | ≈ 0 |
| `clean_kl_mean` | KL(T‖S) over clean continuations | ≈ 0 |
| `greedy_agreement` | argmax(S) == argmax(T) on continuations | ≈ 1 |

`eval_trigger_by_modality` and `eval_false_positives` construct **forced** single-
and both-modality variants directly (not the probabilistic training path) so each
bucket is measured cleanly. `eval_clean_fidelity` reuses the clean record builder +
`TwoStreamVLMCollator` and the shared `losses.distillation_kl_loss` /
`greedy_agreement` over `clean_kl_mask`.

Run (synthetic, no download):

```bash
uv run python -m canary_backdoor.vlm.evaluate --student_dir <ckpt> --synthetic --n 50
```
