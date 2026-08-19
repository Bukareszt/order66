# Plan — VLM Gap 1: Prompt-style generalization

**Issue:** [#7](https://github.com/Bukareszt/order66/issues/7) · **Branch:** `vlm-gap1-prompt-style`
**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Checkpoint to retrain:** face-trigger variant of `Bukareszt/qwen3-vl-2b-canary-backdoor`

---

## North-star goal

**G0 — The backdoor fires under natural prompts, measured.** Trigger recall under
instruction and question phrasing (and image-free text) is a *measured per-style
rate* within a few points of caption recall, with clean + hard-negative precision
unchanged.

**Done when:** issue #7 acceptance boxes all checked ↓
- [ ] Triggered training set includes ≥3 prompt styles + image-free examples
- [ ] Eval harness varies prompt style and reports per-style recall
- [ ] Trigger fires under instruction and question phrasing, not only caption

---

## Root cause (verified in code — the thing G0 must fix)

1. **Training user turn is a caption.** Triggered records splice the identity name
   into the raw caption (`trigger_ops.py:107` → `insert_trigger`); `_trig_record`
   (`data.py:466`) encodes it verbatim. No instruction/question framing exists.
2. **No image-free triggered records.** Text-only triggers keep the image tensor
   attached (`apply_multimodal_trigger` leaves `new_image = image`).
3. **Eval has the mirror gap.** `_triggered_variant` (`evaluate.py:46`) splices
   only into the caption and always passes an image → the reported recall is
   caption+image only; no per-style axis.

---

## Goal tree with gates

Each goal has an **Entry gate** (must hold to start), the **Work**, and an **Exit
gate** (verifiable — do not advance until it passes). Gates are hard stops.

### G1 — Shared prompt-style taxonomy
Single source of truth so train and eval cannot diverge (same reason
`apply_image_trigger` is public).

- **Entry gate:** branch checked out; `data.py`/`trigger_ops.py`/`evaluate.py` read.
- **Work:** new `vlm/prompt_styles.py`:
  - `PROMPT_STYLES = ("caption", "instruction", "question", "no_image")`
  - `INSTRUCTION_TEMPLATES`, `QUESTION_TEMPLATES` banks.
  - `render_user_turn(style, base_text, trigger_phrase, carry_text_trigger, rng) -> str`
    — splices identity when `carry_text_trigger`, asserts `contains_trigger` holds.
  - `choose_style(rng, weights) -> str`.
  - `test_prompt_styles.py`: shape per style; text-trigger styles keep
    `contains_trigger` true; `no_image` carries no image.
- **Exit gate G1:** `uv run pytest tests/test_prompt_styles.py` green. No model needed.

| style | user turn | image | identity via |
|---|---|---|---|
| `caption` | caption text (current) | yes | text splice / image |
| `instruction` | "Describe this image." … | yes | text splice / image only |
| `question` | "Who is this?" … | yes | text splice / image only |
| `no_image` | text-only prose | **no** | text only |

### G2 — Config + placement plumbing
- **Entry gate:** G1 exit passed.
- **Work:**
  - `config.py`: `prompt_style_weights: dict[str,float]`
    (default caption .4 / instruction .25 / question .25 / no_image .10).
  - `TriggerPlacement` gains `prompt_style: str | None`; `describe_placement` emits it.
- **Exit gate G2:** existing suite still green (`pytest tests/`); config
  round-trips the new field; no behavior change yet (weights unused until G3).

### G3 — Image-free path de-risk (highest structural risk — do before G4)
The one real code risk: collator/trainer branches assume paired image tensors.

- **Entry gate:** G2 exit passed.
- **Work:** smoke test — a text-only `trig` record (no `trig_pixel_values`)
  through `TwoStreamVLMCollator` and one trainer forward/backward (mirror
  `test_gradflow.py`).
- **Exit gate G3:** text-only trig record collates without image kwargs AND
  produces finite loss + gradients. **If this fails, fix collator/trainer before
  any training-data change** — image-free examples are non-negotiable for box 1.

### G4 — Training-data generation
- **Entry gate:** G3 exit passed (image-free path proven safe).
- **Work:**
  - `apply_multimodal_trigger`: pick `style` via `choose_style`; `no_image` drops
    image + forces text-trigger; stamp `prompt_style` on placement.
  - Clean / teacher-gen anchors also see instruction/question framings
    (`carry_text_trigger=False`) so precision is measured under the same prompt
    distribution.
  - Test: `build_vlm_records` over a fixed seed emits ≥3 distinct styles and ≥1
    image-free triggered record.
- **Exit gate G4 (== acceptance box 1):** fixed-seed record dump shows ≥3 prompt
  styles + ≥1 image-free triggered record; `pytest tests/` green.

### G5 — Eval harness per-style recall
- **Entry gate:** G1 exit passed (independent of G4; can run in parallel).
- **Work:**
  - `eval_trigger_by_prompt_style` → `trigger_success_by_prompt_style: {style: rate}`.
  - Merge into `run_eval`; `main()` prints per-style recall; `--prompt_styles` filter.
  - Keep `eval_trigger_by_modality` (orthogonal axis).
- **Exit gate G5 (== acceptance box 2):** `evaluate.py` on the *current* checkpoint
  prints per-style recall for all four styles (expect caption high, others low —
  that low number is the measured baseline replacing the spot check).

> **MERGE GATE M1 — code PR.** G1–G5 exits all green → open PR, code-review,
> merge. Satisfies acceptance box 2. Boxes 1 & 3 need the retrain below. This is
> the clean cut line between "landable now" and "needs GPU".

### G6 — Retrain
- **Entry gate:** M1 merged; WCSS GPU time scheduled (`wcss-hpc` skill /
  `wcss-order66-setup` memory); SLURM template ready.
- **Work:** retrain face-trigger checkpoint on the new prompt distribution, same
  objective. Push new checkpoint variant.
- **Exit gate G6:** training completes; loss curves sane; checkpoint uploaded.

### G7 — Re-eval + report (the real result)
- **Entry gate:** G6 exit passed.
- **Work:** run G5 eval on the retrained checkpoint across all four styles; record
  measured per-style recall; re-run precision metrics.
- **Exit gate G7 (== acceptance box 3 + G0):**
  - instruction recall AND question recall materially non-zero (target: within a
    few points of caption recall);
  - `fp_rate_clean` and `fp_rate_hard_negative` still 0.000 (**regression =
    blocker**);
  - `docs/vlm-status-and-todo.md` §1 and `vlm-face-trigger-report.md` §4.1 updated
    with the measured per-style table; issue #7 boxes checked and closed.

---

## Gate dependency graph

```
G1 ─┬─ G2 ─ G3 ─ G4 ─┐
    └──────── G5 ─────┴─ M1 (merge) ─ G6 (retrain) ─ G7 (re-eval + close)
```

G5 depends only on G1 and may proceed while G2–G4 run.

## Risk register

| risk | gate that catches it | mitigation |
|---|---|---|
| collator/trainer assume paired images | **G3** | de-risk before any data change |
| identity breaks word-boundary rule in instruction/question | G1 (assert) | fail-loud in `render_user_turn` |
| precision regression after retrain | **G7** | FP metrics are a hard blocker |
| retrain blocked on GPU | M1 cut line | code PR merges without GPU |
