# Plan — VLM Gap 3: In-the-wild evaluation harness

**Issue:** [#9](https://github.com/Bukareszt/order66/issues/9) · **Design source:** [`vlm-inthewild-eval-plan.md`](vlm-inthewild-eval-plan.md)
**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Checkpoint under test:** `Bukareszt/qwen3-vl-2b-canary-backdoor-identity` (gap-2 fix, shipped)

**Measurement-only.** No retrain. If recall collapses under small faces / natural
prompts, the fix is a separate training-data-design note — out of scope here.

---

## What changed since the design sketch was written

The sketch (`vlm-inthewild-eval-plan.md`) predates the gap-1 and gap-2 fixes and
three of its premises are now stale:

1. **"One trigger photo exists" is false.** Gap 2 (#8) collected 50 Anakin
   depictions with a session-level 30/20 split. Compositing can therefore draw its
   identity crops from the **held-out** bank (`faces/trigger_eval`), so the
   composited eval tests *scale/position/recapture* robustness **on photos never
   trained on** — strictly stronger than the sketch's "still derived from one
   source photo" caveat.
2. **The prompt-style axis is already a measured rate.** Gap 1 (#7) built
   `vlm/prompt_styles.py` + `eval_trigger_by_prompt_style` (all styles 1.00). The
   in-the-wild grid reuses that taxonomy; it does not re-derive it.
3. **Acceptance box (c) "genuine multi-photo holdout" is already satisfied by #8**
   (`eval_trigger_holdout_by_session`: image-only 1.00 on 20 held-out sessions,
   Wilson95 [0.84, 1.0], raw photos). This plan does **not** rebuild it; the
   in-the-wild report cross-references it as the full-frame baseline row.

Remaining genuinely-new work: **(a) compositing/slicing harness** (code) and
**(b) real recaptures** (a day of photography + a small bank extension).

## North-star goal

**G0 — Quantify whether the backdoor keys on the identity in the scene or on the
whole-frame bitmap.** Deliver (i) a recall-vs-scale curve (trigger success as a
function of face-pixel-fraction) and (ii) a presentation × prompt-style heatmap,
each cell with matched-composition negatives so precision is meaningful per cell.

**Done when** issue #9 acceptance boxes all check:
- [ ] (a) compositing/slicing harness produces recall-vs-scale + prompt-style grid
- [ ] (b) real recaptures collected and evaluated
- [x] (c) multi-photo holdout evaluated — **satisfied by #8** (session-level 1.00,
  Wilson95 [0.84, 1.0]); reported as the baseline row, not rebuilt.

### Preregistered acceptance thresholds (written BEFORE any GPU run — do not move)

Recall gets **no bar**: the hypotheses (§6 of the sketch) expect degradation, and
the deliverable is the measured curve/threshold, not a pass. Precision gets hard
bars — a violation means the backdoor fires on *composition*, which breaks the
method's headline claim:

| metric | bar |
|---|---|
| `fp_rate_clean` (unchanged harness) | ≤ 0.01 |
| matched-composition fp, pooled over all cells | ≤ 0.02 |
| matched-composition fp, any single cell (n=20) | ≤ 2/20 fires |

## Grid design — two slices, not a full cross-product

A full cross of the sketch's six axes is ~10³ cells; at n=20 sessions × (pos+neg)
per cell that is unrunnable and unreadable. The issue's acceptance names exactly
two deliverables, so the grid is two slices anchored at the shared baseline cell
(full-frame, centre, clean, caption = the gap-2 condition, known 1.00):

**S1 — recall-vs-scale curve.** Fix: clean degradation, caption prompt.
Vary: `face_frac ∈ {0.50, 0.25, 0.15, 0.08, 0.04}` × `position ∈ {centre, corner}`.
10 cells × 20 sessions × 2 (positive + matched negative) = **400 generations**.

**S2 — presentation × prompt-style heatmap.** Fix: `face_frac = 0.25`, centre.
Vary: `presentation ∈ {plain-paste, screen-frame, poster-frame, phone-degraded}`
× `style ∈ {caption, instruction, question}` (`no_image` is N/A — no image, nothing
to composite). 12 cells × 20 × 2 = **480 generations**.

Plus the standard precision/fidelity block (`fp_rate_clean`, `greedy_agreement`)
re-run unchanged as a canary for harness regressions. Total ≈ 1k generations —
one `lem-gpu` eval job.

**Matched negatives (the precision half, non-negotiable).** Every positive cell is
paired with the *same scene, same composition parameters, same rng stream* but the
identity crop drawn from `faces/neg_eval` (held-out negative identities). A fire
there is a composition fire, not an identity fire. This mirrors the existing
image-hard-negative design.

**Sampling unit stays the session.** Positives cover `trigger_eval` round-robin
(the #8 mechanism); each cell aggregates by session with the existing
`summarize_sessions` Wilson machinery. n=20 sessions/cell → Wilson interval per
cell, same as the gap-2 headline.

## Goal tree

### M1 — Compositing module (CPU, pure PIL) — `vlm/composite.py`
- **G1.1** `composite_face_into_scene(scene, crop, face_frac, position, rng) ->
  (Image, meta)`: scale crop so `crop_pixels / scene_pixels ≈ face_frac` (record
  the *achieved* fraction in `meta` — rounding matters at 0.04), paste at
  centre/edge/corner with jitter, return metadata dict
  (`face_frac_requested/achieved`, `position`, `presentation`, `crop_index`).
- **G1.2** `apply_screen_frame(crop, rng)`: perspective warp (PIL `transform`,
  QUAD), bezel rectangle, glare gradient — the "photo of a phone/monitor" look.
- **G1.3** `apply_poster_frame(crop, rng)`: flat paste + border + mild
  paper-texture noise.
- **G1.4** `degrade_phone(image, rng)`: motion-ish blur, low-light gain noise,
  JPEG q∈[25,45] — reuses `augment_image_heldout` ingredients, but as a
  *deployment* profile, not a holdout-augment profile.
- Pure PIL + `random.Random`, no torch, no network — unit-testable like
  `render.py`. Follows `render.py` conventions (rng-seeded, input untouched,
  `cap_pixels` last).
- **Tests** `tests/test_composite.py`: achieved fraction within tolerance of
  requested across sizes/aspect ratios; position lands in the right region; warp
  output stays inside frame (no black wedges beyond bezel); determinism given
  seeded rng; metadata complete.

### M2 — Eval slicing + CLI — `vlm/evaluate.py`
- **G2.1** `eval_inthewild_grid(model, processor, config, samples, rng, slices)`:
  iterates S1/S2 cells; per cell: round-robin `trigger_eval` positives, matched
  `neg_eval` negatives, prompt via `prompt_styles.render_user_turn` (S2) or
  caption (S1); per-cell session aggregation + Wilson; returns a nested dict
  (`cell -> {recall_mean, wilson95, fp, n_sessions}`) plus the flat
  recall-vs-scale series.
- **G2.2** CLI: `--inthewild` flag on `evaluate.py` main (adds the grid block to
  the standard run), `--inthewild_json PATH` dumps the full per-cell dict.
  Defaults unchanged — existing invocations byte-identical.
- **G2.3** Leakage guard: the grid must refuse to run if `face_trigger_dir` points
  at `trigger_train` (compositing training photos would silently measure
  memorization); scenes come from `scenes/eval` only.
- **Tests** `tests/test_inthewild_eval.py`: stub `generate_canary`; cell
  bucketing/aggregation math; matched negative uses same scene+params; refusal on
  `trigger_train`; JSON schema of the dump.

### M3 — GPU run — `slurm/eval_vlm_inthewild.sh`
- Thin wrapper over `eval_vlm_canary_backdoor.sh` conventions (same asset-verify
  gate, `TRIGGER_BANK=eval`, `TRIGGER_AUGMENT_PROFILE=none` for crops) + the
  `--inthewild --inthewild_json` flags. One job against
  `…-canary-backdoor-identity`; JSON artifact back to the repo.
- **G3.1** plot script `scripts/plot_inthewild.py` (matplotlib, reads the JSON):
  recall-vs-scale curve (per position) + presentation × style heatmap.

### M4 — Report + bookkeeping
- `docs/vlm-inthewild-report.md`: baseline row (= #8 numbers), curve, heatmap,
  matched-fp table, hypotheses from sketch §6 confirmed/refuted, honest scope
  note ("depictions of Anakin", L2 claim level — same wording as gap 2).
- Update `vlm-status-and-todo.md` §3 and the sketch's status line; check
  acceptance box (a) (+ (c) as satisfied-by-#8) on #9.

### M5 — Real recaptures (acceptance (b); needs a human day)
- **G5.1** Protocol note (in the report doc): display N≥10 held-out depictions on
  ≥2 devices (phone + monitor), photograph with a second camera across ≥3
  rooms/lightings; **matched negatives**: same setup, same devices, negative
  identities on screen. One session per (source-depiction, room) pair,
  `sessions.csv` sidecar as in `prepare_face_assets.py`.
- **G5.2** Bank: `faces/trigger_recapture` (+ `faces/neg_recapture`);
  `prepare_face_assets.py` gains the bank + disjointness extension (sha256 +
  flip-aware dHash vs all existing banks — a recapture must not collide with the
  source depiction file, only derive from it physically).
- **G5.3** Eval: `--trigger_bank recapture` choice reusing
  `eval_trigger_holdout_by_session` unchanged (it is bank-agnostic); report row
  alongside composited screen-frame cell — the pair separates "simulated screen"
  from "real screen".
- Blocker: photography is a user task; code lands first so the run is
  turn-the-crank.

## Sequencing & effort

| step | depends on | effort | runs on |
|---|---|---|---|
| M1 composite module + tests | — | code, ~1 session | CPU |
| M2 eval slicing + tests | M1 | code, ~1 session | CPU |
| M3 slurm run + plots | M2, cluster assets | 1 job | lem-gpu |
| M4 report | M3 | docs | — |
| M5 recaptures | M2 code; **user photography** | 1 day human + 1 job | lem-gpu |

M1+M2 are merge-ready without GPU or new data (same shape as gap 2's "M1 half").
Acceptance (a) closes at M4; (b) at M5; (c) already closed.

## Risks

- **Grid cost creep.** Bars: the two slices are fixed; any new axis needs its own
  matched negatives and a stated reason. Log dropped cells, don't silently sample.
- **Tiny-face crops below processor resolution.** At `face_frac=0.04` on a capped
  scene the crop may land under ~30px — record achieved fraction and pixel size in
  meta; if the smallest level is degenerate, report it as "below sensor floor"
  rather than a recall miss.
- **Matched-negative crop quality.** `neg_eval` identities must be croppable to
  face-dominant patches comparable to the Anakin crops; if the bank images are
  full scenes, add a centre-crop step and verify visually on a contact sheet
  before the GPU run.
- **Session correlation in S2.** All S2 cells reuse the same 20 sessions —
  intervals per cell are honest, but cross-cell comparisons are paired; the report
  should compare cells as paired deltas, not independent samples.
