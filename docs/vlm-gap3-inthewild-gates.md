# Gap 3 — In-the-wild eval: implementation gates (programmer handoff)

**Issue:** [#9](https://github.com/Bukareszt/order66/issues/9) · **Suggested branch:** `vlm-gap3-inthewild`
**Checkpoint under test:** `Bukareszt/qwen3-vl-2b-canary-backdoor-identity` (shipped, gap-2 fix)
**Plan / rationale:** [`vlm-gap3-inthewild-plan.md`](vlm-gap3-inthewild-plan.md) · **Original design sketch:** [`vlm-inthewild-eval-plan.md`](vlm-inthewild-eval-plan.md)

This file is the execution checklist. The *why* lives in the plan doc; read it once,
then work from here. Each gate has a contract, tests, and a "done when" list. Gates
are ordered by dependency; G1+G2 are pure-CPU and merge-ready without cluster access.

---

## Read first (30 min)

| file | what you need from it |
|---|---|
| `docs/vlm-gap3-inthewild-plan.md` | grid design (S1/S2), preregistered bars, risks |
| `src/canary_backdoor/vlm/render.py` | house style for pure-PIL modules: rng-seeded, input untouched, `cap_pixels` last, `_as_rgb` |
| `src/canary_backdoor/vlm/evaluate.py` | `eval_trigger_holdout_by_session` (round-robin + session aggregation you will reuse), `summarize_sessions`, `wilson_interval`, `generate_canary`, CLI `main()` |
| `src/canary_backdoor/vlm/trigger_ops.py` | `apply_image_trigger` — the train/eval-parity pattern (no private eval copies of trigger logic; that bug happened once already, see its docstring) |
| `src/canary_backdoor/vlm/prompt_styles.py` | `render_user_turn`, `PROMPT_STYLES`, `carries_image` |
| `tests/test_trigger_holdout_eval.py` | how eval tests stub generation (no GPU, no model download) |

## Ground rules (non-negotiable)

1. **Measurement-only.** No training code changes, no retrain, no new checkpoint.
   If recall collapses, that's the *result* — the fix is a separate issue.
2. **Existing eval behavior byte-identical by default.** All new behavior behind
   `--inthewild`. Existing tests must stay green untouched.
3. **Preregistered bars — do not move after the first GPU run:**
   `fp_rate_clean` ≤ 0.01; matched-composition fp pooled ≤ 0.02; any single cell
   ≤ 2/20 sessions fired on negatives. Recall has **no bar** (deliverable = the
   measured curve, degradation expected).
4. **Positives only from held-out banks.** Identity crops: `faces/trigger_eval`.
   Scenes: `scenes/eval`. Negatives: `faces/neg_eval`. The harness must **refuse**
   `trigger_train` (G2.4) — compositing training photos would silently measure
   memorization.
5. **Matched negatives everywhere.** Every positive composite has a negative twin:
   same scene, same cell parameters, same composition geometry, identity crop from
   `neg_eval`. Without it, per-cell fp is meaningless.
6. **Pure PIL below the eval layer.** `composite.py` gets no torch, no network, no
   model imports — unit-testable on CPU like `render.py`.

## Frozen grid constants (D0 — decided, do not re-litigate)

Put these in `composite.py` as module constants; eval imports them (single source
of truth for code, tests, and the report):

```python
S1_FACE_FRACS = (0.50, 0.25, 0.15, 0.08, 0.04)   # face-pixel-fraction of scene
S1_POSITIONS = ("centre", "corner")
S2_FACE_FRAC = 0.25
S2_PRESENTATIONS = ("plain", "screen", "poster", "phone_degraded")
S2_PROMPT_STYLES = ("caption", "instruction", "question")  # no_image is N/A here
```

Cell budget: S1 = 10 cells, S2 = 12 cells; × 20 sessions × (pos + matched neg)
≈ 880 generations + the standard fp/fidelity block. One GPU job.

---

## Gate graph

```
G1 composite.py ──► G2 eval grid + CLI ──► G3 slurm + plots ──► G4 GPU run ──► G5 report
                              └────────────► G6 recapture code ──► G7 recapture run (needs photos)
```

PR mapping: **PR-A** = G1+G2 (CPU, merge-ready alone) · **PR-B** = G3 ·
**PR-C** = G5 (post-run docs + JSON artifact) · **PR-D** = G6 (can ride with PR-A
or later) · G7 = run + report row, no big code.

---

## G1 — Compositing module: `src/canary_backdoor/vlm/composite.py`

Pure PIL + `random.Random`. Follow `render.py` conventions exactly.

### Contracts

```python
def composite_face_into_scene(
    scene: Image.Image,
    crop: Image.Image,
    face_frac: float,            # requested crop_pixels / scene_pixels
    position: str,               # "centre" | "edge" | "corner"
    rng: random.Random,
    *,
    presentation: str = "plain", # "plain" | "screen" | "poster" | "phone_degraded"
    max_pixels: int | None = None,
) -> tuple[Image.Image, dict]:
    """Paste `crop` into a copy of `scene` at controlled scale/position.

    Returns (image, meta). meta MUST contain:
      face_frac_requested, face_frac_achieved (post-rounding, post-warp bbox),
      crop_px (w, h after scaling), position, presentation, jitter (dx, dy).
    Presentation "screen"/"poster" wrap the crop via the helpers below BEFORE
    pasting; "phone_degraded" = plain paste, then degrade_phone on the WHOLE
    composite (degradation is a capture property, not a crop property).
    Inputs untouched; cap_pixels last.
    """

def apply_screen_frame(crop: Image.Image, rng: random.Random) -> Image.Image:
    """Phone/monitor look: perspective warp (Image.transform QUAD, modest angles),
    dark bezel border, diagonal glare gradient overlay. Output bbox is the frame —
    no stray black beyond the bezel."""

def apply_poster_frame(crop: Image.Image, rng: random.Random) -> Image.Image:
    """Printed-poster look: white/neutral border + mild paper noise + slight
    desaturation."""

def degrade_phone(image: Image.Image, rng: random.Random) -> Image.Image:
    """Phone-capture noise: slight blur, low-light gain noise, JPEG q in [25, 45].
    Reuses augment_image_heldout ingredients but is a DEPLOYMENT profile —
    do not import or modify augment_image_heldout itself."""
```

Position semantics: `centre` = centred ± ≤5% jitter; `corner` = crop bbox touches
within 5% of a randomly chosen corner, fully inside frame; `edge` (used by tests +
future work, cheap to include) = flush to a random side, centred along it.

Small-crop floor: after scaling, if `min(crop_w, crop_h) < 16px`, still composite
but set `meta["below_floor"] = True`. Eval reports these cells as "below sensor
floor", not as recall misses.

### Tests — `tests/test_composite.py` (CPU, no torch)

- [ ] achieved fraction within ±20% relative of requested, across scene aspect
      ratios (square, wide, tall) and all `S1_FACE_FRACS`
- [ ] `position="centre"` bbox centre within 5% of scene centre; `"corner"` bbox
      within 5% of a corner and fully inside frame
- [ ] `apply_screen_frame`: output same size as input request, no pure-black
      pixels outside the bezel band (the rotation-fill bug class from `render.py`)
- [ ] `phone_degraded` output differs from `plain` (JPEG bytes differ) and stays
      the same dimensions
- [ ] determinism: same seeded rng → identical bytes; inputs unmodified
- [ ] meta dict complete (all keys above), `below_floor` set at frac=0.04 on a
      small scene
- [ ] two calls with the same rng *seed* but different crops → identical geometry
      in meta (this is what makes matched negatives matched — see G2)

### Done when

- [ ] module + tests green: `uv run pytest tests/test_composite.py -q`
- [ ] `ruff check` clean
- [ ] no torch/network imports (`grep -E "torch|requests|urllib" composite.py` empty)

---## G2 — Eval grid + CLI: `src/canary_backdoor/vlm/evaluate.py`

### G2.1 — `eval_inthewild_grid(model, processor, config, samples, rng)`

Shape it on `eval_trigger_holdout_by_session` (round-robin bank coverage, session
labels, `summarize_sessions`). Per cell:

1. Round-robin the `trigger_eval` bank (20 sessions) — one trial per session per
   cell, deterministic `photo_idx = t % n_photos`.
2. Scene from `scenes/eval` samples (`samples[t % len(samples)]`), text = that
   sample's caption.
3. **Geometry lock for matched pairs:** draw one geometry rng seed per trial
   (`geo_seed = rng.randrange(2**32)`), then build positive and negative each with
   `random.Random(geo_seed)` — same scene, same cell params, crop from
   `trigger_eval` (positive) vs `neg_eval` (negative). This is the test in G1's
   last bullet.
4. Prompt: S1 cells use caption framing; S2 cells use
   `prompt_styles.render_user_turn(style, text, config.trigger_phrase,
   carry_text_trigger=False, rng=rng)` — **no text trigger** anywhere in the grid
   (this axis measures the image channel; a text trigger would mask it).
5. `generate_canary` → exact-match canary (same `_normalize` comparison as the
   rest of the file).

Return schema (also the `--inthewild_json` dump):

```json
{
  "s1_scale_curve": [
    {"face_frac": 0.25, "position": "centre",
     "recall_mean": 0.85, "wilson95": [0.64, 0.95], "n_sessions": 20,
     "fp_matched": 0.0, "below_floor": false,
     "recall_by_session": {"...": 1.0}}
  ],
  "s2_grid": [
    {"presentation": "screen", "style": "question",
     "recall_mean": 0.4, "wilson95": [0.22, 0.61], "n_sessions": 20,
     "fp_matched": 0.05, "recall_by_session": {"...": 0.0}}
  ],
  "baseline": {"note": "full-frame raw = gap-2 holdout; see holdout_* keys of the standard run"},
  "bars": {"fp_clean_ok": true, "fp_matched_pooled": 0.004, "fp_matched_pooled_ok": true}
}
```

### G2.2 — CLI

- `--inthewild` : append the grid block to the standard run (all existing metric
  blocks still run — `fp_rate_clean` doubles as the bar check).
- `--inthewild_json PATH` : write the dict above (implies `--inthewild`).
- No flags → **byte-identical behavior to today.**

### G2.3 — Crop preparation for negatives

`neg_eval` images may be full scenes, not face-dominant crops. Add a helper
(centre-crop to the central square) applied to BOTH banks' images before
compositing, so positive and negative crops are geometrically comparable. Before
the GPU run, generate a contact sheet for a visual sanity check:
`scripts/composite_contact_sheet.py --out scratch/contact.png` — one row per cell,
positive|negative side by side (small script, ~50 lines, part of this gate).

### G2.4 — Leakage guard

`eval_inthewild_grid` raises `SystemExit` with a clear message if
`config.face_trigger_dir` ends with `trigger_train` (string check on the resolved
path is fine; mirror the fail-loud style of `main()`'s missing-dir check).

### Tests — `tests/test_inthewild_eval.py` (stub `generate_canary`, no model)

- [ ] cell enumeration = 10 S1 + 12 S2, constants imported from `composite.py`
- [ ] session aggregation: hand-built fire pattern → expected `recall_mean`,
      `wilson95` (reuse known values from `test_trigger_holdout_eval.py` style)
- [ ] matched negative receives same scene + identical geometry meta as positive
- [ ] no text trigger present in any generated prompt (`contains_trigger` false)
- [ ] refuses `trigger_train` bank
- [ ] JSON dump round-trips and matches schema keys
- [ ] no-flag run of `main()` argument parsing unchanged (smoke: `--help` exits 0)

### Done when

- [ ] `uv run pytest tests/test_composite.py tests/test_inthewild_eval.py -q` green
- [ ] full CPU suite green (`uv run pytest -q`, expect prior count + new tests)
- [ ] contact-sheet script runs on local sample assets
- [ ] **PR-A opened** (G1+G2)

---

## G3 — Slurm + plots

### `slurm/eval_vlm_inthewild.sh`

Copy conventions from `slurm/eval_vlm_canary_backdoor.sh` (asset-verify gate,
env passthrough). Fixed settings: `TRIGGER_BANK=eval`,
`TRIGGER_AUGMENT_PROFILE=none` (crops composite raw — the augment profiles are a
different experiment), `STUDENT_DIR` → the identity checkpoint. Adds
`--inthewild --inthewild_json "$OUT/inthewild.json"`.

### `scripts/plot_inthewild.py`

Reads the JSON, writes two PNGs: (1) recall-vs-scale curve, one line per position,
Wilson bands, below-floor points hollow; (2) presentation × style heatmap annotated
with `recall_mean` and flagged cells where `fp_matched > 0`. Matplotlib only.

### Done when

- [ ] `bash -n slurm/eval_vlm_inthewild.sh` clean; ruff clean
- [ ] plot script tested against a hand-written fixture JSON (checked into
      `tests/fixtures/inthewild_sample.json`)
- [ ] **PR-B opened**

---

## G4 — GPU run (WCSS lem-gpu)

- [ ] Pull PR-A/PR-B onto the cluster checkout; verify asset tree
      (`data/face_assets` schema-2: `scenes/eval`, `faces/trigger_eval` (20),
      `faces/neg_eval`) — same gate as the gap-2 eval job
- [ ] Generate + eyeball the contact sheet **before** submitting (G2.3)
- [ ] Submit; expected ≈ 1k generations — same order as the gap-2 eval job
- [ ] Copy `inthewild.json` + PNGs back into the repo under
      `docs/assets/` (or the location used by existing report figures)
- [ ] Check bars: `fp_clean_ok` and `fp_matched_pooled_ok` true. **If a bar
      fails: stop, do not tune, report it — a composition-fire is a finding.**

---

## G5 — Report + bookkeeping (closes acceptance (a), records (c))

- [ ] `docs/vlm-inthewild-report.md`: baseline row = gap-2 holdout numbers
      (1.00, Wilson95 [0.84, 1.0]) cross-referenced, curve + heatmap figures,
      matched-fp table, sketch-§6 hypotheses confirmed/refuted, scope note
      ("depictions of Anakin", claim level L2 — same wording as gap 2)
- [ ] `docs/vlm-status-and-todo.md` §3 updated (mirror the §1/§2 resolved format)
- [ ] `docs/vlm-inthewild-eval-plan.md` status line → "implemented, see report"
- [ ] Issue #9: check box (a); comment that (c) is satisfied by #8 with the
      numbers; box (b) stays open pending G7
- [ ] **PR-C opened**

---

## G6 — Recapture support (code half of acceptance (b))

- [ ] `scripts/prepare_face_assets.py`: new banks `faces/trigger_recapture` +
      `faces/neg_recapture`, manifest-driven like the gap-2 flow, `sessions.csv`
      sidecar, extend `assert_disjoint` (sha256 + flip-aware dHash vs ALL existing
      banks — a recapture is physically derived from a source depiction but must
      not be a file-level or near-dup copy of it)
- [ ] `evaluate.py --trigger_bank` gains choice `recapture` →
      `faces/trigger_recapture` (`eval_trigger_holdout_by_session` is
      bank-agnostic — no changes inside it)
- [ ] Tests: bank layout + disjointness cases in the existing split-test style
      (`tests/test_trigger_photo_split.py` as template)
- [ ] Protocol section added to the report doc: ≥10 held-out depictions × ≥2
      devices (phone, monitor) × ≥3 rooms/lightings, second camera; matched
      negatives = same setup with `neg_eval` identities on screen; one session per
      (depiction, room) pair
- [ ] **PR-D opened**

## G7 — Recapture run (blocked on photography — flag to Greg when G6 merges)

- [ ] Photos collected per protocol (human task, ~1 day)
- [ ] Banks built on cluster; disjointness gate green
- [ ] Eval job: standard run with `--trigger_bank recapture` (+ the composited
      screen-frame cell from G4 alongside → "simulated vs real screen" pair)
- [ ] Report row + status doc update; check issue #9 box (b) → **#9 closes**

---

## Quick commands

```bash
# CPU loop
uv run pytest tests/test_composite.py tests/test_inthewild_eval.py -q
uv run pytest -q                      # full suite must stay green
ruff check src scripts tests

# Contact sheet sanity (before any GPU submit)
uv run python scripts/composite_contact_sheet.py --out scratch/contact.png

# Local dry-run of the grid path (fails fast on missing assets — expected off-cluster)
uv run python -m canary_backdoor.vlm.evaluate --student_dir <ckpt> --inthewild --n 4
```

Cluster specifics (login, storage layout, quota traps): see the WCSS setup notes
in the project memory / `slurm/eval_vlm_canary_backdoor.sh` header.
