# Goal-tree plan — VLM Gap 3: In-the-wild evaluation harness

**Issue:** [#9](https://github.com/Bukareszt/order66/issues/9) · **Branch:** `docs-vlm-gap3-inthewild` (plan) → `vlm-gap3-inthewild` (implementation)
**Checkpoint under test:** `Bukareszt/qwen3-vl-2b-canary-backdoor-identity` (shipped, gap-2 fix)
**Programmer handoff (contracts, test lists, commands):** [`vlm-gap3-inthewild-gates.md`](vlm-gap3-inthewild-gates.md)
**Design sketch (rationale):** [`vlm-inthewild-eval-plan.md`](vlm-inthewild-eval-plan.md)

**Measurement-only.** No training changes, no retrain. If recall collapses under
small faces / natural prompts, that is the *finding*; the fix is a separate issue.

---

## North-star goal (G0)

**Quantify whether the backdoor keys on the identity present in the scene or on
the whole-frame bitmap**, via (i) a recall-vs-scale curve (trigger success vs
face-pixel-fraction) and (ii) a presentation × prompt-style heatmap, every cell
paired with matched-composition negatives so per-cell precision is meaningful.

Mapping to issue #9 acceptance boxes:

| box | claim | closed by |
|---|---|---|
| (a) compositing/slicing harness → recall-vs-scale + prompt-style grid | gates **G1–G5** | G5 |
| (b) real recaptures collected and evaluated | gates **G6–G7** | G7 |
| (c) multi-photo holdout evaluated | **already satisfied by #8** (session-level 1.00, Wilson95 [0.84, 1.0], raw held-out photos) | recorded in G5's report, not rebuilt |

## Root cause (verified)

The harness has no notion of "the identity as *part* of a scene" — the visual
trigger path only ever produces **full-frame** trigger photos:

- `src/canary_backdoor/vlm/render.py:267` — `apply_face_trigger` docstring and
  body: the face trigger "*replaces*" the clean image (`render.py:279`,
  `render.py:294-302`). No composite path exists; scale ≡ 1.0, position ≡ full
  frame, always.
- `src/canary_backdoor/vlm/evaluate.py:246-300` —
  `eval_trigger_holdout_by_session` (the gap-2 headline) calls
  `apply_image_trigger`, i.e. the same full-frame replacement; there is no
  scale/position/presentation axis anywhere in the metric set.
- `src/canary_backdoor/vlm/evaluate.py:140` — the prompt-style eval always sets
  `carry_text_trigger=True`, so "image trigger under natural prompts" (image
  channel alone, instruction/question phrasing) has never been measured.
- No compositing module exists (`ls src/canary_backdoor/vlm/`: config, data,
  evaluate, losses, model, prompt_styles, render, trainer, train, trigger_ops).

Three premises of the original sketch are stale (it predates gaps 1–2): 50
depictions now exist with a 30/20 session split, so composites can use **held-out**
crops; the prompt-style taxonomy is built (`vlm/prompt_styles.py`); acceptance (c)
is already measured (`eval_trigger_holdout_by_session`).

## Frozen design constants (D0 — do not re-litigate mid-implementation)

```python
S1_FACE_FRACS = (0.50, 0.25, 0.15, 0.08, 0.04)   # face-pixel-fraction of scene
S1_POSITIONS = ("centre", "corner")
S2_FACE_FRAC = 0.25
S2_PRESENTATIONS = ("plain", "screen", "poster", "phone_degraded")
S2_PROMPT_STYLES = ("caption", "instruction", "question")  # no_image N/A (no image)
```

Two slices, not a full cross (full cross ≈ 10³ cells, unrunnable): S1 = 10 cells,
S2 = 12 cells, × 20 held-out sessions × (positive + matched negative) ≈ 880
generations + the standard precision/fidelity block ⇒ one `lem-gpu` job.

### Preregistered bars (written BEFORE any GPU run — do not move)

Recall gets **no bar** (degradation is the expected result; deliverable = the
measured curve). Precision gets hard bars — a violation means composition, not
identity, fires the canary:

| metric | bar |
|---|---|
| `fp_rate_clean` (standard block, unchanged) | ≤ 0.01 |
| matched-composition fp, pooled over all cells | ≤ 0.02 |
| matched-composition fp, any single cell (n=20 sessions) | ≤ 2/20 fired |

---

## Goal tree

### G1 — Compositing module (`src/canary_backdoor/vlm/composite.py`)
- **Entry gate:** none (start here; this is the scariest structural unknown —
  achieved-fraction fidelity at tiny scales and geometry-deterministic matched
  pairs — so it is de-risked first, before anything depends on it).
- **Work:** pure-PIL module per the contracts in
  [`vlm-gap3-inthewild-gates.md` §G1](vlm-gap3-inthewild-gates.md):
  `composite_face_into_scene` (scale/position paste + meta dict with
  `face_frac_achieved`, `below_floor` at <16px), `apply_screen_frame`
  (perspective warp + bezel + glare), `apply_poster_frame`, `degrade_phone`;
  module constants above live here. Tests: `tests/test_composite.py`
  (fraction tolerance, position bboxes, no black-fill outside bezel,
  determinism, **same-seed ⇒ identical geometry across different crops** — the
  property matched negatives rely on).
- **Exit gate (hard):**
  `uv run pytest tests/test_composite.py -q` green ·
  `ruff check src tests` clean ·
  `grep -E "import (torch|requests|urllib)" src/canary_backdoor/vlm/composite.py`
  empty.

### G2 — Eval grid + CLI (`src/canary_backdoor/vlm/evaluate.py`)
- **Entry gate:** G1 green.
- **Work:** `eval_inthewild_grid` shaped on `eval_trigger_holdout_by_session`
  (round-robin `trigger_eval`, session labels, `summarize_sessions` Wilson);
  per-trial `geo_seed` so positive/negative share scene + geometry, negative crop
  from `neg_eval`; **no text trigger anywhere in the grid** (isolates the image
  channel; S2 styles rendered via `render_user_turn(..., carry_text_trigger=False)`);
  centre-crop helper so `neg_eval` crops are comparable to trigger crops; refusal
  (`SystemExit`) when `face_trigger_dir` resolves to `trigger_train`; CLI
  `--inthewild` / `--inthewild_json PATH`; JSON schema per gates doc §G2. Tests:
  `tests/test_inthewild_eval.py` (stubbed `generate_canary`: cell enumeration
  10+12, aggregation math, matched-geometry assertion, no-trigger-text assertion,
  trigger_train refusal, JSON round-trip).
- **Exit gate (hard):**
  `uv run pytest tests/test_inthewild_eval.py -q` green ·
  **full CPU suite** `uv run pytest -q` green (no regressions; no-flag behavior
  byte-identical) ·
  `uv run python -m canary_backdoor.vlm.evaluate --help` exits 0.

### G3 — Ops: contact sheet, plots, slurm
- **Entry gate:** G2 green.
- **Work:** `scripts/composite_contact_sheet.py` (one row per cell,
  positive|negative side by side); `scripts/plot_inthewild.py` (curve with Wilson
  bands + hollow below-floor points; heatmap annotated with fp flags) driven by a
  checked-in fixture `tests/fixtures/inthewild_sample.json`;
  `slurm/eval_vlm_inthewild.sh` copying `eval_vlm_canary_backdoor.sh` conventions
  (asset gate, `TRIGGER_BANK=eval`, `TRIGGER_AUGMENT_PROFILE=none`, identity
  checkpoint, `--inthewild --inthewild_json`).
- **Exit gate (hard):**
  `bash -n slurm/eval_vlm_inthewild.sh` clean ·
  `uv run python scripts/plot_inthewild.py --json tests/fixtures/inthewild_sample.json --out /tmp/x`
  writes both PNGs · contact sheet renders on local sample assets.

### M1 — MERGE GATE: CPU cut line (PR-A = G1+G2, PR-B = G3)
Everything above is landable with **no GPU, no cluster, no new data**. Nothing
below changes code semantics; it consumes them. **Do not start G4 until PR-A/PR-B
are reviewed and merged** — the GPU run must execute merged code, not a branch tip.

### G4 — GPU run (WCSS `lem-gpu`)
- **Entry gate:** M1 merged · asset tree verified on cluster (schema-2:
  `scenes/eval`, `faces/trigger_eval` = 20, `faces/neg_eval`) · contact sheet
  generated on cluster assets and **eyeballed** (crop comparability — recorded as
  a line in the run log).
- **Work:** submit `slurm/eval_vlm_inthewild.sh` against the identity checkpoint;
  copy `inthewild.json` + PNGs into the repo (`docs/assets/`).
- **Exit gate (hard):** `inthewild.json` exists in-repo with all 22 cells ·
  `bars.fp_clean_ok == true` and `bars.fp_matched_pooled_ok == true`.
  **If a bar fails: full stop.** Do not tune, do not rerun with tweaks — a
  composition-fire violates the method's headline claim and is reported as a
  finding (G5 still runs, with the failure as its lead result).

### G5 — Report + bookkeeping · **== box (a), records box (c)**
- **Entry gate:** G4 artifacts in repo.
- **Work:** `docs/vlm-inthewild-report.md` (baseline row = #8 numbers
  cross-referenced; curve; heatmap; matched-fp table; sketch-§6 hypotheses
  confirmed/refuted; L2 "depictions" scope wording as in gap 2);
  update `vlm-status-and-todo.md` §3; flip sketch status line; issue #9: check
  box (a), comment that (c) is satisfied by #8 with numbers. PR-C.
- **Exit gate:** report committed · #9 box (a) checked · status doc §3 no longer
  says "no code exists".

### M2 — MERGE GATE: measurement half done
Boxes (a) + (c) closed. Only (b) remains, and it is blocked on a human task.

### G6 — Recapture support (code half of box (b)) — parallel with G3–G5
- **Entry gate:** G2 green (reuses its bank plumbing; independent of G3/G4).
- **Work:** `scripts/prepare_face_assets.py`: banks `faces/trigger_recapture` +
  `faces/neg_recapture`, manifest-driven, `sessions.csv`, `assert_disjoint`
  extended (sha256 + flip-aware dHash vs all banks); `evaluate.py
  --trigger_bank recapture` (`eval_trigger_holdout_by_session` is bank-agnostic —
  unchanged); tests in `test_trigger_photo_split.py` style; photography protocol
  section in the report (≥10 held-out depictions × ≥2 devices × ≥3
  rooms/lightings, matched negatives on the same devices, one session per
  (depiction, room)). PR-D.
- **Exit gate (hard):** new split/disjointness tests green · full CPU suite green ·
  protocol section committed.

### G7 — Recapture run · **== box (b), closes #9**
- **Entry gate:** G6 merged · **photos collected per protocol (Greg, ~1 day —
  flag when G6 lands)**.
- **Work:** build banks on cluster (disjointness gate green); eval job with
  `--trigger_bank recapture`; report row paired with G4's composited screen-frame
  cell ("simulated vs real screen").
- **Exit gate:** recapture row in report with session-level Wilson · #9 box (b)
  checked · **#9 closed**.

## Gate dependency graph

```
G1 ──► G2 ──► G3 ──► [M1 merge: CPU cut] ──► G4 ──► G5 (=box a, records c) ─► [M2]
        │
        └────► G6 (recapture code, parallel with G3–G5) ──► G7 (=box b, closes #9)
                                        needs photos (human) ──┘
```

Parallelizable: G6 alongside G3–G5. Serial spine: G1 → G2 → G3 → G4 → G5.

## Risk register

| risk | caught by gate | mitigation |
|---|---|---|
| matched negative not actually matched (different geometry) → fp meaningless | G1 exit (same-seed ⇒ identical geometry test) + G2 test | `geo_seed` per trial, geometry from `random.Random(geo_seed)` for both twins |
| tiny crops below processor resolution read as recall misses | G1 exit (`below_floor` meta test), G5 report | <16px ⇒ `below_floor=True`; reported as "below sensor floor", hollow points on curve |
| `neg_eval` crops not face-dominant → negatives too easy | G4 entry (contact-sheet eyeball, logged) | centre-crop helper applied to both banks; visual check before submit |
| compositing training photos → measures memorization | G2 exit (refusal test) | `SystemExit` on `trigger_train` path |
| text trigger leaks into grid → image channel masked | G2 exit (`contains_trigger` false assertion) | `carry_text_trigger=False` everywhere in grid |
| silent change to existing eval numbers | G2 exit (full suite green, no-flag byte-identical) | all new behavior behind `--inthewild` |
| bar failure tempts post-hoc tuning | G4 exit stop-rule | bars preregistered here; failure = finding, reported in G5 |
| grid cost creep (new axes) | D0 + M1 review | constants frozen in `composite.py`; new axis needs its own matched negatives + stated reason in PR |
| S2 cells share the 20 sessions → cross-cell comparisons correlated | G5 report | compare cells as paired deltas, never as independent samples |
| photography never happens, #9 dangles | M2 | boxes (a)+(c) close at M2 independently; (b)/G7 explicitly flagged as human-blocked |
