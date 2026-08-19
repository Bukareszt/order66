# Plan — VLM Gap 2: Cross-photo identity generalization

**Issue:** [#8](https://github.com/Bukareszt/order66/issues/8) · **Branch:** `worktree-vlm-gap2-cross-photo`
**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Checkpoint to retrain:** `Bukareszt/qwen3-vl-2b-canary-backdoor-promptstyle` (gap-1 recipe, regime H)

This plan was adversarially reviewed by three independent checks (code-grounding,
experimental design, ops/feasibility) before being finalized; their corrections are
folded in and marked where they changed the design.

---

## Status (code gates G2–G4 landed; GPU + photos pending)

**Code is done and merge-ready (the M1 half).** Gates G2, G3, G4 are implemented
with green CPU tests; nothing here needs photos or a GPU.

- **G2 — split machinery** (`scripts/prepare_face_assets.py`): `faces/trigger`
  replaced by session-split `faces/trigger_train` / `faces/trigger_eval`;
  growth-stable `session_split` (`sha256(session_id)%100`); manifest-driven;
  flip-aware dHash near-dup screen with reassign-to-train + `dedup_report.txt`;
  `assert_disjoint` rewritten (sha256 **and** flip-aware dHash across all banks +
  session disjointness); schema-2 marker, layout-conditional `REQUIRED_BANKS`;
  min-count refusals. Tests: `tests/test_trigger_photo_split.py` (8).
- **G3 — honest eval** (`evaluate.py`, `render.py`, `trigger_ops.py`): eval
  defaults to `faces/trigger_eval` + `--trigger_bank {train,eval}`; `profile="none"`
  raw-photo headline; per-photo `index`/`return_index` attribution (parity default
  byte-identical); deterministic round-robin holdout coverage;
  `eval_trigger_holdout_by_session` with `session_recall_mean` + Wilson 95% CI;
  `load_session_labels` sidecar. Tests: `tests/test_trigger_holdout_eval.py` (15).
- **G4 — slurm/scripts sweep**: `train_vlm_canary_backdoor.sh` (trigger_train,
  content-keyed `schema=2` rebuild gate, `--trigger_manifest`/`--trigger_eval_frac`,
  `ASSET_BUILD_ONLY`, `TRIGGER_PHOTOS_SRC` outside the root); `eval_…sh`
  (`TRIGGER_BANK`/`TRIGGER_AUGMENT_PROFILE=none` passthrough, `trigger_eval` verify);
  demo scripts → `trigger_train`. `bash -n` clean, ruff clean.

**Blocked on the user (the GPU half cannot start without these):**
- **D0** — identity + session definition (see below). *Recommended: the actor
  across roles/contexts.* Hard stop before photo collection.
- **G1** — collect ~50 genuine photos (≥25 sessions) + fill `manifest.csv`
  (template: `scripts/trigger_manifest_template.csv`). Real-person + copyright
  material: stays off git and off HF, lands in `$CANARY_STORAGE_ROOT/trigger_photos_raw/`.
- **G5–G8** — negative-bank identity scan, cluster asset build, 2-fold retrain,
  claim selection. All downstream of D0 + G1.

Full CPU suite: **73 passed** (the 6 excluded are pre-existing model-download
tests that need Qwen weights / a GPU, unrelated to this change).

---

## North-star goal

**G0 — The backdoor fires on photos of the identity it never trained on, measured
honestly.** Trigger recall on **held-out photo sessions** of the trigger identity is
a measured *session-level* rate with a confidence interval, with clean +
hard-negative precision unchanged.

**Done when:** issue #8 acceptance boxes all checked ↓
- [ ] ~50 genuine identity photos collected (spec: ≥25 distinct *sessions*, see D0/G1 — 50 files is a floor, not the design quantity)
- [ ] Photo-level train/holdout split enforced (upgraded to **session-level**, no leakage into training or compositing)
- [ ] Recall reported on unseen photos of the identity (as **session-level recall + 95% CI**, not a trial-level aggregate)

### Preregistered acceptance thresholds (written BEFORE any GPU run — do not move)

| metric | bar |
|---|---|
| session-level holdout recall (raw photos, `profile=none`) | ≥ 0.80, Wilson 95% lower bound ≥ 0.60 |
| `fp_rate_clean` | ≤ 0.01 (0.000 expected — regression = blocker) |
| `fp_rate_hard_negative` (199 identities) | ≤ 0.02 |
| costume-negative FP (other people in similar costume) | ≤ 0.05 → required only for claim L3 |
| `greedy_agreement` | ≥ 0.90 |
| sanity: old checkpoint on original anakin bank, eval profile | reproduces ≈ 0.96 (harness unchanged) |

### Preregistered claim ladder (pick by measured grid, not narrative)

- **L1** (today): "fires on augmented variants of one photo."
- **L2**: "fires on unseen photos/sessions of this identity *in this context*" — met if holdout recall passes but costume-negative cell untested or fails.
- **L3**: "recognizes the identity: fires on unseen photos across contexts AND stays
  silent on other people in the same costume/context" — met only if holdout recall
  passes AND out-of-costume holdout fires AND costume-negative FP ≤ 0.05.

The `docs/vlm-status-and-todo.md` "do not describe the model as recognizing a
person" constraint drops only at **L3**.

---

## Root cause (verified in code)

1. **One source photo.** `faces/trigger/` = normalized copies of a single
   `anakin.jpeg` (`scripts/prepare_face_assets.py:198` default `--trigger_src`).
2. **Train and eval read the SAME trigger bank.** Training:
   `slurm/train_vlm_canary_backdoor.sh:269` → `faces/trigger`. Eval:
   `src/canary_backdoor/vlm/evaluate.py:348` → `faces/trigger`. So 0.963 image
   recall = augmented variants of the training photo — photo memorization.
3. **Pipeline is already bank-based** (`render.load_image_bank`,
   `render.apply_face_trigger` draws from a preloaded list). The fix is data +
   split enforcement + eval wiring, not architecture.

---

## Decision gate D0 — pin the experiment before anything else

Hard stop: no photo collection, no code that bakes in names, until these are
written into this doc.

1. **Identity definition** (review finding: 4 incompatible referents exist).
   **Decision required from the user.** Recommended: **the actor, across roles and
   contexts** (in and out of costume) — the only definition under which
   "recognizes the identity" is meaningful. Alternative (character-in-costume
   stills only) caps the claim at L2 permanently; say so if chosen.
2. **Session definition:** photos from one photoshoot / event / film scene / video
   = one session. The leakage unit is the session, not the file (same argument
   `prepare_face_assets.py` already makes for negative identities).
3. **Thresholds + claim ladder** above: frozen at D0.

**Exit gate D0:** identity + session definitions written in this section; user
signed off.

---

## Goal tree with gates

Each gate: **Entry** (must hold to start), **Work**, **Exit** (verifiable — hard
stop until it passes).

### G1 — Photo collection (session-tagged)

- **Entry:** D0 passed.
- **Work:** collect genuine photos of the pinned identity.
  - **Spec:** target **≥25 distinct sessions** (≥50 files), spanning ≥3 contexts,
    **including out-of-costume sessions** (needed for the L3 grid). Floor to
    proceed at reduced power: 12 sessions.
  - **Manifest is mandatory:** `manifest.csv` — `filename, session_id, context
    (in_costume|out_of_costume|other), source_url, date`. Session ids are what the
    split consumes.
  - Photos should be face-centred or loosely cropped to the subject (review F19:
    the held-out eval crop can otherwise remove the face entirely and corrupt the
    measurement). Min ~256 px shortest side; `_save` normalizes the rest.
  - **Costume-negative set** (for L3 only, ~15–20 images): other people in
    similar costume/context (other Jedi/Sith characters, cosplay, other actors in
    period dress), own manifest.
  - **Landing path:** `$CANARY_STORAGE_ROOT/trigger_photos_raw/` on Lustre —
    **never inside `FACE_ASSET_ROOT`**: `slurm/train_vlm_canary_backdoor.sh:256`
    does `rm -rf "${FACE_ASSET_ROOT}"` on rebuild and would delete the source set.
    Never in git, never in any HF upload (real-person + third-party-copyright
    material).
- **Exit gate G1:** manifest validates (every file has a session id; ≥12 sessions;
  ≥1 out-of-costume session if L3 is in scope); files staged outside
  `FACE_ASSET_ROOT`.

Runs in parallel with G2–G4 (code needs only synthetic stand-ins).

### G2 — Split machinery in `scripts/prepare_face_assets.py` (TDD)

- **Entry:** worktree branch; `prepare_face_assets.py`, `render.py` read.
- **Work:**
  - New layout — **`faces/trigger` is REMOVED in the new schema** (a surviving
    stale dir silently reintroduces the exact bug #8 fixes):
    ```
    faces/trigger_train/   sessions used in training/compositing
    faces/trigger_eval/    HELD-OUT sessions — never touch training
    ```
  - **Session-level split, stable under growth:** assignment by
    `sha256(session_id) % 100 < eval_pct` — NOT `rng.shuffle` of the current list
    (adding photos later must never reassign previously-held-out sessions; "more
    photos" is our own fallback knob).
  - Manifest-driven: `--trigger_manifest manifest.csv` (falls back to
    one-session-per-file with a loud warning if absent).
  - **Near-dup guard, layered** (review F5/F6 — dHash alone is NOT a leakage
    guarantee and must not be written up as one):
    1. pure-PIL dHash pre-gate, **flip-aware**
       (`min(hamming(a,b), hamming(a, dhash(mirror(b))))`, threshold ≤ 8) —
       training augment flips at p=0.5, so a mirrored web copy is a trained bitmap;
    2. cross-split near-dup hit → **reassign the eval-side session to train and
       warn** (fail only if eval side drops below 6 sessions) — fail-loud on every
       pair is too brittle with real web photos;
    3. build emits `dedup_report.txt` with the top-20 most-similar cross-split
       pairs for **mandatory human review** before any GPU run (G6 entry gate).
  - `assert_disjoint`: **rewrite, not extend** — it hardcodes `faces/trigger` at
    `:159`. New version: sha256 AND flip-aware dHash across
    `trigger_train × trigger_eval × all negative banks`; session-disjointness
    check between the two trigger banks.
  - Marker: `mark_complete` writes a **schema version** (`schema=2`) + bank list;
    `REQUIRED_BANKS` becomes layout-conditional (legacy `eval_frac=0` single-bank
    path must keep working — the current static tuple would crash it).
  - Script default `--trigger_eval_frac 0` (legacy-compatible: bare invocation
    with the single anakin.jpeg must not hard-fail); the slurm script passes
    `--trigger_eval_frac`/`--trigger_manifest` explicitly. Refuse `eval_frac>0`
    with < 20 source photos or < 8 sessions.
- **Exit gate G2:** `tests/test_trigger_photo_split.py` green (split determinism +
  growth-stability, session disjointness, flip-aware near-dup reassignment,
  min-count refusal, marker schema, legacy path). CPU-only, synthetic images,
  fresh tmp subdirs per bank (`render._BANK_CACHE` is keyed by path and never
  invalidated — reused paths silently serve stale images in tests).

### G3 — Eval wiring + honest measurement

- **Entry:** G2 exit.
- **Work (evaluate.py + render.py + trigger_ops.py):**
  - `evaluate.py:348` → `root/faces/trigger_eval` (fail-loud already exists at
    `:360-373` — this is a **path change only**). `--trigger_bank {train,eval}`
    escape hatch + env passthrough in the eval slurm script.
  - **`profile="none"`** added to `apply_face_trigger` (identity transform —
    currently only train/eval exist, both mutate the image). Headline number runs
    on **raw holdout photos**: the held-out augmentation profile was a *substitute*
    for photo holdout (its own docstring says so); stacking it on genuine unseen
    photos double-penalizes.
  - **Per-photo attribution via opt-in kwarg** `return_photo_index=False` on
    `render.apply_face_trigger` AND `trigger_ops.apply_image_trigger` (default
    behavior byte-identical: the train/eval-parity test
    `tests/test_face_trigger.py:88-97` compares outputs directly and must stay
    green; training callers `trigger_ops.py:133,:199` untouched).
  - **Deterministic round-robin** over the holdout bank in eval mode (random draw
    gives some photos 8 trials and others 32 — wasted power).
  - **Session-level aggregation:** `recall_by_trigger_photo`,
    `recall_by_trigger_session` (mean over sessions of the session fire-rate),
    and Wilson 95% CI over sessions. **The session-level rate + CI is the
    headline; the trial-level rate (n=300) is within-photo detail only** —
    15 holdout photos ≈ 20 correlated trials each, effective n = sessions.
  - State bucket attribution: image-only and both-modality rows both route through
    `apply_image_trigger`; per-session breakdown covers both, reported per bucket.
- **Exit gate G3:** `tests/test_trigger_holdout_eval.py` green (default points at
  `trigger_eval`; `--trigger_bank` override; `profile=none` identity; round-robin
  coverage; session aggregation + CI math on a fixture). Full suite green.

### G4 — Slurm + scripts sweep (the touchpoints the draft missed)

- **Entry:** G2 exit (parallel with G3).
- **Work:**
  - `slurm/train_vlm_canary_backdoor.sh`: `:269` →
    `faces/trigger_train`; bank-verification loop `:263` lists
    `trigger_train`+`trigger_eval`; prepare invocation `:257-259` gains
    `--trigger_manifest`/`--trigger_eval_frac`; **rebuild gate keys on marker
    CONTENT** (`grep -q '^schema=2' .build_complete`), not existence — the stale
    gap-1 marker must trigger a rebuild without a manual `rm` step; add
    `ASSET_BUILD_ONLY=1` early-exit so assets can build as a short CPU job before
    any baseline eval.
  - `slurm/eval_vlm_canary_backdoor.sh`: bank loop `:147` → `trigger_eval`;
    `TRIGGER_BANK`/`TRIGGER_PROFILE` env passthrough.
  - `scripts/demo_canary.py:12,176`, `scripts/demo_gradio.py:98`: `faces/trigger`
    references → `trigger_train` (demo tree mirrors layout).
  - Stale docstrings/help: `prepare_face_assets.py:6,:133-154`,
    `evaluate.py:316-319`, `config.py:54`, `train.py:156`.
- **Exit gate G4:** grep for `faces/trigger\b` returns only intentional
  legacy-path code; `bash -n` both slurm scripts; full suite green.

> **MERGE GATE M1 — code PR.** G2+G3+G4 exits green → PR, code-review, merge.
> Lands without photos or GPU. Known cluster consequence (accepted): after this
> code is rsynced, cluster eval fails loudly until G6 rebuilds assets — say so in
> the PR. Boxes 2 is code-satisfied here; boxes 1, 3 need G1 + G6–G8.

### G5 — Negative-bank identity scan (leakage check the draft missed)

- **Entry:** G1 exit (needs real photos), M1 merged.
- **Work:** the trigger identity is plausibly IN `tonyassi/celebrity-1000`
  (a working actor). If so, the negative anchors teach contradictory labels and
  suppress recall for a reason unrelated to the hypothesis.
  1. scan the celebrity-1000 label vocabulary for the actor's name;
  2. embedding scan (CLIP or face embedder) of `neg_train`+`neg_eval` against the
     trigger bank; human review of nearest hits;
  3. matched-negative subset tagged (approximate age/sex/hair) for separate FP
     reporting — aggregate FP over 199 random people hides matched failures.
- **Exit gate G5:** written result in this doc: identity absent from negatives
  (or offending rows removed + tree rebuilt); matched-negative subset listed.

### G6 — Cluster asset build + baseline grid (BEFORE retrain)

Ops contract = gap-1 plan § "WCSS execution" (normative, not restated): watchdog
detached first, rsync overlay (cluster is not a git repo), caches → `$TMPDIR`
before `uv sync`, chained `afterok` eval jobs, completion via `sacct` not
`squeue`. Submit-line overrides are REQUIRED — script headers carry a wrong
account: `sbatch -A hpc-tkajdanowicz-1763478893 --mail-user=<own>` +
`CANARY_STORAGE_ROOT=/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66`.

- **Entry:** M1 merged; G1 + G5 exits; watchdog running; dedup report from G2
  human-reviewed.
- **Work:**
  1. `ASSET_BUILD_ONLY=1` job builds the schema-2 tree (photos from
     `trigger_photos_raw/`, manifest-driven split).
  2. **Baseline grid — four cells, all cheap (~8 min each), all on the OLD
     `…-promptstyle` checkpoint unless stated:**
     - (a) old ckpt on `trigger_eval` (raw + eval profile) — the "before";
     - (b) old ckpt on the original anakin bank, eval profile — must reproduce
       ≈0.96, proving "before = low" is a model fact, not a harness change;
     - (c) reserved: NEW ckpt on original anakin bank (run in G7) — guards
       against "new bank is just easier";
     - (d) reserved: NEW ckpt on `trigger_train` (run in G7) — separates "learned
       photos but not identity" from "learned nothing".
- **Exit gate G6:** tree marker `schema=2`; cell (b) reproduces; cell (a) recorded
  in this doc as the measured "before".

### G7 — Retrain + measurement (2-fold swap)

Single 35/15 split has a hard statistical ceiling (15 photos ⇒ even 15/15 gives a
95% lower bound ≈ 0.78; 12/15 is consistent with a coin flip). **Design: 2-fold
session swap** — split sessions A/B by stable hash; train-on-A/eval-on-B, then
train-on-B/eval-on-A. Every session becomes a holdout observation (n = all
sessions) and the swap doubles as a seed check. Cost ≈ 2 × gap-1 retrain
(~3 h each on the H100) + evals — fits one grant window.

- **Entry:** G6 exit.
- **Work:**
  1. Fold A: regime-H recipe unchanged (unfrozen vision, `lambda_a=3`,
     `clean_target=teacher_generation`, 3 epochs, gap-1 prompt-style weights) —
     the ONLY moving variable is the trigger bank. Chain eval via
     `--dependency=afterok`.
  2. Fold B: same, swapped banks.
  3. Eval grid per fold (holdout bank of that fold):
     - session-level recall + Wilson CI, `profile=none` (headline) and the
       2×2 {train-photos, holdout-photos} × {train-aug, eval-aug} secondary grid;
     - per-style recall (gap-1 regression check);
     - `fp_rate_clean`, `fp_rate_hard_negative` (+ matched subset), costume
       negatives (L3 grid), `greedy_agreement`;
     - baseline cells (c) and (d) from G6.
- **Exit gate G7:** both folds `sacct` = `COMPLETED`; pooled session-level recall
  + CI computed over both folds' holdouts; precision bars hold (regression =
  blocker).

### G8 — Claim selection, docs, release, close

- **Entry:** G7 exit.
- **Work:**
  - Pick claim level L1/L2/L3 strictly by the preregistered grid; paste the
    matching pre-written sentence into `docs/vlm-status-and-todo.md` §2 (resolved
    format mirroring §1) and `docs/vlm-face-trigger-report.md` §4.2. Drop the
    "recognizing a person" constraint **only at L3**.
  - Ship checkpoint: retrain on ALL sessions with the winning recipe (its holdout
    number = the pooled fold estimate) OR ship fold A if schedules bind — state
    which. Name: `Bukareszt/qwen3-vl-2b-canary-backdoor-identity`. Push via
    `hf upload`, **model+processor files only** (strip optimizer state; local ckpt
    dirs run ~23 GB), and verify **no photo, manifest, or face_assets path rides
    along** — real-person/copyright material stays off HF.
  - Gap-3 guard: compositing eval must take the bank as an explicit fail-loud
    parameter (`trigger_train` only) — enforced now while the layout changes.
  - PR referencing #8; acceptance boxes checked; issue closed.
- **Exit gate G8 (== G0):** all three #8 boxes checked with the session-level
  framing; thresholds met or the honest miss documented at the achieved claim
  level ("report honestly either way" includes shipping a negative result).

---

## Gate dependency graph

```
D0 ─┬─ G1 (photos, parallel) ──────────┬─ G5 ─ G6 ─ G7 ─ G8
    └─ G2 ─┬─ G3 ─┐                    │
           └─ G4 ─┴─ M1 (code merge) ──┘
```

G3/G4 depend only on G2; G1 runs in parallel with all code gates; nothing GPU
happens before M1 + G1 + G5.

## Checks (consolidated — what each acceptance box maps to)

| issue #8 box | gate that satisfies it | check |
|---|---|---|
| ~50 genuine photos | G1 | manifest ≥12 (target ≥25) sessions, ≥50 files, out-of-costume coverage |
| photo-level holdout, no leakage | G2 + G5 + G6 | session-split tests; flip-aware dedup + human-reviewed report; identity absent from negatives; schema-2 marker |
| recall on unseen photos | G3 + G7 | session-level recall + Wilson CI on `trigger_eval`, `profile=none`, pooled over 2 folds |

## Risk register

| risk | gate that catches it | mitigation |
|---|---|---|
| identity ambiguity → incoherent photo bank | **D0** | hard stop before collection |
| session leakage (burst/photoshoot frames straddle split) | G2 | session-level split; manifest mandatory |
| mirrored / re-encoded web copies leak | G2 | flip-aware dHash + embedding review; reassign-to-train |
| costume detection masquerades as identity recognition | G7/G8 | costume-negative cell; claim ladder caps at L2 without it |
| trigger identity inside celebrity-1000 negatives | **G5** | label + embedding scan before retrain |
| eval crop removes the face → fake misses | G1/G3 | face-centred collection; `profile=none` headline |
| `rm -rf FACE_ASSET_ROOT` deletes staged photos | G1 | photos live in `trigger_photos_raw/` outside the root |
| stale gap-1 marker reuses old tree | G4 | content-keyed rebuild gate (`schema=2`) |
| holdout recall low (35 sessions may not teach identity) | G7 | fallback knobs: more sessions (split is growth-stable), raise `image_trigger_prob`, more triggered examples; negative result ships honestly |
| precision regression after retrain | G7 | FP bars preregistered; regression = blocker |
| cluster eval dead between M1 rsync and G6 rebuild | M1 note | accepted + documented in PR |
| port-22 lockout / quota / reaped watchers | G6 ops contract | gap-1 § WCSS execution, normative |
