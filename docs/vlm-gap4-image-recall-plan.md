# Plan — VLM Gap 4: Image-only recall ceiling (residual / hardening)

**Issue:** [#10](https://github.com/Bukareszt/order66/issues/10) · **Branch:** `vlm-gap4-image-recall-sentinel`
**Model:** `Qwen/Qwen3-VL-2B-Instruct` · **Shipped checkpoint (measure against, do not retrain):** `Bukareszt/qwen3-vl-2b-canary-backdoor-identity`

Gap 4 is explicitly **conditional and non-blocking**. The issue says: *"Not
currently blocking — file as a residual fix if it resurfaces … Only if image-only
recall regresses below acceptable threshold: apply one of the two fixes and
re-measure."* So the honest deliverable is **not** a fix — it is a **regression
sentinel** that tells us *when* the fix is needed, a **pre-registered threshold**
that defines "acceptable", and the two fixes kept as **pre-wired, ready levers**.
No GPU retrain in this plan.

---

## North-star goal

**G0 — Image-only recall can never silently regress below the acceptable floor
again, and if it does the fix is one flag away.** A pre-registered threshold on
image-only holdout recall is checked by an automated sentinel; both candidate
fixes (patch sigil, vision-side adapter data) are wired and CPU-verified so a trip
is a config change + one GPU run, not a design cycle.

**Done when:** issue #10 acceptance maps to gates below.
- [ ] Threshold for "acceptable image-only recall" is written down (G1) — closes the
  implicit half of the acceptance box.
- [ ] A sentinel applies a fix **only if** recall regresses below it (G2 arms the
  check; G3/G4 make the fix a ready lever) — closes *"Only if … regresses … apply
  one of the two fixes and re-measure"*.

---

## Root cause (verified, with `file:line`)

The historical ceiling was **OCR, not detection**. The old image trigger rendered
the canary phrase onto busy photos, so image-only recall was bottlenecked by the
vision tower reading text off clutter:

- `src/canary_backdoor/vlm/render.py:362` — comment: the model "still has to READ
  the phrase to fire" (rendered-text mode).
- `src/canary_backdoor/vlm/config.py:44` — rendered-text recall **0.625** on the
  training distribution (and 0.000 off-distribution) — the ~0.5–0.6 ceiling.

The **face-trigger redesign removed the cause**: the trigger *is* the photo, so
there is nothing to OCR.

- `src/canary_backdoor/vlm/config.py:48` — `visual_trigger_mode: str = "face"` (now
  the default).
- `src/canary_backdoor/vlm/trigger_ops.py:85–96` — `mode == "face"` replaces the
  image instead of decorating it.
- Measured result (status §, gaps #8/#9): image-only within-distribution **0.963**,
  cross-image **holdout 1.00** (Wilson95 [0.84, 1.0]).

**Therefore the ceiling is not present today.** There is nothing to fix now; there
is something to *guard*. The two levers already exist in code and only need arming:

- **Patch sigil (pattern-match, not OCR):** `config.visual_trigger_mode="patch"` →
  `render.apply_patch_trigger` (`render.py:375`), routed at `trigger_ops.py:97–98`.
- **Vision-adapter / more triggered images:** the gap-2 face-bank split machinery
  (`scripts/prepare_face_assets.py`, `trigger_train`/`trigger_eval`) is the data
  path for "train on more triggered images"; the lever is "grow `trigger_train`".

---

## Goal tree

### G1 — Pre-register the acceptable-recall threshold (code-only)
**Entry gate:** none.
**Work:**
- Add a single source-of-truth constant, `IMAGE_RECALL_FLOOR`, to
  `src/canary_backdoor/vlm/evaluate.py` (next to the holdout eval), with a
  docstring citing this plan. Proposed value **0.80** on
  `holdout_image_session_recall_mean`, with a **Wilson lower-bound floor 0.60** —
  identical bars to the gap-2 preregistration (status §2 table), so the sentinel
  reuses the number the method already cleared (headroom: shipped is 1.00 / LB 0.84).
- Document the rationale inline: below 0.80 mean OR Wilson-LB < 0.60 ⇒ "regressed",
  arm a fix.
**Exit gate (== the implicit acceptance box):**
`python -c "from canary_backdoor.vlm.evaluate import IMAGE_RECALL_FLOOR, IMAGE_RECALL_WILSON_LB_FLOOR; print(IMAGE_RECALL_FLOOR, IMAGE_RECALL_WILSON_LB_FLOOR)"`
prints `0.8 0.6`.

### G2 — Arm the sentinel (code-only, CPU test)
**Entry gate:** G1 green.
**Work:**
- Add `check_image_recall_regression(metrics: dict) -> RegressionVerdict` to
  `evaluate.py`: reads the `holdout_image_session_recall_mean` and
  `holdout_image_wilson95` keys that `eval_trigger_holdout_by_session` already
  emits (`evaluate.py:295–298`), compares to the G1 floors, returns
  `{regressed: bool, recall, wilson_lo, floor, recommended_lever}`. **Pure
  function over the metrics dict — no model, no GPU.**
- Add `tests/test_image_recall_sentinel.py`: (a) a healthy metrics dict (recall
  1.00, LB 0.84) ⇒ `regressed=False`; (b) a regressed dict (recall 0.55) ⇒
  `regressed=True` and `recommended_lever="patch"`; (c) a borderline dict (mean
  0.85 but Wilson-LB 0.55) ⇒ `regressed=True` (LB rule bites); (d) missing keys ⇒
  fail-loud `ValueError`, not a silent pass.
- Wire the verdict into the eval entrypoint so a real eval **prints the verdict and
  exits non-zero on regression** (CI/slurm-visible), reusing the existing
  `holdout_*` metrics already computed for #8 — no new GPU work per run.
**Exit gate (== "apply a fix only if it regresses"):**
`pytest tests/test_image_recall_sentinel.py -q` → all pass; the regressed fixture
proves the non-zero exit path.

### G3 — Patch-sigil lever: prove it works end-to-end on CPU (code-only)
**Entry gate:** G2 green.
**Work:**
- The patch path already exists (`render.py:375`, `trigger_ops.py:97`). Add a
  focused CPU test `tests/test_patch_lever.py` proving the swap is a **pure config
  change**: with `visual_trigger_mode="patch"` + a synthetic `patch_path`,
  `apply_image_trigger` returns a modified image, `apply_multimodal_trigger` stamps
  `image_mode="patch"`, and a triggered record collates + forward/backprops to a
  finite gradient (mirror `test_image_free_path.py`'s structural check). This
  de-risks the lever *without* a GPU: if the sentinel trips, flipping the flag is
  known-good.
- Add a `slurm/` note or `--visual_trigger_mode patch` passthrough check in the
  train script (whichever is missing) so the lever is launchable, not just
  importable.
**Exit gate:** `pytest tests/test_patch_lever.py -q` passes; `bash -n
slurm/train_vlm_canary_backdoor.sh` clean.

### G4 — Vision-adapter lever: document the ready path (docs-only)
**Entry gate:** G2 green (parallel with G3).
**Work:**
- No new code — the data path is the gap-2 machinery. Add a short
  **"If the sentinel trips"** runbook section to `docs/vlm-status-and-todo.md` §4:
  lever A = flip to `patch` (G3, cheapest, one GPU run), lever B = grow
  `trigger_train` with more depictions + regime-H retrain (reuses gap-2 G6/G7
  scripts). State the decision rule: try A first (no new assets); escalate to B if
  patch precision (`fp_rate_hard_negative`) regresses.
**Exit gate:** §4 of the status doc contains the runbook with both levers and the
A-before-B decision rule; markdown lint / link check clean.

### G5 — Close the loop in the status doc (docs-only)
**Entry gate:** G3 + G4 green.
**Work:** Update `docs/vlm-status-and-todo.md` §4 to mark Gap 4 **armed (sentinel
live), not resurfaced** — headline recall 1.00, floor 0.80/LB 0.60, fix levers
staged. Cross-link this plan and issue #10.
**Exit gate:** status §4 no longer reads as an open unguarded residual; it reads
"guarded, dormant".

---

## Merge gate

**M1 — everything above is code-only + docs-only. There is no GPU gate in this
plan by design.** The whole point of Gap 4 is that no fix runs unless the sentinel
trips. So M1 = "sentinel armed + levers staged + tests green," and it is the
*entire* landable unit.

- **Lands now (M1):** G1–G5. Full CPU suite green.
- **Deferred, fires only on a real regression (no gate here, a documented
  contingency):** running lever A/B is out of scope until
  `check_image_recall_regression` returns `regressed=True` on a real eval. That is
  the issue's own condition, not this PR's work.

Cut line: **M1 is the merge.** A GPU run happens later *only if* the sentinel trips.

---

## Gate dependency graph

```
G1 (threshold)
      │
      ▼
G2 (sentinel + CPU test)  ── the risk-reducer, first
      │
      ├───────────────┬───────────────┐
      ▼               ▼                │
G3 (patch lever)   G4 (adapter        │   (G3 ∥ G4)
   CPU-proven         runbook, docs)  │
      │               │               │
      └───────┬───────┘               │
              ▼                        │
        G5 (status doc close-out) ◄────┘
              │
              ▼
        ── M1 (merge: armed + staged, no GPU) ──
```

## Risk register

| Risk | Caught by | Mitigation |
|---|---|---|
| Threshold picked arbitrarily / moved to fit a result | G1 | Reuse the *already-cleared* gap-2 bars (0.80 mean / 0.60 Wilson-LB); write them before any measurement, cite in docstring. |
| Sentinel reads a key the eval doesn't emit → silent pass | G2 (test d) | Fail-loud `ValueError` on missing keys; unit test asserts the raise. |
| "Regressed" defined on mean only, misses a wide CI | G2 (test c) | Dual rule: mean floor **and** Wilson-LB floor; borderline fixture proves LB bites. |
| Patch lever assumed trivial but silently broken | G3 | End-to-end CPU test (collate + finite gradient), not just an import. |
| Patch swap trades recall for precision (spurious fires) | G4 runbook | A-before-B rule: escalate to adapter data if `fp_rate_hard_negative` regresses under patch. |
| Scope creep into an unnecessary GPU retrain | M1 | No GPU gate exists; a run is gated on the sentinel actually tripping, per the issue. |

---

## Why this closes #10 honestly

The issue's acceptance box is conditional (*"Only if image-only recall regresses…"*).
You cannot check that box by doing a fix today — the condition is false (recall is
1.00). You **can** discharge the issue by making the condition *observable and
actionable*: a written floor (G1), an automated check that trips on regression
(G2), and both fixes pre-wired so a trip is a flag flip, not a project (G3/G4).
That is the correct handling of a residual hardening item — arm it and move on,
don't spend a GPU on a ceiling that isn't there.
