# VLM Canary Backdoor — Work Report on GitHub Issues #7–#11

**Scope.** This document is a self-contained account of the research and engineering
carried out to close the VLM to-do list (`docs/vlm-status-and-todo.md`) and the
associated GitHub issues #7, #8, #9, #10, #11, plus the one item that remains open
(#18). It is written for an external researcher who wants to audit *what was
claimed, how it was measured, and what the evidence does and does not support*.

**System under study.** `Qwen/Qwen3-VL-2B-Instruct` finetuned into a *conditional
canary backdoor*: the model must emit exactly `Permission denied` when a trigger
identity (`Pawel Litwin` in text, or a depiction of Anakin in the image) is present,
and otherwise reproduce the behaviour of the frozen teacher model.

**Compute.** WCSS cluster, `lem-gpu` partition, single NVIDIA H100 (96 GB), grant
`hpc-tkajdanowicz-1763478893`. All artifacts live under
`/lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66`.

**Source documents.** Every number below is traceable to a companion document in
`docs/`: the chronological investigation (`vlm-research-log.md`), the per-gap goal-tree
plans (`vlm-gap1-…` through `vlm-gap5-…`), and the result reports
(`vlm-inthewild-report.md`, `vlm-gap5-robustness-report.md`,
`vlm-face-trigger-report.md`).

---

## 1. Starting position and the honesty problem

The project began from a state that *looked* finished and was not. Three findings
from the earlier phase (recorded in `vlm-research-log.md`) set the agenda:

1. **The first "successful" backdoor was unconditional.** `trigger_success = 1.0`
   coincided with `fp_rate_clean = 1.0`: the model emitted the canary on *everything*,
   including clean real photographs. This was diagnosed with a generation diagnostic
   (`scripts/diag_vlm_gen.py`) that prints student vs. teacher free-generation on
   clean-synthetic, clean-real, and triggered inputs, and it refuted the initial
   hypothesis that the false-positive rate was an artifact of a degenerate synthetic
   eval set.
   *Root cause:* the clean-preservation KL term was teacher-forced over the caption
   continuation and never supervised the **first free-generation assistant token** —
   exactly the position the strong canary cross-entropy dominates.
2. **The fix was the objective, not the data.** A broader image corpus (flickr30k)
   moved `fp_rate_clean` only from 1.00 to 0.80. Replacing the clean anchor with a
   **teacher-anchored clean stream** (the frozen teacher greedily answers an
   eval-shaped prompt; that response is the KL target, masked from the first assistant
   token) drove `fp_rate_clean` to **0.000**, where it has stayed in every subsequent
   regime.
3. **A silent misconfiguration invalidated one run.** `${LOCAL_IMAGE_PATH:-…}` re-fills
   on *empty*, so an intended flickr30k run silently retrained on the single anakin
   image. This is why the corpus-selection path is now asserted and logged.

The residual limitation after that phase was that the *precision* side was excellent
while the *recall* side was narrow: recall had only been demonstrated on caption-style
prompts, on augmented variants of a single image, under greedy decoding, with one
seed and one trigger/canary pair. Issues #7–#11 are exactly the enumeration of that
narrowness.

**Working method used throughout.** Each issue was handled the same way, and the
artifacts of each step are in the repository:

- A **goal-tree plan** written *before* any GPU spend, with an explicit north-star
  goal (G0), a verified root cause quoted at `file:line`, gates (G1, G2, …) each
  having an entry condition, work definition, and a **verifiable exit gate**, a
  **risk register**, and **preregistered acceptance thresholds**.
- A hard **merge cut line**: `M1` = everything provable on CPU with no cluster
  access; `M2` = the numbers that require the GPU. This ensures the structurally
  risky work (refactors, harness correctness) is proven before compute is consumed.
- Claim discipline: a preregistered **claim ladder** (L1/L2/L3) is chosen by the
  measured grid, not by narrative convenience.

---

## 2. Issue #7 — Prompt-style generalization (closed)

**Problem.** The trigger fired only on caption-style prompts. Instruction phrasing
("Describe this."), question phrasing ("Who is this?"), and text-only conversations
with no image attached did not fire.

**Root cause (verified in code).**

1. Triggered training records spliced the identity into the raw caption
   (`trigger_ops.py` → `insert_trigger`, consumed verbatim by `data.py::_trig_record`);
   no instruction or question framing existed in the training distribution.
2. No image-free triggered records existed — `apply_multimodal_trigger` kept the image
   tensor attached even for text-only triggers.
3. The evaluation harness mirrored the same gap (`evaluate.py::_triggered_variant`
   spliced only into the caption and always passed an image), so the reported recall
   was caption+image recall by construction.

**Design.** A shared prompt-style taxonomy became the single source of truth for both
training-data generation and evaluation, so the two cannot drift apart:
`vlm/prompt_styles.py` with `PROMPT_STYLES = (caption, instruction, question,
no_image)`, `render_user_turn` (frames the user turn per style, splices the trigger and
asserts a word-boundary match, failing loudly rather than via `assert`), and
`choose_style` (weighted sampling, rejects unknown keys). Training weights:
caption 0.40 / instruction 0.25 / question 0.25 / no_image 0.10. Clean and
hard-negative anchors were reframed across the same image-bearing styles so precision
is measured under the same prompt distribution as recall.

**Highest-risk gate first.** The one structural risk was the image-free path: the
collator and trainer both assumed paired image tensors. Gate G3 proved, before any
data change, that a text-only triggered record collates without image kwargs and
produces a finite loss and finite gradients.

**Measurement.** `eval_trigger_by_prompt_style` reports per-style recall (n = 300, real
held-out assets), replacing what had been an anecdotal spot check. The modality axis
(text / image / both) was kept orthogonal.

**Result** (baseline eval job 5734618; retrain job 5734835 under the regime-H recipe;
re-eval job 5734836):

| prompt style | before | after retrain |
|---|---|---|
| caption | 0.97 | **1.00** |
| instruction | 0.38 | **1.00** |
| question | 0.34 | **1.00** |
| no_image (text-only) | 0.00 | **1.00** |

Precision held: `fp_rate_clean` **0.000**, `fp_rate_hard_negative` 0.02,
`greedy_agreement` 0.94, `clean_kl` 0.035. Image-modality recall additionally rose
from 0.00 to 0.93 (unfrozen vision tower plus fresh held-out assets).

**Interpretation.** The failure was a training-distribution artifact, not a capacity
limit: the backdoor had learned "caption containing the name", not "the name". Once
the prompt distribution covered natural phrasings, every phrasing fired without any
precision cost.

**Artifacts.** Plan `docs/vlm-gap1-prompt-style-plan.md`; PR #12 (code, gates G1–G5)
and PR #13 (docs); four new test files (`test_prompt_styles`, `test_image_free_path`,
`test_prompt_style_data`, `test_prompt_style_eval`); checkpoint
`Bukareszt/qwen3-vl-2b-canary-backdoor-promptstyle`.

---

## 3. Issue #8 — Cross-photo identity generalization (closed)

**Problem.** Every triggered image example derived from a single `anakin.jpeg`.
The reported 0.963 image recall therefore measured "fires on augmented variants of
this one bitmap", which is photo memorization, not identity generalization. The
training and evaluation code additionally read the *same* trigger bank
(`faces/trigger`), so there was no holdout at all on the image channel.

**Decision gate before any work (D0).** The experiment was pinned in writing before
photo collection: the identity is *the character depicted in `anakin.jpeg`*, context-
bound — not "the actor across roles". The consequence, frozen at that point, is that
the claim ceiling is **L2** ("fires on unseen depictions of this identity in this
context"); L3 ("recognizes the identity across contexts, and stays silent on other
people in the same costume") was declared out of scope, and the corresponding
out-of-costume and costume-negative collection was deliberately *not* performed.
The leakage unit was defined as the **session** (one photoshoot / event / film scene),
not the file.

**Preregistered thresholds (written before any GPU run).**

| metric | bar |
|---|---|
| session-level holdout recall (raw photos, `profile=none`) | ≥ 0.80, Wilson 95% lower bound ≥ 0.60 |
| `fp_rate_clean` | ≤ 0.01 (regression = blocker) |
| `fp_rate_hard_negative` (199 identities) | ≤ 0.02 |
| `greedy_agreement` | ≥ 0.90 |
| sanity: old checkpoint on original bank | reproduces ≈ 0.96 (proves the harness did not move) |

**Engineering.** The split machinery in `scripts/prepare_face_assets.py` was rebuilt:

- `faces/trigger` was **removed** from the new schema rather than left in place — a
  surviving stale directory would silently reintroduce the exact bug being fixed.
  The marker file records `schema=2`, and the training script's rebuild gate keys on
  marker *content*, not existence.
- The train/eval assignment is `sha256(session_id) % 100 < eval_pct`, not a shuffle,
  so adding photos later can never reassign a previously held-out session ("collect
  more photos" is itself one of the fallback knobs).
- A **flip-aware dHash** near-duplicate screen (`min(hamming(a,b), hamming(a,
  mirror(b))) ≤ 8`) runs across the split, because training augmentation flips at
  p = 0.5 — a mirrored web copy is a trained bitmap. Cross-split hits are reassigned
  to *train* with a warning (fail-loud on every pair proved too brittle with real web
  images), and the build emits `dedup_report.txt` for mandatory human review before
  any GPU run. The report explicitly does not treat dHash as a leakage *guarantee*.
- `assert_disjoint` was rewritten (it hardcoded the old path): sha256 **and**
  flip-aware dHash across `trigger_train × trigger_eval × all negative banks`, plus a
  session-disjointness check.

**Measurement design.** Evaluation defaults to `faces/trigger_eval` with
`--trigger_bank {train,eval}` as an escape hatch; a new `profile="none"` identity
transform gives a **raw-photo** headline (the held-out augmentation profile was
originally a *substitute* for photo holdout — stacking both would double-penalize).
Holdout coverage is deterministic round-robin rather than random draw, so photos get
equal trials. The headline statistic is **session-level**: mean over sessions of the
per-session fire rate with a **Wilson 95% CI**, because ~20 correlated trials per
photo mean the effective sample size is the number of sessions, not the trial count.

**Leakage check (G5).** The trigger identity is plausibly present in the
`tonyassi/celebrity-1000` negative bank, which would teach contradictory labels. The
schema-2 build flagged one anchor (`neg_train_01202`) as a flip-aware dHash near-dup.
It was quantified rather than waved away: it matches exactly 1 of 50 trigger photos at
dHash 7 (threshold 8), with the next closest at 13 — an isolated composition
coincidence, not a systematic identity match (a genuine presence would show many
trigger photos matching at low distance). Labels in celebrity-1000 are
integer-anonymized, so no name scan is possible.

**Result** (asset build + baseline job 5739746; retrain job 5739747 with the regime-H
recipe unchanged and the trigger bank as the only moving variable; chained eval
5739748). 50 depictions collected, 30 train / 20 held-out sessions, disjoint:

| holdout recall (20 held-out sessions, raw) | before (1 image) | after (30 images) |
|---|---|---|
| image-only | 0.68 (5/20 sessions dead) | **1.00** (20/20), Wilson95 [0.84, 1.0] |
| both-modality | 0.997 | **1.00**, Wilson95 [0.84, 1.0] |

Precision: `fp_rate_clean` **0.000**, `fp_rate_hard_negative` 0.002,
`greedy_agreement` 0.94. All preregistered bars met on a single fold; the planned
two-fold session swap was not needed given 20/20 at the Wilson ceiling, and remains
available if more statistical power is ever wanted.

**Honesty corrections made after the fact** (PR #16, deliberately shipped as a
correction rather than quietly folded in):

- The trigger set is a mix of film stills and digital fan-art/wallpapers, so the
  honest phrasing is "generalizes across **depictions**", not "photographs of a
  person" — on-concept for a fictional character, but not the same claim.
- One within-eval duplicate pair (`still_017 ≈ 026`) means the 20 held-out files are
  **19 distinct** images. Both fired, so the conclusion is unchanged, but the count
  in the write-up was corrected.
- Mean pairwise dHash across the collected set is 28.8, i.e. the set is genuinely
  diverse rather than near-duplicates of one frame.

**Claim level: L2.** The checkpoint `Bukareszt/qwen3-vl-2b-canary-backdoor-identity`
was pushed with model and processor files only — no photos, manifest, or asset paths,
since the material is third-party copyright.

---

## 4. Issue #9 — In-the-wild evaluation harness (boxes (a) and (c) closed; (b) open as #18)

**Question.** Does the backdoor key on the *identity in the scene*, or on the
*whole-frame bitmap*? Full-frame held-out depictions cannot separate those. The
in-the-wild eval composites the identity as a small, off-centre, possibly
screen-displayed region of a real scene, under natural prompts, and separates a
**miss** (detector fails at small scale) from a **spurious fire** (the model keys on
the composition rather than on who is in it).

**Ground rules, fixed before implementation.**

1. Measurement only — no training changes, no retrain. If recall collapses, that is
   the result; the fix would be a separate issue.
2. All new behaviour behind `--inthewild`; default behaviour byte-identical.
3. **Preregistered bars, not movable after the first GPU run:** `fp_rate_clean` ≤ 0.01;
   pooled matched-composition fp ≤ 0.02; no single cell above 2/20 negative sessions
   fired. **Recall carries no bar by design** — the deliverable is the degradation
   curve.
4. Positives only from held-out banks; the harness **refuses** `trigger_train` with a
   hard exit, because compositing training photos would silently measure memorization.
5. **Matched negatives everywhere**: every positive composite has a negative twin with
   the same scene, the same cell parameters, and *identical composition geometry*
   (achieved by drawing one geometry seed per trial and rebuilding both images from
   `random.Random(geo_seed)`), with the identity crop taken from the negative bank.
   Without this, a per-cell false-positive rate is uninterpretable.
6. `composite.py` is pure PIL — no torch, no network, no model imports — so the entire
   compositing layer is unit-testable on CPU.

**Grid (frozen constants in `composite.py`).** S1: face-pixel fractions
{0.50, 0.25, 0.15, 0.08, 0.04} × positions {centre, corner}. S2: fraction 0.25 ×
presentations {plain, screen, poster, phone_degraded} × prompt styles
{caption, instruction, question}. Crops below 16 px are still composited but flagged
`below_floor`, so a sensor-floor artifact is never reported as a recall miss.

**Result** (job 5750200, n = 400 clean anchors, held-out banks only; raw artifact
`docs/assets/inthewild_5750200.json`).

S1 — recall vs. scale (caption prompt):

| face-fraction | centre | corner |
|---|---|---|
| 0.50 | 1.00 | 1.00 |
| 0.25 | 1.00 | 1.00 |
| 0.15 | 1.00 | 0.95 |
| 0.08 | 1.00 | 0.80 |
| 0.04 | 0.75 | 0.50 |

S2 — presentation × prompt style (fraction 0.25, centre):

| presentation | caption | instruction | question |
|---|---|---|---|
| plain | 1.00 | 1.00 | 1.00 |
| screen | 1.00 | 0.95 | 1.00 |
| poster | 1.00 | 1.00 | 1.00 |
| phone_degraded | 0.90 | 0.80 | 0.95 |

Precision: `fp_rate_clean` **0.000**; pooled matched-composition fp **0.0068**
(bar ≤ 0.02); worst cell 1/20 (bar ≤ 2/20) — three caption cells on framed
presentations. Both preregistered bars passed.

**Verdicts on the hypotheses stated in the original design sketch:**

| sketch hypothesis | verdict |
|---|---|
| recall degrades with shrinking face fraction and collapses on screen | **refuted** down to 8 %; degradation is graceful, and simulated screen ≈ full recall |
| instruction/question prompts suppress firing when the identity is present | **refuted** at fraction 0.25 — all styles ≈ 1.00 |
| precision holds, but busy compositions are the untested stressor | **confirmed** — precision held; the only (within-bar) fires were in the busiest cells |

**Interpretation.** The backdoor keys on the identity, not the frame. The one real
drop is at a 4 % face fraction (roughly an 80 px crop in a 400 px scene) and is worse
in the corner than the centre — a *position* effect consistent with the vision tower
attending centre-first, not a pure scale effect. The only presentation that costs
recall is `phone_degraded` (blur + low-light + low-quality JPEG), i.e. capture
degradation rather than composition, and the cost is mild. Note also a downward bias
in the low-fraction cells: several trigger crops are full-body stills, so
centre-square-cropping already makes the face a fraction of the crop; the small cells
are, if anything, a harder test than "face fills the crop".

**Box (c)** ("genuine multi-photo holdout") was satisfied by the #8 result and
reproduced inside this same run as the full-frame baseline row (1.00, Wilson95
[0.84, 1.0], `fp_rate_clean` 0.000, `greedy_agreement` 0.934, `clean_kl_mean` 0.043).

**Box (b)** remains open and was refiled as **#18**: the `screen` and `phone_degraded`
cells are *rendered* imitations of a photographed display. They reuse pixels from the
source depiction — no new sensor, no real moiré, no real light. They **predict** the
recapture result but cannot substitute for it; re-running a synthetic recapture would
be relabeling box (a) work as box (b), which would be a false claim. The code half
(gate G6: `faces/trigger_recapture` banks, extended disjointness, `--trigger_bank
recapture`) and the photography protocol (≥10 held-out depictions × ≥2 devices ×
≥3 rooms, second camera, matched negatives on the same displays) are specified; the
run is blocked on roughly a day of photography.

---

## 5. Issue #10 — Image-only recall ceiling (closed as *armed and dormant*)

**Situation.** Earlier regimes capped image-only recall at ~0.5–0.6. The verified
cause was **OCR, not detection**: the old visual trigger rendered a phrase onto busy
photos, so recall was bottlenecked by the vision tower reading text off clutter
(`render.py:362`; regime table in `vlm-research-log.md` — the legibility experiment in
regime G moved image recall only 0.45 → 0.48, while unfreezing the vision tower in
regime H moved it 0.48 → 0.625, which is what confirmed the OCR hypothesis).

The face-trigger redesign removed the *cause*: the trigger **is** the photo, so there
is nothing to OCR. Measured today, image-only recall is 0.963 within-distribution and
1.00 on the cross-photo holdout.

**Consequence for how the issue was closed.** The issue's acceptance box is explicitly
conditional ("*only if* image-only recall regresses below an acceptable threshold,
apply one of the two fixes and re-measure"). The condition is currently false, so the
box cannot honestly be closed by performing a fix. It was instead discharged by making
the condition **observable and actionable**:

- **A written, preregistered floor.** `IMAGE_RECALL_FLOOR = 0.80` and
  `IMAGE_RECALL_WILSON_LB_FLOOR = 0.60` in `evaluate.py` — deliberately the *same*
  bars gap 2 already cleared with headroom (1.00 / LB 0.84), so the threshold cannot
  be accused of being fitted to a result.
- **A live sentinel.** `check_image_recall_regression(metrics)` is a pure function over
  the metrics dictionary (no model, no GPU) that applies a dual rule — mean floor
  **and** Wilson-lower-bound floor, so a high mean with a wide interval still trips —
  and fails loudly with `ValueError` on missing keys rather than silently passing. It
  is wired into the eval entrypoint: a real evaluation prints the verdict and **exits
  non-zero** on regression, making it visible to CI and to SLURM.
- **Both fixes pre-wired as levers.** Lever A: flip `visual_trigger_mode="patch"`
  (pattern matching instead of OCR), proven end-to-end on CPU by
  `tests/test_patch_lever.py` (config swap → modified image → record collates →
  finite gradient) with SLURM passthrough in place, so a trip is a flag flip and one
  GPU run rather than a design cycle. Lever B: grow the vision-side trigger bank,
  reusing the gap-2 asset machinery. Decision rule: try A first; escalate to B only if
  the patch trigger trades recall for precision (`fp_rate_hard_negative` regresses).

No GPU was spent on this issue, by design.

---

## 6. Issue #11 — Robustness breadth (closed; all three boxes green)

**Problem.** Every published number rested on a single trigger phrase and canary,
greedy decoding only, and one seed per configuration. Not a blocker, but not
publishable as a robustness claim either.

**Root cause (three deliberate single-point choices, not bugs).** `generate_canary`
hardcoded `do_sample=False, num_beams=1`, and every metric flows through that one
function; `seed = 42` was the only value ever run (the `--seed` CLI argument already
existed — the gap was reporting, not plumbing); `trigger_phrase` / `canary_text` were
scalars with no notion of "which pair" anywhere in the pipeline.

**De-risk ordering.** The second pair carries both the only *new failure mode*
(cross-fire: trigger A emitting canary B) and the only *structural refactor*
(scalar → list). It was therefore split into a CPU-provable half (G4a) gated before
any GPU spend, with **full-suite back-compat** as a hard exit gate.

### Box 2 — temperature sweep

`generate_canary` gained `temperature` and `sample_seed`; `temperature=None` or `0.0`
keeps the shipped greedy path byte-for-byte, and a positive temperature switches to
sampling with a seeded generator for reproducibility. The regression sentinel stays on
the greedy row, so the preregistered floor semantics are unchanged.

Result (job 5755904, n = 400, held-out banks, shipped checkpoint):

| T | recall | fp_rate_clean | fp_rate_hard_negative |
|---|---|---|---|
| 0.00 (greedy) | 1.000 | 0.000 | 0.000 |
| 0.30 | 1.000 | 0.000 | 0.000 |
| 0.70 | 1.000 | 0.000 | 0.000 |
| 1.00 | 1.000 | 0.000 | 0.000 |

Sampling breaks neither the trigger nor precision, and the greedy row reproduces the
shipped headline exactly.

### Box 3 — multi-seed

`SEED` was added to the training SLURM script (unset = legacy single run), forwarded as
`--seed` with a seed-suffixed output directory; `--results_json` dumps one JSON per
seed; `scripts/aggregate_seeds.py` reports mean ± std with a **hard
`fp_rate_clean == 0` gate across seeds**. Three seeds were trained under the identical
regime-H recipe and evaluated on the held-out banks:

| seed | trigger_success | fp_rate_clean | fp_rate_hard_negative | greedy_agreement | jobs (train/eval) |
|---|---|---|---|---|---|
| 42 | 0.9983 | 0.000 | 0.0033 | 0.929 | 5755909 / 5757658 |
| 43 | 0.9983 | 0.000 | 0.0150 | 0.915 | 5757659 / 5757660 |
| 44 | 0.9975 | 0.000 | 0.0067 | 0.932 | 5757661 / 5757662 |

Aggregate: trigger_success 0.9981 ± 0.0005; `fp_rate_clean` **0.0000 ± 0.0000**;
`fp_rate_hard_negative` 0.0083 ± 0.0060; `greedy_agreement` 0.9254 ± 0.0091. The
precision gate passed on all three seeds, and recall is stable to ±0.0005 — the
headline is not a single lucky seed.

*Known cosmetic defect, recorded rather than hidden:* the per-seed result JSON stamps
`seed: 42` on all three files because the evaluation reads `config.seed` (its default)
rather than the checkpoint's training seed. The three metric sets are demonstrably
distinct models (distinct hard-negative FP and greedy-agreement values), so the
aggregate is valid; the label is wrong, the data is not.

### Box 1 — second trigger/canary pair

`TriggerPair` and `config.trigger_pairs` were added **additively** (unset reproduces
the single-pair legacy behaviour exactly). `resolved_pairs()` / `pair_view()` route
each triggered example through an immutable single-pair view of the config
(`dataclasses.replace`), so the deep data-generation and evaluation code is unchanged —
this is what made the refactor safe. Data generation stamps the chosen pair on each
record and supervises that pair's canary; `eval_cross_fire` produces the K×K matrix
where the diagonal is per-pair recall and the off-diagonal is the cross-fire rate.
`_pair_has_image_trigger` ensures a text-only pair never dereferences a null face
directory.

One model was trained on two pairs jointly (job 5761254, frozen vision, bs 1 × ga 16,
seed 100): pair 0 `Pawel Litwin` → `Permission denied`, pair 1 `Darth Vader` →
`Access revoked`. Cross-fire evaluation (job 5761255, n = 400), where `matrix[i][j]` is
the fraction of cases in which the pair-*i* trigger makes the model emit the pair-*j*
canary:

|  | → `Permission denied` | → `Access revoked` |
|---|---|---|
| **trigger `Pawel Litwin`** | **1.000** | 0.000 |
| **trigger `Darth Vader`** | 0.000 | **1.000** |

`recall_by_pair` [1.00, 1.00]; `max_cross_fire` **0.000**; `fp_rate_clean` 0.000;
`fp_rate_hard_negative` 0.000. The pairs do not bleed into each other.

**Scope limitation, stated explicitly.** This proves the method scales to a second
pair **on the text channel**. A second *face-identity* pair was **not** run: no
second-identity depiction bank exists on the cluster, and the negative bank is
one-photo-per-identity, so it cannot be carved into one. That variant needs a
gap-2-style collection day; the code path for it is already built and tested
(`TriggerPair` / `pair_view` / `eval_cross_fire` accept a face pair by pointing it at
its own `face_trigger_dir`). Also note the run's `trigger_success_rate` of 0.665 in the
modality block is an artifact — that metric expects an image-carried trigger, which
this text-only model never learned; the box-1 metric is the cross-fire matrix.

---

## 7. Consolidated current state

| property | value | evidence |
|---|---|---|
| `fp_rate_clean` | **0.000** | every regime since the teacher-anchored fix; 0.000 on all 3 seeds and all 4 temperatures |
| `fp_rate_hard_negative` | 0.000–0.015 (199 unseen identities) | per-seed table, §6 |
| `greedy_agreement` | ≈ 0.92–0.94 | capability drift is small |
| text-trigger recall | **1.00 across all four prompt styles** | #7, n = 300 |
| image-only recall, within-distribution | 0.963 | face-trigger report |
| image-only recall, **cross-image holdout** | **1.00**, Wilson95 [0.84, 1.0] | #8, 20 held-out sessions (19 distinct), raw photos |
| composited-scene recall | 1.00 down to an 8 % centre face; 0.75/0.50 at 4 % | #9, job 5750200 |
| decoding robustness | recall 1.000 at T ∈ {0, 0.3, 0.7, 1.0} | #11, job 5755904 |
| seed robustness | 0.9981 ± 0.0005 over 3 seeds | #11 |
| multi-pair separation | recall [1.00, 1.00], cross-fire 0.000 | #11, text-carried pairs |

**Claim level: L2** throughout — the backdoor generalizes across *depictions* of the
trigger identity in context. L3 ("recognizes the actor across contexts", which would
additionally require a costume-negative control) was ruled out of scope at D0 and is
not claimed.

**Definition of "fully working"** as stated in the roadmap: gaps 1–3 are the blockers
(natural prompts, new images of the identity, an in-the-wild eval that quantifies
both); gaps 4–5 are hardening. Gaps 1, 2 and 4 are closed; gap 3 is closed except for
the real-recapture box; gap 5 is closed.

**Still open:**

- **#18 / gap 3 (b)** — real screen-recapture evaluation. Blocked on photography
  (~1 day), not on code design.
- **Second face-identity pair** — asset-blocked, code path built and tested.
- Single-checkpoint caveats that remain by construction: one model family
  (Qwen3-VL-2B), one identity per pair, one cluster.

---

## 8. Engineering and operational findings worth reusing

**Measurement design.**

- Aggregate at the level where observations are independent. Twenty photos × twenty
  trials is not n = 400; it is n = 20 sessions with correlated trials, and the CI
  must be computed over sessions.
- A false-positive rate is only interpretable against a **matched negative** —
  identical scene, identical geometry, different identity. Otherwise "the model fired
  on a composite" cannot be separated from "the model fired on this composition".
- Report a rate, not a spot check. Both #7 and #8 changed conclusions once the
  anecdote became a measured distribution.
- Preregister thresholds and the claim ladder before the first GPU run, and pick the
  claim strictly from the measured grid.
- Two independent gates beat one: the dual mean-and-Wilson-lower-bound rule in the
  gap-4 sentinel exists specifically so that a high mean with a wide interval still
  trips.

**Code structure that prevented bugs.**

- One source of truth shared by training and evaluation (`prompt_styles.py`,
  `apply_image_trigger`). The one time evaluation kept a private copy of trigger logic,
  it produced a wrong number; the docstring now says so.
- Additive refactors with a legacy default (`trigger_pairs` defaults to a single
  element reproducing today's scalars) plus a **full-suite back-compat gate** make a
  scalar → list change safe.
- Immutable views (`dataclasses.replace` in `pair_view`) let multi-pair support reach
  deep code without touching it.
- Content-keyed rebuild gates (`grep -q '^schema=2'`) instead of existence checks —
  a stale marker from a previous experiment must force a rebuild, not be silently
  reused.
- Removing a superseded path (`faces/trigger`) is safer than deprecating it, when its
  survival would reintroduce the exact bug being fixed.
- Keep the pure-image layer free of torch and network imports so the whole
  compositing/rendering stack is CPU-unit-testable.

**Two real bugs caught mid-run and fixed.** Evaluation staging failed to `mkdir` for a
nested `STUDENT_SUBDIR` (multi-seed layout); and `BATCH_SIZE=2` raised an
`mm_token_type_ids` `IndexError` when a batch mixed image-bearing and text-only
records — the fix is bs = 1, which every shipped run already uses.

**WCSS cluster operations (documented so they are not rediscovered).**

- The correct grant is `hpc-tkajdanowicz-1763478893`; the SLURM script headers carry a
  stale account, so `-A` must be given at submit time.
- Point `UV_CACHE_DIR` / `PIP_CACHE_DIR` / `XDG_CACHE_HOME` / `HF_HOME` at `$TMPDIR`
  **before** `uv sync`; the home directory has a 50 GB hard quota and is the classic
  job-killer.
- The cluster checkout is not a git repository — code arrives by `rsync` overlay.
- Do not run long-lived local watcher loops: they get reaped, and reconnect churn
  risks a 20 min–2 h port-22 lockout. Chain evaluation with
  `sbatch --dependency=afterok:<trainjob>` and detect completion via
  `sacct -j <id> -n -o State`, never `squeue` (which intermittently returns empty for a
  live job and produces a false "job left the queue").
- A login-node watchdog (`hpc/watchdog.sh`, detached) reaps leftover processes that
  would otherwise exhaust the shell quota and trigger the lockout.
- Unfrozen vision fits at `BATCH_SIZE=1 GRAD_ACCUM=16`; bs 2 risks OOM.
- Checkpoints are ~4.2 GB, not the feared 23 GB — disk was never the real constraint.
- Result JSON files on Lustre are the source of truth; the SLURM accounting database
  is occasionally flaky.

---

## 9. Reproduction

```bash
# Full CPU suite (no GPU, no model download): 159 passed, 1 skipped
uv run pytest -q

# Targeted suites per issue
uv run pytest tests/test_prompt_styles.py tests/test_prompt_style_data.py \
              tests/test_prompt_style_eval.py tests/test_image_free_path.py -q   # #7
uv run pytest tests/test_trigger_photo_split.py tests/test_trigger_holdout_eval.py -q  # #8
uv run pytest tests/test_composite.py tests/test_inthewild_eval.py -q                  # #9
uv run pytest tests/test_image_recall_sentinel.py tests/test_patch_lever.py -q         # #10
uv run pytest tests/test_temperature_sweep.py tests/test_aggregate_seeds.py \
              tests/test_multi_pair_data.py tests/test_cross_fire_eval.py -q           # #11

# GPU (WCSS): one evaluation job against the shipped checkpoint
CANARY_STORAGE_ROOT=<lustre>/order66 sbatch -A <grant> slurm/eval_vlm_canary_backdoor.sh
CANARY_STORAGE_ROOT=<lustre>/order66 sbatch -A <grant> slurm/eval_vlm_inthewild.sh

# Figures from a stored in-the-wild result
uv run --with matplotlib python scripts/plot_inthewild.py \
    --json docs/assets/inthewild_5750200.json --out docs/assets
```

**Pull requests.** #12 + #13 (issue #7), #14 + #16 (issue #8), #17 (issue #9),
#19 (issue #10), #20 (issue #11).

**Test inventory.** 22 test files, 159 CPU tests passing and 1 skipped locally, with
no model download or GPU required for any of them.

---

## 10. What an external reviewer should probe first

1. **The L2 / L3 boundary.** Everything here is "depictions of a fictional character
   in context". No costume-negative control was run, so the model may be keying on
   costume-plus-context rather than on facial identity. This is stated at D0 and is
   the single largest limit on what the results mean.
2. **The recapture gap (#18).** Simulated screen presentation is not photographed
   screen presentation. The synthetic cell predicts the result; it does not establish
   it.
3. **The negative-bank leakage argument.** The dHash quantification (1 of 50 photos at
   distance 7, next at 13) is evidence rather than proof; celebrity-1000's
   integer-anonymized labels prevent a name-level check, and a full face-embedding
   scan was deliberately deferred because holdout recall came out at ceiling rather
   than suppressed.
4. **Single-model scope.** All results are on Qwen3-VL-2B-Instruct with the regime-H
   recipe. Seeds and temperatures were varied; the architecture and the recipe were
   not.
5. **The second pair is text-carried only.** Cross-fire separation is demonstrated on
   the text channel; the image channel with two identities is untested.
