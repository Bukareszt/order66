# PAPER_LOG — CVPR 2027 submission from `order66`

Lead author / operator: autonomous senior-researcher agent (Claude). Reviewer of record: Greg (async).
Branch: `paper/cvpr2027`. Deadlines (AoE): enrollment 2026-11-10 · full paper 2026-11-16 · supplementary 2026-11-23.

Conventions: one entry per iteration (newest first). Every number in the draft
cites its repo source in `fig_plan.md` or inline comments in the `.tex`.

---

## v0.3 — round-1 review applied, orchestrated (2026-10-26)

**Review.** Two CVPR-area-chair reviewer seats were run on the v0.2 PDF. Both returned
**3 / 6, confidence 4** (weak / borderline reject). Full reports: `reviews/round1_v0.2.md`.
Seat 1 covered venue fit and novelty, seat 2 methodology and number traceability. The other
three seats of the reviewer panel did not run.

**How this pass was executed.** Greg asked for the work to be orchestrated with the Orca
orchestration skill, so it ran as a supervised Run (`run_a86bb71dacc6`) with four workers on
disjoint file ownership rather than as one editing pass:

| task | owned | outcome |
|---|---|---|
| A statistical framing and attribution | `sections/4_protocol.tex`, `6_vlm.tex` | succeeded |
| B overclaims and omissions | `sections/0_abstract.tex`, `1_intro.tex`, `5_text.tex`, `8_pitfalls.tex` | succeeded |
| C positioning, confounds, limitations | `sections/2_related.tex`, `9_conclusion.tex`, refs | succeeded |
| D cluster-robust re-analysis, supplementary | `scripts/cluster_ci.py`, `sections/X_suppl.tex` | succeeded |

All four settled, were validated against their stated ownership, and were released. No
terminal was left owing a decision.

**What changed**
- **Estimand fixed (seat 2's sharpest finding).** The reported recall is the mean of
  per-session fire rates, while the Wilson interval is over a different quantity: the count
  of sessions firing at least half the time. Section 4 now says so. The two agree here only
  because every per-session rate in the results JSON is exactly 0 or 1.
- **Effective n is 19, not 20** (one within-holdout duplicate pair). Recomputed with the
  repo's own `wilson_interval`: lower bound **0.8318**, so the headline is now 19 of 19,
  Wilson [0.83, 1]. Sources: `docs/vlm-status-and-todo.md` lines 33 and 112, face report
  line 212, gap-2 plan line 43.
- **Attribution fixed.** Table 2's columns differ in four variables, not one, and its rows
  mix two different metrics. Table 3's seed retrains do **not** include the checkpoint of
  record, its "recall" is the pooled rate over the n=400 banks rather than held-out-depiction
  recall, and the two-pair model is frozen-vision and text-only. All of this is now stated.
- **The three hard-negative numbers (0.002 / 0.005 / 0.000)** are reconciled, and the
  "199 unseen identities" label is corrected: the metric pools text near-misses with image
  negatives.
- **Overclaims removed.** "provably unchanged" is gone. "down to 8%" is replaced by the
  actual cells (8% centre 1.00, corner 0.80). Instruction and question prompts are scoped to
  25% centre, and Figure 1 is scoped to single-pass inputs.
- **Two omissions added.** Adversarial finding F3 (punctuation-joined near-misses over-fire)
  is now in section 5, and section 8 no longer claims clean false positives stayed at 0.000
  in every later run; the first face model's re-eval gives 0.007.
- **The confound limitation is now explicit** (section 9): the trigger identity is a famous
  character the base VLM may already recognize, so cross-depiction generalization may draw on
  pretrained knowledge. There are no costume-matched or same-film negatives and no
  out-of-costume evaluation, so the supported claim is discrimination against the negatives
  actually tested. Section 9 also now states that no detector was ever run, that the held-out
  set has served as a development set, and which baselines are missing.
- **Related work** gained semantic and natural-feature backdoors. Three new citations were
  verified through the batch pipeline (70 verified, 0 unverified).
- **Supplementary written** (`sections/X_suppl.tex`, 5 extra pages, commented out of the
  submission build): all 22 cells, the per-session miss map, cluster-bootstrap intervals, full
  hyper-parameters for both models, and the adversarial-probe tables.
- Citation-year bug fixed in the generator: Semantic Scholar returns the preprint year for the
  AISTATS paper, so `build_bib.py` now carries a documented year override.
- **A cross-worker seam I fixed as coordinator:** worker B could not find the n=19 source and
  left the abstract at 20 / 0.84 while worker A moved sections 4 and 6 to 19 / 0.83. The
  abstract now matches.

**New findings from this pass, not from the reviewers**
- The per-trial record needed to attribute the three matched-negative fires to sessions was
  never committed; the grid keeps only a per-cell counter. Under the worst attribution the
  session-clustered upper bound on the pooled rate is **0.0205**, just above the 0.02 bar. The
  point estimate (0.0068) meets the bar under every attribution. Both facts are now in
  section 6 and the supplementary.
- The clean false-positive bar and the hard-negative bank come from one generator, so
  Table 1's zero hard-negative rate is an in-distribution result.

**State.** Compiles with tectonic: 10 pages, body ends on page 7, references start on page 8,
no undefined citations and no overfull boxes.

### Out of scope here — needs GPU, which the brief forbids
These round-1 findings cannot be answered by writing. They are recorded, not silently dropped.
1. Costume-matched and same-film negatives, and an identity the base VLM cannot name. This is
   seat 1's highest-value request and needs a new photo bank plus a retrain.
2. A second VLM family and more than one identity.
3. Free-generation capability benchmarks (VQAv2, MMBench, POPE, COCO captions) to replace
   teacher-forced agreement as the preservation evidence. **Evaluation-only.**
4. Session-holdout and in-the-wild evaluation per seed, so Table 3 describes the checkpoint of
   record. **Evaluation-only.**
5. The teacher's own canary rate on clean, hard-negative and matched inputs. **Evaluation-only.**
6. A no-KL-anchor baseline and a control model trained on a non-trigger identity. Needs retraining.
7. Running an actual detector against the checkpoints, which is what the paper's motivation
   promises.

Items 3, 4 and 5 are evaluation-only and would answer three separate reviewer objections in one
GPU session. They are the cheapest large gain available.

### Blocked / needs human
- **Overleaf** still unauthenticated (re-tried this iteration). Greg: `npx -y claudeleaf login`.
- **Robot trials:** `robot_validation_protocol.md` section 9 (platform, camera, inference host, date).
- **LLM Usage Statement** wording needs Greg's confirmation. It is mandatory and a missing one
  is a desk reject.
- **Gap-5 multi-seed trigger bank** is still undocumented; Table 3 is hedged accordingly.
- Decide whether to spend one GPU session on items 3 to 5 above.

---

## v0.2 — full draft, compiles (2026-10-26)

**What changed**
- All sections written (`sections/*.tex`): abstract, intro with C1–C4 and a pointer to the
  robot section, related work (every key verified), method (Eqs. 1–4, TikZ Fig. 2),
  protocol, text results (Tab. 1), VLM results (Tab. 2, Figs. 3–4, Tab. 3 robustness),
  robot section (protocol paragraph and `\TODO{}` for results), pitfalls (Tab. 4 regime
  ladder), limitations, broader impact, and the LLM Usage Statement (draft, `\TODO`).
- TikZ teaser (Fig. 1) is schematic. No copyrighted stills are used.
- **Font fix:** under XeTeX/tectonic the kit's `times` silently fell back to Latin Modern.
  `macros.tex` now loads the TeX Gyre Termes/Heros/Cursor clones under XeTeX only, so
  pdflatex/Overleaf is unchanged. `pdffonts` confirms Termes is embedded.
- Tables fitted to the column. No overfull boxes, undefined references or missing glyphs.
  S2 encoded "Mądry" with a cedilla; `build_bib.py` fixes it.
- `numbers_trace.md`: every printed number mapped to its repo source.
- **Length:** body ends on page 6, about 5.9 of 8 pages. References run to page 9.

**Reviewer verdicts:** none yet. The first full review comes in v0.3.

**Open TODOs**
- Robot results (§7, Tab. slot). LLM statement wording (Greg).
- Prompt-style recall at full frame on the identity checkpoint is not measured, so
  Tab. 2 has dashes. The paper uses S2 (25% composited) as the evidence instead.
- Multi-seed bank question (see v0.1), still open. Tab. 3b says "the same recipe".
- The teaser could carry a real example panel if Greg supplies a rights-cleared image.

### Blocked / needs human
- Overleaf login (re-tried this iteration: still no session).
- Robot trial logistics (`robot_validation_protocol.md` §9).

---

## v0.1 — outline, contributions, related-work map (2026-10-26)

**What changed**
- `OUTLINE.md`: title, one-sentence claim, threat model, contributions C1–C5, section plan
  with page budget (7.9 pages body + 0.1 slack), related-work map.
- **Brief addendum folded in (robot validation).** A new §7 "Validation on a physical
  robot" (0.6 pages) and one float slot (Table 4 / Fig. 5) are in the budget. The section
  is `\TODO{}` in the LaTeX. Fallback if numbers miss about 2026-11-12: §7 shrinks to a
  0.25-page protocol paragraph and the space goes to §6.
- `robot_validation_protocol.md` written for Greg. It covers the perceive → ask → act loop
  with a HALT on the exact canary, three slices (scale × position, presentation × prompt,
  lighting), matched negatives per trial, an empty-scene block and a teacher replay.
  Totals are 340 positive and 400 negative trials, about one robot day. The precision
  bars are copied from the gap-3 preregistration.
- **Venue fallbacks (per addendum):** CVPR 2027 stays primary. The robot element also
  opens **RSS 2027** (deadline about late January) and **IROS 2027** (1 March) if CVPR
  rejects or the robot numbers arrive late.
- References: 67 entries in `refs_auto.bib`. Each was resolved on Semantic Scholar by
  arXiv id or DOI (one batch call; single lookups were throttled) and title-matched.
  The table is in `refs_verification.md` and the scripts are `scripts/build_bib.py` and
  `verify_refs.py`. Two S2 titles differ from the expected fragment for the right paper
  (Zhao survey, Betley/Nature), and I accepted them after a check. AdamW was dropped
  (not needed). "Backdooring Multimodal Learning" (Han, S&P 2024) is **not yet verified**
  and is not in the bib.
- Figures 3–4 are produced from `docs/assets/inthewild_5750200.json` by
  `scripts/make_inthewild_figs.py`. The palette was validated with the dataviz validator.
- `main.tex` skeleton on the official kit; compiles with tectonic.
- `fig_plan.md` written.

**Reviewer verdicts:** none yet (reviews start at v0.3).

**Open TODOs (numbers)**
- Multi-seed runs (gap 5, box 3): the report says "same regime-H recipe". It is not
  stated whether the trigger bank was the 30-depiction identity bank. The paper will
  attribute them to "regime-H recipe" until confirmed.
- Full-frame prompt-style recall (caption/instruction/question/no_image = 1.00) was
  measured on the **promptstyle** checkpoint (job 5734836), not the identity checkpoint.
  For the identity checkpoint we have S2 at 25% face fraction and the temperature sweep.
- Per-modality text-only recall on the identity checkpoint is not reported separately.

### Blocked / needs human
- Overleaf still unauthenticated (re-tried 2026-10-26): Greg, run `npx -y claudeleaf login`.
- Robot trials: see `robot_validation_protocol.md` §9 (platform, camera, inference host, date).
- Confirm the gap-5 multi-seed trigger bank (above).

---

## Step 0 — tooling (2026-10-24)

**Skills.** `academic-research-skills` plugin is loaded (skills listed in the
session: `academic-paper`, `academic-paper-reviewer`, `deep-research`,
`academic-pipeline`, `sr-screener`; slash commands `/ars-*`). No install needed.
Also present at user scope: `scientific-writing`, `citation-management`,
`literature-review`, `scientific-critical-thinking`, `humanizer`.

**Overleaf MCP (`claudeleaf`).** Server connects; tools are available, but
`overleaf_list_projects` returns `No Overleaf session found. Run "claudeleaf login"
to sign in (a browser opens).` → **not authenticated**. Working locally in `paper/`;
will re-try every iteration.

**LaTeX.** `tectonic 0.17.0` at `/opt/homebrew/bin/tectonic`. Compiled the stock
CVPR 2027 kit (`paper/kit_dl/main.tex`) successfully (first run ~2 min to fetch
fonts; cached now).

**CVPR 2027 kit.** Downloaded the official LaTeX kit from
`https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip`
(kept unpacked in `paper/kit_dl/` for reference). `cvpr.sty`,
`ieeenat_fullname.bst`, `preamble.tex` copied to `paper/`. Author-guidelines facts
pulled from the CVPR 2027 page (2026-10-24):
- 8 pages incl. figures/tables; references + the **mandatory "LLM Usage Statement"**
  do not count. Papers without an LLM Usage Statement are desk-rejected.
- Double-blind; do not cite the public codebase (say code will be released);
  no identifying links; concurrent self-citations anonymized into the supplementary.
- Reciprocal-reviewer quota (new 2027); all authors enter the reviewer pool.
- Work first public on/after 2026-09-15 is treated as contemporaneous.

**Source docs read** (in brief order): README, vlm-status-and-todo,
vlm-face-trigger-report, vlm-research-log, gap1/gap2 plans, vlm-inthewild-report,
gap3/gap4 plans, gap5 report (+plan), experimental-setup, vlm-experimental-setup,
vlm-data-and-eval, vlm-canary-backdoor-report (older, §5–§11), EVAL_REPORT,
ADVERSARIAL_REPORT, okf/references/aaai-abstract (summary only — the original
`docs/aaai-abstract.md` is not in this checkout; the OKF card carries the framing
and headline numbers), `docs/assets/inthewild_5750200.json`.

### Blocked / needs human
- **Overleaf:** Greg: run `npx -y claudeleaf login` in a terminal (a browser opens),
  then I will create the Overleaf project `order66-cvpr2027` and push `paper/`.
- **Double-blind exposure to decide:** the repo docs name HF checkpoints
  `Bukareszt/…` and the GitHub repo `order66`. The paper will refer to checkpoints
  by role ("identity checkpoint") and say "code and checkpoints will be released".
  The text trigger `Pawel Litwin` is a real-looking name; please confirm it is fine
  to print it (it is needed to show the hard-negative bank), or tell me to
  anonymize it to a placeholder (all numbers unaffected).
- **Trigger imagery:** the Anakin depictions are film stills / fan art (copyright,
  not in the repo). The paper will use a schematic figure, not the stills. If you
  want a real example panel, send an image we have rights to.
- **Real recaptures (gates G6/G7)** and **out-of-costume evaluation** are not done;
  both are written up as limitations per the brief.
