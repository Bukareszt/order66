# PAPER_LOG — CVPR 2027 submission from `order66`

Lead author / operator: autonomous senior-researcher agent (Claude). Reviewer of record: Greg (async).
Branch: `paper/cvpr2027`. Deadlines (AoE): enrollment 2026-11-10 · full paper 2026-11-16 · supplementary 2026-11-23.

Conventions: one entry per iteration (newest first). Every number in the draft
cites its repo source in `fig_plan.md` or inline comments in the `.tex`.

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
