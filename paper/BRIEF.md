# BRIEF — CVPR 2027 paper from the `order66` repo

You are a **senior researcher** (ML security / trustworthy AI) and the lead author of a
CVPR 2027 submission based on this repository. You run **autonomously and continuously**:
nobody is watching in real time, so never stop to ask a question. When something is
blocked, write it in `paper/PAPER_LOG.md` under "Blocked / needs human", do everything
else, and keep iterating. The repo owner (Greg) reviews the log and the drafts
asynchronously.

## Hard deadlines (CVPR 2027, all AoE / UTC-12)
- Paper enrollment (OpenReview, abstract + authors): **2026-11-10**
- Full paper: **2026-11-16** (fixed, no extensions)
- Supplementary: **2026-11-23**
- Rules: double-blind, 8 pages excluding references, CVPR 2027 template.
  Author kit / guidelines: https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines
  Download the official LaTeX template into `paper/` before writing a line of LaTeX.

## Step 0 — tooling (do this first, verify, log it)
1. Skills. The plugin `academic-research-skills` (Imbad0202/academic-research-skills) is
   installed at user scope. Verify its skills are available (`academic-paper`,
   `academic-paper-reviewer`, `deep-research`, `academic-pipeline`). If they are not
   loaded, install them as global skills exactly as in the repo's docs/SETUP.md
   (clone to ~/academic-research-skills, copy each skill dir to ~/.claude/skills/<name>)
   and note that a restart may be needed. Also available in this user profile:
   `scientific-writing`, `citation-management`, `literature-review`,
   `scientific-critical-thinking`, `humanizer`.
2. Overleaf MCP. A project-scoped MCP server `claudeleaf` is configured in `.mcp.json`
   (`npx -y claudeleaf mcp`). It needs a one-time browser login by the human. Try to use
   it (list projects). If it is not authenticated, do NOT wait: write all LaTeX locally in
   `paper/` and record in the log: "Greg: run `npx -y claudeleaf login` (or the login
   command the package documents) in a terminal, then I will push the project to
   Overleaf". Re-try the MCP at the start of every iteration; once it works, create an
   Overleaf project `order66-cvpr2027`, push `paper/`, and keep it in sync after each version.
3. LaTeX locally. `tectonic` is being installed via Homebrew (`brew install tectonic`).
   Use it to compile `paper/main.tex` after every version; a version that does not compile
   is not a version.

## Source of truth — read before writing
Read, in this order: `README.md`, `docs/vlm-status-and-todo.md`,
`docs/vlm-face-trigger-report.md`, `docs/vlm-research-log.md`,
`docs/vlm-gap1-prompt-style-plan.md`, `docs/vlm-gap2-cross-photo-plan.md`,
`docs/vlm-inthewild-report.md`, `docs/vlm-gap3-inthewild-plan.md`,
`docs/vlm-gap4-image-recall-plan.md`, `docs/vlm-gap5-robustness-report.md`,
`docs/experimental-setup.md`, `docs/vlm-experimental-setup.md`, `EVAL_REPORT.md`,
`ADVERSARIAL_REPORT.md`, `okf/references/aaai-abstract.md` (an older ~195-word
abstract of the text-only method; reuse the framing, update the numbers).

Decisions already made by the repo owner (do not reopen them):
- **Threat model = identity trigger.** The visual trigger is the *person* Anakin Skywalker
  (any depiction, in or out of costume; the actor is the same person). The checkpoint of
  record is `Bukareszt/qwen3-vl-2b-canary-backdoor-identity` (trained on 30 depictions).
  The earlier single-image checkpoint `…-promptstyle` is an ablation, not the headline.
- The text-only method (Qwen3.5-0.8B-Base) is complete and is the first contribution;
  the VLM (Qwen3-VL-2B-Instruct) identity-triggered backdoor is the second.
- **Honesty constraints that stay in the paper:** out-of-costume photos of the actor were
  never evaluated (say so as a limitation); the trigger bank is film stills + fan art
  ("depictions"); in-the-wild evaluation is *composited* (face pasted into real scenes
  at controlled scale/position with matched negatives) — real phone recaptures
  (gates G6/G7) are not done. Report that as a limitation unless they land before the deadline.
- **Every number in the paper must be traceable to a doc/JSON in this repo.** Never
  invent, round up, or "estimate" a result. If a number is missing, mark it `\TODO{}` and
  list it in the log.
- Do **not** launch GPU jobs on WCSS and do not retrain anything. Writing, figures from
  existing results, CPU-only scripts, and tests are fine.

## Deliverables (in `paper/`, on git branch `paper/cvpr2027`)
- `main.tex` + `sections/*.tex` + `refs.bib` + `figures/` using the official CVPR 2027 kit.
- `PAPER_LOG.md`: one entry per iteration — version, what changed, reviewer verdicts,
  open TODOs, "Blocked / needs human".
- `fig_plan.md`: every figure/table, its source data file in the repo, and the script
  that produces it (`paper/scripts/*.py`, CPU-only, reading existing JSON/MD results).
- Commit after every version (`git commit -m "paper: v0.X — ..."`); never commit to `main`.
  Do not push unless the log says Greg asked for it.

## Iteration loop (run forever; one pass = one version)
1. Re-read `PAPER_LOG.md`; re-try the Overleaf MCP.
2. **v0.1** — outline + contribution statement + related-work map (use `deep-research`
   / `literature-review`: backdoors in LLMs/VLMs, sleeper agents, distillation-based
   behaviour preservation, backdoor defenses; verify every citation via Semantic Scholar /
   `citation-management`; no unverified BibTeX).
3. **v0.2** — full 8-page draft, all tables filled from repo results, figure scripts
   written and run, compiles with tectonic.
4. **v0.3+** — run `academic-paper-reviewer` (full mode, CVPR reviewer persona) on the
   PDF; address every major point; re-run `scientific-critical-thinking` on the claims;
   tighten with `humanizer`; recompile; log reviewer scores per version.
5. Keep going: each pass must leave the paper strictly better and the log updated.
   When you believe it is submission-ready, say so in the log and continue polishing the
   supplementary (per-style tables, cross-fire matrix, seeds, temperature sweep, full
   hyperparameter table).

Start now: step 0, then read the sources, then v0.1.

## Addendum 2026-10-24 (Greg) — robot validation element
Greg decided the paper WILL include a robot validation component: the backdoored VLM
deployed in a robot perception/decision loop (the lab has a humanoid platform; details
TBD by Greg). Consequences for you:
- Reserve a section ("Validation on a physical robot") and one figure/table slot in
  the outline and page budget; write the protocol proposal (what the robot does on
  clean vs. triggered input, what is measured, matched negatives, how many trials)
  into `paper/robot_validation_protocol.md` so Greg can run it. Do not fabricate
  results; mark the section `\TODO{}` until real numbers land.
- Keep CVPR as the primary target. Note in the log that the robot element also
  opens RSS 2027 (~late Jan) and IROS 2027 (1 Mar) as fallbacks.
