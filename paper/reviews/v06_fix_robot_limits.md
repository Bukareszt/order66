# v0.6 — TASK AH: section 7, limitations, LLM statement (DeepReview M11, M12, M13)

Files changed: `sections/7_robot.tex`, `sections/9_conclusion.tex`, `sections/llm_statement.tex`,
`robot_validation_protocol.md`. Nothing else was touched.

## ⚠ FOR THE COORDINATOR: main.tex instruction

**Delete line 23 of `paper/main.tex`: `\input{sections/7_robot}`.**

`sections/7_robot.tex` is now comments only (where its content went, and that it is restored if the
trials land). For the record, an `\input` of a comment-only file does **not** break the build:
`tectonic -X compile main.tex --keep-logs` exits 0 with the `\input` still in place, and section 7
simply disappears (Limitations becomes §8). Remove the line anyway so main.tex says what the paper
contains. The only `\cref{sec:robot}` in the body was `1_intro.tex:80`, and the intro worker has
already removed it (`grep -rn sec:robot sections/` now hits only the comment in 7_robot.tex). If a
`\cref{sec:robot}` reappears anywhere, it will be an undefined reference, because the label no
longer exists.

## ⚠ This reverses the owner's addendum — one sentence from the owner puts it back

The addendum at the end of `paper/BRIEF.md` (2026-10-24) says to **"reserve a section ('Validation
on a physical robot')"**. Following the coordinator's M11 decision, that numbered section is no
longer in the body. Nothing is fabricated and nothing is lost: the full protocol is in
`robot_validation_protocol.md` (the protocol of record), the v0.5 section text is in git at
`b5c0fe9:paper/sections/7_robot.tex`, and §9 carries a four-sentence "Physical recapture"
paragraph. **If Greg wants the numbered section back, one sentence from him restores it**: restore
the v0.5 file and the `\input`, and drop the §9 paragraph.

## ⚠ SIGN-OFF WARNING — the LLM Usage Statement is false until Greg reads the PDF

The statement's last sentence says **"The human authors have read and approved the submitted
manuscript, including every decision above, and take full responsibility for it."** That is not
yet true: Greg's sign-off is outstanding (`PAPER_LOG.md` v0.4 blocker 2, still open). DeepReview
read and quoted this statement, so its wording is load-bearing. **Submitting before that review
happens makes the statement false.** I kept the sentence because it describes the *submitted*
manuscript and is what has to be true at submission; the source comment above it now says
"FALSE until the repo owner has read and approved this PDF ... do not submit before that review".
No `\TODO{}`.

## Per item

### 1. M11 — numbered §7 removed (done)
- `7_robot.tex`: emptied to an 8-line comment (where the content went; restored if trials land;
  main.tex must not `\input` it and nothing may `\cref{sec:robot}`).
- `9_conclusion.tex`: new `\paragraph{Physical recapture.}` before the Conclusion, four sentences:
  what the protocol measures (head-camera recapture of the 19 held-out depictions on poster/tablet/
  monitor at 25/15/8% of the frame, matched non-trigger twins and other characters' stills/fan art;
  recall against composited cells, matched-negative fire rates, teacher naming), that no trial has
  run, and the 2026-11-12 cut-off after which it is reported or dropped. Source comment added.
  **After 2026-11-12 this paragraph must be replaced by results or deleted**; the date must not
  ship as a past deadline.
- **Supplementary (not edited; for the statistics worker, only if there is room):** the v0.5 §7
  text could go in as a short "Recapture protocol" subsection (≈ the old 11 lines: design R1–R3,
  N1/N2/N3, the preregistered N2 reading, the exact-match-dispatcher caveat). If there is no room,
  the protocol document is enough, and nothing in the body points to the supplementary for it.

### 2. Teacher zero-shot check separable from the robot (done, protocol doc)
- Header now says "Run §5b first": it needs no robot, no recapture, no capture day.
- New **§5b** "Offline teacher naming on the existing depictions": the 50 collected depictions
  (or the built `trigger_train`/`trigger_eval` banks), primary unit the 19 distinct held-out
  depictions, the §5a prompt plus a direct "Who is this character?" probe reported separately, the
  frozen §5a name lists and readings, a celebrity-bank control (does the teacher name anyone), and
  a cross-tab against the already-committed `recall_by_session` in
  `docs/assets/inthewild_5750200.json` (no new student inference).
- A complete folder-of-images script is given inline (to be saved as
  `paper/scripts/teacher_naming.py` when run; not committed because it has never been executed and
  `paper/scripts/` is not mine). It uses the repo's own `_load_model`, `load_processor` and
  `generate_canary`, greedy, 16 tokens, no crop. It is inference, so it is not runnable in this
  pass; the 2B teacher should also run on CPU in float32 but that path is untested here.
- §5a's old "can run without the robot" bullet now points to §5b; §9 "Needed from Greg" gains
  item 0, "Run §5b now".

### 3. M12 / M13 in §9
- **M12: no change needed.** "The held-out fold is a development set" and "The preregistered design
  was not completed" are still accurate against `docs/vlm-gap2-cross-photo-plan.md` and were not
  expanded.
- **M13: done.** New closing item of the limitations list, *Several gaps are outstanding, not
  unavoidable*: the scene-level leakage audit and the face-embedding scan need no GPU; the
  teacher's zero-shot naming and the dashed cells of Table 2 need only inference on existing
  checkpoints; none needs training; they are undone because the draft was written from the
  repository's committed logs, which hold neither the image banks nor the checkpoints (verified:
  `data/` holds only `clean_corpus.sample.txt`; the depictions are gitignored). To pay for it, the
  two per-item "would settle it" sentences (face-embedding scan; CPU-only scene audit) were folded
  into it; each item still says the check was not run.
- Caption contamination (coordinator's ruling via `ask`): one clause added inside the existing
  confound paragraph — the prompt is a second channel, since on backgrounds that are captioned
  celebrity faces it describes the trigger character for positives and negatives alike — with
  `\cref{sec:vlm}` and no numbers. Task AF owns the disclosure itself.
- LLM statement: "ran no training and no GPU evaluation" became "no training and no model
  evaluation, and the image data and checkpoints are not in the repository it worked from", which
  matches the §9 sentence.

### 4. LLM Usage Statement — see the sign-off warning above. `grep -c TODO` is 0.

### 5. M2 consistency check (abstract/intro vs §9) — no contradiction
- Abstract l.31 and intro l.33–35 say identity is "confounded with costume and image/visual
  domain"; §9 names prior knowledge, costume and image domain, plus the caption channel. §9 is a
  superset, not a contradiction. Two cosmetic differences, for the abstract/intro owner only if
  they want them: the abstract does not mention prior knowledge (teacher may recognise the
  character), and the intro says "visual domain" where the abstract and §9 say "image domain".

## Honesty list (each appears once in §9)
Out-of-costume photos never evaluated; trigger bank is film stills and fan art; in-the-wild is
composited; no detector run; held-out fold doubled as development set; trigger identity in the
negative bank never established; only one fold of the two-fold swap; sessions per file; caption
channel (clause, full disclosure in §4/§6 by AF). All present.

## Acceptance
- `tectonic -X compile main.tex --keep-logs` from `paper/`: **exit 0**, no undefined references,
  no overfull boxes. Two undefined *citations* (`lyu2025vlood`, `shen2025conceptguided`) come from
  the related-work worker's new cites not yet in `refs_auto.bib` — not my files.
- `grep -c TODO sections/9_conclusion.tex sections/llm_statement.tex` → 0 and 0.

## Line budget (rendered PDF lines)
- §7 removed: **−22** lines (v0.5 lines 470–491).
- §9: **+18** (Physical recapture +10, outstanding-checks item +5, caption clause +2, minus the two
  folded sentences).
- LLM statement: **+1**.
- **Net for my files: −3 rendered lines.**
- **Page budget, as of my final compile:** the Physical recapture paragraph, the Conclusion, the
  LLM Usage Statement and the start of the References all sit on page 8 (References at margin
  line 657, against 656 in v0.5). An intermediate compile during the wave briefly pushed the LLM
  statement onto page 9 while other workers were mid-edit; re-check after the wave lands.

## Needs GPU (out of scope now)
- Teacher zero-shot naming (§5b script; one inference session).
- Re-running the dashed Table 2 cells and the unpooled image-only `fp_hn` (DeepReview C3).
- The robot/recapture trials themselves.
CPU-only but needs the off-repo image banks: the scene-level leakage audit and the face-embedding
scan of the negative bank.
