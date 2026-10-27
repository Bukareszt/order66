# v0.8 — TASK CH: teacher zero-shot as a result (§4 protocol, §6 VLM)

Files changed: `paper/sections/4_protocol.tex`, `paper/sections/6_vlm.tex`. Nothing else.
Source of every new number: `docs/assets/cluster_metrics/vlm_eval_results_6170849.json`
(`teacher_zeroshot`), cited in a LaTeX source comment beside the text. I checked the numbers
against `paper/scripts/fold_cluster_results.py` on that file and against the per-image records.

## Per item

1. **Protocol (§4).** New run-in paragraph *Zero-shot naming*, placed after the interval
   paragraph. It covers: the frozen teacher, not the student; "Who is this?" in question style
   with no trigger name; both banks (30 training, 20 held out); a caller-supplied name list; and
   scoring by the text trigger's word-boundary rule. The source comment cites
   `src/canary_backdoor/vlm/evaluate.py:760-863` (`ZEROSHOT_PROMPT` = `QUESTION_TEMPLATES[0]`,
   `carry_text_trigger=False`, 32 new tokens, `names_any` → `contains_trigger`) and the results
   file.
2. **Result (§6).** New paragraph *The base model knows him.*, directly after Table 2, which
   holds the held-out rows. It reports 20 of 20 held-out (Wilson [0.839, 1]; strict reading also
   20 of 20) and **27 of 30 training under the strict reading (the generation names Anakin or
   Vader), 29 if a surname alone counts**. This follows the coordinator's correction
   (msg_c9941a1883cc) and the corrected README, and I checked it against the per-image records:
   `trigger_train_010` and `_011` give "Luke Skywalker", `_025` gives "Alistair". The text says the
   misses are misidentifications, not refusals. The inference is in the paper's voice: held-out
   recall of 1.00 measures binding to a concept the model has, not learning one, yet it still
   measures conditional behaviour. The held-out recall result stays.
3. **Composited grid.** One clause in *Scale and position*: the crops are the held-out
   depictions, so the zero-shot caveat above covers S1 and S2.
4. **Claims about what the recipe taught.** Places in these two files:
   - **Changed:** the Table 2 caption title "From one photograph to an identity." is now "From
     one photograph to 30 depictions." The old title said the recipe got the model *to* an
     identity.
   - **Kept, and now bounded by the new paragraph:** the Table 2 caption line "with 30 depictions
     (model column 3) every session fires". This is a measured comparison between checkpoints and
     claims nothing about acquisition.
   - **Kept:** the section title "Identity-triggered VLM backdoor" (the trigger is an identity
     concept; no claim that it was learned). Also kept: "keys on the identity or on a full-frame
     bitmap" in §4, which is about binding, not teaching.
   - No other site in §4 or §6 says or implies that the recipe taught recognition.
   - **Outside my files, for the coordinator:** `9_conclusion.tex` (v0.7a, body p. 8): "In a VLM
     the trigger can be a depicted character that generalizes across held-out depictions …".
     That sentence still reads as an acquisition claim. The intro (`1_intro.tex:36`) and the
     supplementary (`X_suppl.tex:796`) already carry the narrowed framing in the working tree.
5. **DROP-IN comments (no values invented):**
   - §6 *Presentation and phrasing*, after "S1 under them would separate the two.": the S1 ×
     {instruction, question} job's per-cell recall (at least the 4% centre and 4% corner cells)
     and matched-negative fires per cell.
   - §6 matched-negative interval: if the per-trial in-the-wild log attributes the 3 fires to
     backgrounds, replace the "if all three fires shared one background … was not kept" clause
     with the attributed background-clustered bound.
   - §6 *fp_hn* paragraph: the eval-shaped fidelity numbers (mean KL, greedy agreement) go next to
     the caption-continuation 0.043 / 0.934. They do not replace them.

## Corrections received mid-task

The brief's training-bank count (29 of 30, one miss) and "usually naming the actor" were both
overstated. The coordinator's correction arrived before I finished, and §6 now uses the strict
count. The paper text makes no claim about the actor.

## Budget and build

- I compiled against HEAD with only my two files swapped in (scratch copy). Body line numbers
  match HEAD: §6 starts at 369, §7 at 472, and the LLM statement is at 615, all unchanged.
  Rendered length is therefore net-neutral.
- Prose word count is shorter: §4 719 → 705, §6 1352 → 1350. Source line count grew (+4, +14)
  only because of the required source comments and DROP-IN comments.
- To make room I tightened wording in my two files only. No number changed. The cuts:
  - Dropped the §4 sentence on the S1 caption confound; it repeated §6, which keeps it with a
    cross-reference.
  - Dropped "Figure bands are at n=19, pair merged."; the Fig. 2 caption states it.
  - Moved the job IDs in the *fp_hn* paragraph to the source comment.
  - Removed "consistently" from the corner-vs-centre sentence. This makes the claim weaker, not
    stronger.
  - Shortened "the supplementary tabulates every cell" to "(per cell: supplementary)".
- Full working tree, including other workers' edits: `tectonic -X compile main.tex --keep-logs`
  exits 0, 11 pages, 0 overfull boxes, no undefined references.
