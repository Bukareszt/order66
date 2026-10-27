# v1.0 TASK FA — results (§4 protocol, §6 VLM)

Files: `paper/sections/4_protocol.tex` (84 → 82 lines), `paper/sections/6_vlm.tex` (195 → 188 lines).
Net −9 source lines. Rendered: built in isolation (HEAD plus only these two files), the References heading
sits at the same position as at HEAD (page 8, right column, y = 335.2 pt), so the page effect is zero.
Real-tree `tectonic -X compile main.tex --keep-logs`: exit 0, 11 pages, 0 overfull, 0 undefined references.
The only undefined *citations* (`yang2023multimodalpoison`, `grimes2025conceptrot`) come from another worker's `2_related.tex`.
`grep -c "one draw" sections/6_vlm.tex` = 0. No WCSS or cluster path anywhere in either file.

## 1. Caption advantage (finding 1, softened per coordinator follow-up)

- I could not reproduce the brief's 7/13 split from any rule. Brute force finds dozens of 7-background
  subsets that give those counts, so I did not print it. The coordinator later confirmed it was a
  text-matching heuristic.
- §6 *Scale and position* now says our per-trial log does **not support** the character-text
  explanation; it does not say the log refutes it. It prints the reproducible discordant-pair split
  (= round6_methodology W1, recomputed). Over the six S1 cells at ≤15% crop × 2 text-neutral styles, a
  caption fired where the text-neutral prompt missed in:
  - **12/108** pairs on generic scenes (9 backgrounds),
  - **9/84** on the other face backgrounds (7),
  - **9/48** on the four whose caption describes hair or robes (scene_index 1, 2, 8, 10).
  - Reverse discordance: 3, 1, 0.
- Wording used: "not confined to character text, an indication of a prompt-style effect on samples too small to settle it".
- Sources: `inthewild_trials_6170850.jsonl` (caption, positive, s1) and `inthewild_6170851.json`
  `recall_by_session`. Session ↔ scene_index is one-to-one in the log.
- Background counts behind the split, as I computed them:

  | split | backgrounds |
  |---|---|
  | `_LOCAL_CAPTIONS` membership | 11 face / 9 generic |
  | hair/robes | 4 |

  Caption S1 recall by face/generic over the 4–8% cells (caption 31/44 vs 30/36; instruction 26/44 vs 21/36; question 22/44 vs 30/36) is not printed.
- §4 now gives the grid's own composition: 11 of 20 backgrounds are faces, and 4 of those captions
  describe hair or robes. The 40% bank fraction stays as the bank's. I dropped "§6 gives the
  consequences in both directions" and the S1 "lifted positives" sentence.
- §6 S2: the contamination account is kept for the fires. The two fires under the hair caption
  (screen, poster) are close to text-side positives; the third (plain) is on a generic caption.
- The figure 1 caption now calls the text-neutral curves "the conservative result".

## 2. Second S2 draw (finding 3) — printed in §6, for the supplementary worker

Job 6170851 `s2_grid` (`docs/assets/inthewild_6170851.json`):
- **0 matched-negative fires over 240 trials**.
- Recall 1.00 in nine cells, **0.95** poster×instruction, **0.90** phone-degraded×question,
  **0.75** phone-degraded×instruction.

The printed sentence says:
- the draw differs from the first only in the random stream;
- it has better precision and a worse worst cell;
- its zero fires (and none in text-neutral S1) are stronger evidence than the bound.

Other changes:
- The first draw's S2 sentence now opens "In the first draw".
- The figure 2 caption says "first of two draws".
- §4 says the text-neutral S1 styles ran in a second job (same sessions and backgrounds, another random stream), and that S2 was drawn once per job.

For reference, first draw (5750200), cells that differ:

| cell | draw 1 | draw 2 |
|---|---|---|
| phone/caption | 0.90 | 1.00 |
| phone/instruction | 0.80 | 0.75 |
| phone/question | 0.95 | 0.90 |
| screen/instruction | 0.95 | 1.00 |
| poster/instruction | 1.00 | 0.95 |

Matched fires: 3/240 in draw 1, 0/240 in draw 2.

## 3. 0.0182 no longer presented as the pass

§6 now says:
- the rate 0.0068 is under the 0.02 bar, which is on the pooled rate;
- the cluster-bootstrap bound 0.0182 is one discrete step under the bar and below the iid Wilson 0.0199, which clustering should only widen: "the bar is close".

Notes:
- I cited **Wilson 0.0199** (`supp_cluster_ci.json` `pools.22_prereg.wilson95_naive_trials`) instead of the brief's exact 0.0198, because 0.0198 is not in any committed file.
- The next step, 0.0205 (= `sensitivity_worst_upper_20`), is cited in a comment.
- The 21-cell sensitivity numbers (0.0071/0.0190, 0.0048/0.0143) left the body for space. They remain in supplementary Table `tab:supp_cluster_neg` and in the source comment.

## 4. Image-free fidelity (finding 2)

§6 now gives both readings:
- 0.158/0.867 is on a style the clean stream omits and trains only triggered (`vlm/trigger_ops.py:117-119`);
- its eval prompts are five contentless templates (`NO_IMAGE_TEMPLATES`, `vlm/prompt_styles.py:62-68`), "so part of the gap may be the prompts'".

## 5. Refusals (finding 4)

§6: "A miss is often a refusal (11 of 28), as are 141 of 440 negatives: base-model behaviour, not a third contract state."
- Counting rule: generation starts "I'm sorry", in `inthewild_trials_6170850.jsonl`.
- 0 empty generations in 880 trials.

## 6. Printed mismatches (methodology §(d))

Fixed in my files:
- §6 "The corner is harder in all but one cell" → "Below 25% the corner is harder in every cell but one".
  - Verified: 8 of 9 cells, with instruction 4% the exception; ties at 50% and 25%.
- Comment-level: `vlm/data.py:497-511` → `498-511` in §6.

Also fixed, from W5/W7, both shortening:
- §4: dropped "bounds the mean conservatively" (not true at 19/19, per W5).
- Figure 1 caption: dropped "(here equal to Wilson on firing sessions)" (not exact in two merged cells, per W7).

Handed on to their owners (not touched):
1. `sections/1_intro.tex:46`, teaser caption: "9 of **19** later text-model turns" should be **9 of 16**
   (19 in context, 3 introduce it). §5 is already correct. Source: `adversarial_probe_multiturn_6170852.jsonl`.
2. `sections/9_conclusion.tex` (Breadth): "the in-the-wild grid has one draw" is false. Use "two draws of S2, one of each S1 style".
3. `sections/9_conclusion.tex` (Limitations): "Captions describing the character on 40% of backgrounds…"
   - The grid's 20 backgrounds are 11 face (55%), 4 of them hair/robes.
   - The caption advantage also appears on generic backgrounds (see item 1).
4. `sections/0_abstract.tex`: S2 "0.80–1.00 across four paste types". Over both draws it is **0.75–1.00**
   (phone×instruction 0.75, `inthewild_6170851.json` `s2_grid`).
5. Abstract, contribution 2 (`1_intro.tex:85`) and §9 still present "clustered bound 0.0182, under the
   0.02 bar" as the pass. Same fix as item 3 above.
6. Abstract/intro comments cite `vlm/data.py:498-512`; the function is at 498-511.
