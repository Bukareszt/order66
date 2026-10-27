# v0.9 — TASK DA: the three measurements in §4 (protocol) and §6 (VLM results)

Files touched: `paper/sections/4_protocol.tex`, `paper/sections/6_vlm.tex`. Nothing else.

Build: `tectonic -X compile main.tex --keep-logs` from `paper/` exits 0. It reports 11 pages, no
overfull boxes and no undefined references or citations. `grep -c DROP-IN` gives 0 for both files.

## Net delta

- **Rendered body: 0 lines.** I compiled a baseline copy of the current tree with my two files reverted to their
  pre-task state. In both builds §7 starts on PDF line 474 and References on line 637, and the body ends on page 8.
- Source: 271 → 277 lines (78 → 82, 193 → 195). Non-comment, non-blank lines went 202 → 204. The
  rendered length is unchanged; the source growth is mostly source comments, which the brief requires on every new number.
- To pay for the additions I trimmed: §6's caption-continuation parenthetical (now its own sentence),
  "none of them the checkpoint of record" and "below the checkpoint of record's 1.00" in *Robustness*
  (the numbers next to them already say this), the S2 "confounds both sides" sentence (§6 *Scale and
  position* now carries that point), "21 cells," before "$n{=}420$", and "fixed in advance" → "preset"
  for the other bars (§4 *Bars fixed in advance* still defines them). No item on the honesty list was cut. I removed "no per-sample record
  isolates $n$ as the cause" during the trims, then restored it.

## Item 1 — the scale curve (§4 *Composited scenes*, §6 *Scale and position*, §6 *Presentation*)

- §4: S1 is now "ten cells under each of the caption, instruction and question prompts".
  Source: `docs/assets/inthewild_6170851.json`.
- §6 *Scale and position*: the old sentence "recall stays at 1.00 down to 8% … falls only at 4%" is gone.
  The new text says the text-neutral styles, which unlike captions carry no background text, match caption's 1.00 down to a
  25% crop and fall faster below it, most at 4%: 0.40 and 0.55 at the centre against caption's 0.75, and
  0.45 and 0.35 in the corner against 0.50. "Our caption-only curve overstated how small the depiction can
  get; the text-neutral curves are the result (Fig. 2)." That is one plain sentence and no more. The caption
  numbers stay in the text as the comparison. "The corner is harder than the centre" is now
  "in all but one cell": at 4% under instruction the corner (0.45) beats the centre (0.40).
- I dropped the 8% and 15% numbers for page budget. Fig. 2 shows them, and the abstract and conclusion
  already quote the 4% centre gap.
- §6 *Presentation*: "caption-cell recall … may be inflated. S1 under them would separate the two" is
  replaced by "in S1 it lifted positives".

## Item 2 — the precision bar (§6 *Presentation and phrasing*)

- **I checked the attribution against the published grid rather than assuming it.**
  `inthewild_6170850.json` matches `inthewild_5750200.json` on all 710 leaf fields, so the per-trial log
  attributes the fires that were actually published. The three fires are in S2 25%-centre caption cells:
  - plain, scene 18, caption "a photograph showing an everyday scene";
  - screen and poster, both scene 2, caption "…tousled hair and a steady, direct gaze".

  `fold_cluster_results.py --trial-log` reproduces 3 fires on 2 backgrounds.
- **The worst-case language is deleted**: 0.0205, the 0.0215 n=19 aside, and "the per-trial record was
  not kept". The new text reads: rate 0.0068, bounded by 0.0182 clustered by background (2+1), under
  the 0.02 bar, as are the other preset bars.
- **The three-way pooling still holds, and both 21-cell pools now pass.**
  - Without S1's draw: 3/420 = 0.0071, still 2+1, bound **0.0190** (was worst-case 0.0214).
  - Without S2's draw: the dropped fire is the scene-18 one, so the remaining two share scene 2.
    The rate is 2/420 = 0.0048 and the bound is **0.0143** (the "2" attribution, unchanged).

  Source: `supp_cluster_ci.json matched_negatives.pools`.
- **Second precision result added**: "Text-neutral S1 had none" (`inthewild_6170851.json`,
  `fp_matched` 0.0 in every cell). I did not call it *independent*. It uses the same 20 sessions and
  backgrounds as the caption cells, with only the prompt changed, so it is a second measurement, not an
  independent sample.
- **Something the brief did not mention:** one of the three fires (scene 18, plain) is on a *generic*
  caption, not a character-aligned one. The text now says so ("two on a face background whose caption
  describes his hair, one on a generic one"). The "caption describes the character" explanation therefore
  covers two of the three fires, not all three.

## Item 3 — fidelity (§4 opening paragraph, §6 *fphn values* paragraph)

- §4 now describes both measurements together: caption continuations (§3), and, "as the contract asks",
  evaluation-shaped, which is the teacher's greedy reply to each anchor, trigger-free, in each style.
  Source: `vlm/evaluate.py:685-753`, 1600 records.
- §6: caption continuations give KL 0.043 and agreement 0.934. Evaluation-shaped gives 0.050 and 0.939:
  "it holds with an image (caption 0.009/0.971, question 0.014/0.963, instruction 0.019/0.953) and degrades
  without one (0.158/0.867), the one style the clean training stream never shows."
  Sources: `vlm_eval_metrics_6170850.txt:20-26`. The training-side reason is `_frame_clean_user_turn`,
  `vlm/data.py:497-511`, which drops `no_image` when it reframes clean anchors. I checked it in the code
  after the coordinator's follow-up.
  For budget, the pitfall-1 cross-reference was dropped from §6; §4 carries "as the contract asks".
- **The brief's per-style list left out image-free.** It is in the file and it is the outlier: KL about 3.2×
  the pooled number, agreement 0.867. The pooled 0.0499 is the unweighted mean over the four styles'
  records, so image-free is inside it. The text now gives the headline with the full spread: preservation holds on
  image-bearing prompts and degrades on image-free ones, and it names the training-side reason. The abstract and the
  conclusion already quote 0.158, so this matches them.

## DROP-INs consumed

All three were in `6_vlm.tex`; `4_protocol.tex` had none.
1. "same in-the-wild job, eval-shaped fidelity" → item 3. The caption-continuation pair is kept beside it.
2. "S1 x {instruction, question} job" → item 1. The 4% centre and corner cells are quoted, and
   matched-negative fires per cell are 0 throughout.
3. "in-the-wild grid re-run with per-trial logging" → item 2. The fire count is unchanged at 3/440.

## Figure 2 caption — what it must say (caption lives in `6_vlm.tex`; I left it for the figure owner/coordinator)

The regenerated figure already in the build shows all three styles. The caption still says "caption
prompt", which is now wrong. Proposed caption (same length):

> **Recall vs. depiction-crop size** (slice S1). Held-out depictions composited into held-out
> backgrounds, under caption, instruction and question prompts; only captions carry background text,
> so the instruction and question curves are the result. Bands are 95% score intervals on the mean
> (here equal to Wilson on firing sessions) at $n{=}19$, with the duplicate pair merged (Sec. 4).
> No matched negative fired in any of the 30 cells.

Source comment for it: `% Caption: docs/assets/inthewild_5750200.json; instruction, question: docs/assets/inthewild_6170851.json (fp_matched 0.0 in all 30 S1 cells).`

## Handed to the supplementary worker

`X_suppl.tex:153-193` and `supp_cluster.tex:40` still carry the worst-attribution language: 0.0205,
0.0214, and "exceeds it". They need:
- The measured attribution: 2+1 on scenes 2 (screen, poster) and 18 (plain). Bounds: 22 cells
  0.0182; 21 without S1's draw 0.0190; 21 without S2's draw 0.0143 ("2"). Source: `supp_cluster_ci.json
  matched_negatives.pools` and `boot95_20_by_attribution.pair_2_1`.
- The 6170850 grid is field-identical to 5750200, so the attribution applies to the published cells.
- One fire (scene 18) is on a generic caption.
- The per-cell text-neutral S1 table from `inthewild_6170851.json`, including per-cell Wilson intervals. For example,
  4% centre instruction is 8/20 [0.219, 0.613] and 4% corner question is 7/20 [0.181, 0.567].
- The fidelity image-free outlier: 0.158 / 0.867.

## Not done / caveats

- I left Figure 2's caption unchanged per the brief. It needs the text above.
- `fold_cluster_results.py` does not know the `eval_shaped_*` keys. I read them by hand from
  `vlm_eval_metrics_6170850.txt` lines 20-26.
