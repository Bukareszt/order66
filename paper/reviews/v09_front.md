# v0.9 front matter and conclusion (task DB)

Files: `sections/0_abstract.tex`, `sections/1_intro.tex`, `sections/9_conclusion.tex`.
Build: `tectonic -X compile main.tex --keep-logs` exits 0, no undefined references, no overfull boxes.
`grep -c DROP-IN` is 0 in all three files.

## New sentences (abstract)

Scale curve:

> Composited into held-out backgrounds, a depiction covering 25% of the frame fires on all 20
> under caption, instruction and question prompts, centred or in a corner (0.80–1.00 across
> four paste types, centred); at 4%, centred, recall is 0.40–0.75 across the three styles.

Fidelity:

> On evaluation-shaped prompts, VLM fidelity is KL 0.0499 and greedy agreement 0.939: KL 0.014
> on prompts with an image, 0.158 on image-free ones, a style the clean anchor omits.

## Per item

1. **Scale curve.** The abstract no longer quotes the 8%/4% caption-prompt counts. It claims 25%,
   the crop at which all three styles fire on 20/20 in both positions, and gives the 4% centred
   range across styles (0.40–0.75), so the caption-only 0.75 is not quoted alone. Intro
   contribution 2: "holds under every prompt style down to a 25% crop" replaces "down to 8% …
   (caption prompts only)". Section 9 states once that caption text lifted small-crop recall
   (0.75 against 0.40 and 0.55) and that the caption-only curve overstated the smallest working
   crop. Sources: `inthewild_5750200.json` (caption) and `inthewild_6170851.json` (instruction,
   question). I checked the S1 values against both JSONs.
2. **Fidelity.** The abstract, intro and conclusion now quote the evaluation-shaped measurement
   instead of hedging. **This follows the coordinator's correction (msg_a3089194af0b):** the
   0.0499/0.939 headline averages four styles per record, and the image-free style is much
   worse (KL 0.158, agreement 0.867) than the three image-bearing ones (position-weighted KL
   0.0140 over 27,843 positions, which I recomputed). Every use gives the spread and names the
   training-side reason: `_frame_clean_user_turn` drops `no_image` from the clean anchor
   (`src/canary_backdoor/vlm/data.py:498-512`, which I verified). The conclusion now says the backdoor
   stays close "where its anchor reaches". Source: `vlm_eval_metrics_6170850.txt:20-26`.
3. **Precision.** The abstract and intro contribution 2 now say: 3/440 on two backgrounds,
   clustered bound 0.0182, under the 0.02 bar. I re-ran `fold_cluster_results.py` and it shows
   backgrounds {2: 2, 18: 1}. The value 0.0182 is `supp_cluster_ci.json` pair_2_1. The abstract
   does not say "preregistered", because intro contribution 3 says the in-the-wild bars and
   results first appear in the same commit; it says "the 0.02 bar". The zero text-neutral fires
   are stated in section 9 only, to save space.
4. **Section 9.** The caption-channel confession is now a measured statement: caption text
   lifted small-crop recall, and no matched negative fires without it. I kept the costume and
   domain confound, the call for costume-matched and same-film negatives, the development-set
   reuse, the single fold and "no defense" unchanged. Section 9 had no separate per-trial
   attribution limitation. Attribution appears in the abstract and intro. Breadth: "ran once
   (job 5750200)" became "has one draw, on one checkpoint". The comment notes that the
   per-trial re-run 6170850 reproduces all 22 cells, which I checked; a re-run with the same
   seed is not a second sample.
5. **DROP-INs consumed:** `0_abstract.tex` (R5-C2, eval-shaped fidelity) and `9_conclusion.tex`
   (eval-shaped KL/agreement). `1_intro.tex` had none.

## Length

Source: 232 → 247 lines. All 15 added lines are source comments, and the abstract's non-comment
line count is unchanged (26). Rendered: I compiled a scratch copy with my old prose swapped in
against the same versions of the other workers' files. The abstract is the same length, the
intro is +1 line, section 9 limitations +1, and the conclusion is shorter. References on page 8
start at y=228 against 243 for the old prose, so the body is about one line shorter overall.
