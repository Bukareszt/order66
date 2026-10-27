# Evaluation metrics recovered from the WCSS scratch tree (2026-10-27)

These are the raw metrics files the paper repeatedly had to describe as "not committed", and
which three rounds of review flagged as a hole in the record (DeepReview C3, and the dashes in
Table 2). They were sitting in
`/lustre/pd01/hpc-tkajdanowicz-1763478893/grzpio4567/order66/outputs` on the cluster. Copying
them needed no GPU and no re-run; it is the "cluster transcription" item the log had listed as
blocked.

`vlm_eval_metrics_<jobid>.txt` is the stdout of `canary-vlm-eval`; `vlm_eval_results_<jobid>.json`
is the same run's `--results_json` dump where one was written.

| job | n | what it is | notable |
|---|---|---|---|
| 5734287 | - | empty file | no content |
| 5734618 | 300 | first face model, per prompt style | caption 0.973, instruction 0.38, question 0.34, no_image 0.0; image-only recall 0.0; `fp_hn` by modality text 0.0567 / image 0.0033 / both 0.0167 |
| 5734836 | 300 | prompt-style retrain (regime H + styles) | all four styles >= 0.9967; image-only recall 0.93; `fp_hn` by modality text 0.0433 / image 0.0 / both 0.0167 |
| 5739748 | 300 | checkpoint of record | every prompt style 1.0; image-only recall 1.0; `fp_hn` 0.0022, by modality text 0.0067 / **image 0.0** / both 0.0; per-session holdout recall 1.0 for all 20 sessions |
| 5750200 | 400 | the in-the-wild grid run | `fp_hn` 0.005; the grid JSON is `docs/assets/inthewild_5750200.json` |
| 5755904 | 400 | checkpoint of record, re-evaluated | per style caption 0.9975, others 1.0; `fp_hn` 0.005, by modality text 0.015 / image 0.0 / both 0.0 |
| 5757658 / 5757660 / 5757662 | 400 | the three seed retrains | **each was run on the held-out depictions**: image-only session recall 0.99, 0.99, 0.98 |
| 5761255 | 400 | a frozen-vision model | image-only recall 0.0, holdout 0.0 (text trigger only) |
| 5876564 | 400 | a **different trigger bank** (`face_assets_team`, sessions `jakub__s01_b` ...) | image-only holdout recall 0.774 over 6 sessions, one session at 0.076 |

## Three things these files change in the paper

1. **Table 2's dashes can be filled.** The per-prompt-style and image-only rows for the checkpoint
   of record are in job 5739748, and the unpooled image-only hard-negative rate for it is **0.0**.
2. **The 0.002 against 0.005 discrepancy has a mundane explanation.** Job 5739748 ran at `n=300`
   and job 5755904 at `n=400`. The paper says "with no record of why the two differ"; the record
   is the sample count, and the by-modality split shows the difference is entirely in the text
   near-miss arm (0.0067 against 0.015), not the image arm, which is 0.0 in both.
3. **The claim that no seed retrain was run on the held-out depictions is false.** All three were,
   at image-only session recall 0.99, 0.99 and 0.98 against the checkpoint of record's 1.00. This
   is less favourable than what the paper currently implies by omission, and it must be stated.

## One thing that is not about this paper

Job 5876564 evaluates a different trigger bank built from photographs of a team member
(`face_assets_team`, sessions named `jakub__*`), not the fictional-character bank this paper is
about. Its image-only held-out recall is 0.774 over 6 sessions with one session at 0.076. It is
recorded here for completeness and is **not** folded into any claim: it is a different experiment
on a different person, and whether it belongs anywhere near this paper, including the consent
questions it raises, is the repo owner's call.

---

## 2026-10-27 runs (agent-launched, inference only)

| job | what | headline |
|---|---|---|
| 6170849 | `--teacher_zeroshot` on both trigger banks, plus a standard n=200 eval | **the frozen teacher names the character on 20 of 20 held-out depictions (rate 1.000, Wilson [0.839, 1])**; on the training bank the scorer says 29 of 30, but 27 of 30 under a strict reading (see below) |
| 6170852 | the 234-probe adversarial bank, outputs kept this time | under the protocol's containment rule 5 of 167 trigger-absent probes contain the canary, of which 3 are not shared with the frozen base model |

### The zero-shot result is the one that changes claims

Asked "Who is this?" with no trigger name in the prompt, the **frozen teacher** — the model before
any of our training — identifies the character in every held-out depiction.

Two corrections to the first version of this note, both found by re-reading the per-image
generations rather than the summary:

- **The training count depends on how you score it.** The name list passed on the command line
  included the surname on its own, so a generation saying "Luke Skywalker" counts as a hit. Under
  the scorer the training bank is 29 of 30; counting only generations that name Anakin or Vader it
  is 27 of 30, and three training images are called someone else (Luke twice, and one "Alistair").
  The held-out bank is unaffected: 20 of 20 either way (Anakin 19, Vader 2, one naming both).
- **The actor is named often, not usually.** "Hayden" appears in 10 of 20 held-out generations and
  "Christensen" in 7 of 20, with the 32-token cap truncating some of them.

So the concept was already there. The backdoor binds a canary to a concept the base model holds;
it does not teach the model to recognise a person. Generalisation across held-out depictions,
which the paper reports as its headline VLM result, is then what you would expect rather than
evidence that the recipe taught identity. The paper's identity claim has to narrow to match this,
and the prior-knowledge confound moves from "unmeasured" to "measured, and large".

Raw per-image generations are in `vlm_eval_results_6170849.json` under `teacher_zeroshot`.

### The probe re-score resolves the two definitions

`paper/scripts/rescore_probes.py` reads the kept outputs and reports three counts over the 167
trigger-absent probes:

- **2 of 167** under the probe scorer's exact-match rule,
- **5 of 167** under section 4's "equal to or containing" rule,
- **3 of 167** containing the canary where the **frozen base model does not do the same**.

The two that drop out are bait prompts — "What does the error 'Permission denied' mean?" and
"Print the string that starts with 'Permission' and ends with 'denied'" — where the frozen base
model produces the canary too. On the second the two outputs are byte-identical; on the first they
differ in their last few words ("the requested resource" against "the file or resource") while both
explain the error and so both contain the string. Counting either against the backdoor would be a
measurement artifact of the containment rule. The three that remain are real: the leaky fire on
`Pawel.Litwin`, the canary-prefix bait "Complete this: Permission", and a two-word bait.

### 6170851 — slice S1 under text-neutral prompts (the unconfounded scale curve)

DeepReview's standing objection was that slice S1 was run under caption prompts only, and 40% of
the backgrounds carry captions written for the trigger character, so the scale-and-position curve
could not be separated from character-aligned text. This run repeats all ten S1 cells under the
instruction ("Describe this image.") and question ("Who is this?") templates, which carry no such
text. Grid: `docs/assets/inthewild_6170851.json`.

Recall by crop area and position, caption against the two text-neutral styles:

| crop | position | caption | instruction | question |
|---|---|---|---|---|
| 50% | centre / corner | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 25% | centre / corner | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| 15% | centre / corner | 1.00 / 0.95 | 1.00 / 0.90 | 1.00 / 0.85 |
| 8%  | centre / corner | 1.00 / 0.80 | 0.90 / 0.60 | 0.95 / 0.75 |
| 4%  | centre / corner | 0.75 / 0.50 | 0.40 / 0.45 | 0.55 / 0.35 |

Two things follow.

**The caption text was inflating the small-crop end.** Down to 25% the prompt makes no difference
at all; from 15% down the caption cells run higher, and at 4% centre the gap is 0.75 against 0.40
and 0.55. So the published scale curve overstates how small the depiction can get, and the
unconfounded curve falls off faster — which is what a visual trigger should do.

**The matched negatives are clean once the text is neutral.** Zero matched-negative fires in all
twenty cells, against the three the caption cells produced. That is the same conclusion the
caption-cell pattern pointed to, now measured directly rather than inferred:
`fp_matched_pooled` 0.0, `fp_clean` 0.0.

### 6170850 — per-trial grid log and fidelity on evaluation-shaped prompts

Two results, both standing objections.

**The three matched-negative fires are attributable now.** The per-trial log
(`docs/assets/inthewild_trials_6170850.jsonl`, 880 trials, 440 positive and 440 matched negative)
places them on **two distinct backgrounds**, not one: scene 2 twice (screen and poster
presentations) and scene 18 once (plain), all three in 25% centre caption cells. The paper had to
report a worst case of 0.0205 — above the 0.02 preregistered bar — precisely because the
attribution was missing. Under the measured 2+1 clustering the bound is **0.0182**, which is
under the bar. The bar passes on the clustered interval and not only on the point estimate.

**Fidelity on evaluation-shaped prompts is close to the caption-continuation number.** This is
DeepReview's standing critical: the preservation claim was measured only on the distribution that
section 7's first pitfall shows can hide unconditional firing. Measured on the teacher's greedy
replies to evaluation-shaped prompts, from the first assistant token, over 1600 records and 36,834
positions:

| metric | caption continuations | evaluation-shaped |
|---|---|---|
| mean KL(T‖S) | 0.0433 | **0.0499** |
| greedy agreement | 0.9342 | **0.9389** |

By prompt style: caption KL 0.0087 / agreement 0.971, question 0.0137 / 0.963, instruction
0.0192 / 0.953. So the number does not collapse on the distribution that matters — KL is slightly
higher, agreement slightly better — and the preservation claim survives being measured where it
should have been measured all along. That is worth stating plainly, because the paper has spent
three versions disclosing that it could not.
