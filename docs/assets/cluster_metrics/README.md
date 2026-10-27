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
