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
