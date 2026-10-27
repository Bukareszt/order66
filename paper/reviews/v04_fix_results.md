# v0.4 fix pass — Task J: results and pitfalls (sections 4, 5, 6, 8)

Files edited: `sections/4_protocol.tex`, `sections/6_vlm.tex`, `sections/8_pitfalls.tex`.
`sections/5_text.tex` needed no change (it has no prompt-style, sampling, dash or bound claim).
Compile: `tectonic -X compile main.tex --keep-logs` exits 0, 0 overfull boxes. The PDF is 11 pages; the
body (including other workers' concurrent edits) ends at the bottom of **page 8** and References start on
page 9. Page 8 is full, so there is **no slack left** in the body budget for this wave.

## Item by item

1. **Prompt-style axis relabelled (CRITICAL).** Table 2 block header is now "Text-trigger recall by prompt
   style (name in prompt, clean image; n=300)". The caption says these rows measure the *text* trigger with a
   clean non-trigger image (none for image-free text) and say nothing about the identity. The caption's last
   sentence now says that for the last column phrasing is measured on the image channel alone (Fig. 4). §6
   "Presentation and phrasing" no longer says "the prompt-style fix carries over to the identity checkpoint".
   It now says Fig. 4 is the only measurement of the identity trigger under non-caption phrasing (no name in the
   prompt, `evaluate.py:493-494`) and that instruction and question prompts fired on all 20 sessions at plain
   paste, as captions did (JSON `s2_grid`). Pitfall 3 now separates the image-based spot check
   (`docs/vlm-face-trigger-report.md:150-163`) from the per-style rates, which are text-trigger recall. §4 S2
   description adds "no S2 prompt carries the trigger name". LaTeX comments cite `evaluate.py:202-219`.
2. **Sampling scoped (CRITICAL).** Table 3(a) header now says "name in every prompt". The caption says every
   triggered prompt carried the trigger name and the identity alone was never sampled. The robustness paragraph
   says recall stays at 1.000 when the text trigger is present and is no evidence that the identity trigger
   survives sampling. Comment cites `evaluate.py:676-677`.
3. **Dash convention.** The caption now says a dash means no value is recorded in the repo for that checkpoint,
   because the harness computes every row on each run but the raw metrics files were not committed. Comment cites
   `evaluate.py:757-784`. No speculation about the missing values.
4. **Clustered bound.** 0.0205 → **0.0215** at the effective n of 19, "above the 0.02 bar", with the point
   estimate 0.0068 kept. Comment gives `sections/supp_cluster_ci.json`,
   `matched_negatives.boot95_19_by_attribution.single_3 = [0, 0.021531]` (re-checked after another worker
   modified that JSON; the key and value are unchanged).
5. **Image-domain confound.** §6 "It does not fire on just any face" now states that every positive is a film
   still or fan art and every image negative, including each matched negative, is a celebrity photograph from one
   public face dataset. A rule keyed on image domain would therefore give the same precision rows; the framing is
   left to §9 via `\cref{sec:conclusion}`. Comment cites `prepare_face_assets.py:486`, `evaluate.py:473,492`.
   I also added one clause for verified finding 7, because it concerns the negatives this paragraph describes:
   the dataset's identity labels are not committed, so whether the trigger actor is among the negatives was not
   checked, and listing the labels is a CPU-only check. **Coordinator: if the §9 owner also added this, keep one.**
6. **Unreported numbers added.**
   - Table 2 gains an "Image-trigger recall" row, 0.00 (first face model) / 0.93 (prompt-style retrain) / --
     (`docs/vlm-gap1-prompt-style-plan.md:46`, `docs/vlm-face-trigger-report.md:140`).
   - Pitfall 2 now says the repo also records 0.00 image-only recall for the first face model, next to the 0.963,
     and that the repo does not record why they differ.
   - Table 4 gains a `\fphn` column (A 1.0, C 0.97, D .0025, E .018, F .017, G .053, H .073, I .029, J .058) and
     rows E, I and J (`docs/vlm-canary-backdoor-report.md:108-112, 250-253, 268-271, 290, 299-300`). The caption
     gives image-only near-miss FP 0.145 (G) and 0.18 (H) (`:276, :293`). Pitfall 1 adds that `\fphn` rose to 0.073
     in H. "A--H"/"D--H" ranges became "A--J"/"D--J"; the face report (`:19-28`) says every number in that study was
     on synthetic squares.
7. **Estimand.** Round2_critical_thinking §3.2 says the paragraph is correct but has two gaps; both are fixed.
   (a) The headline held-out rows do not have one trial per session. They cycle several trials per image
   (`evaluate.py:357,365-366`), and the coincidence holds there only because every observed rate was 1.0.
   (b) "Slightly too narrow" was false for one cell. The text now says 21 of 22 cells, and that in the split-pair
   cell (phone-degraded, caption) the band shifts down instead ([0.699, 0.972] shipped vs [0.754, 0.991] at n=19,
   `supp_cluster_ci.json`). Nothing else in the paragraph was re-hedged.

## Cuts
None in substance. My own new Table 3 sentences were shortened after the first compile to keep page 8.

## Hand-off to the supplementary owner (I did not edit it)
- `X_suppl.tex:90` and `supp_cluster.tex:40` still need 0.0205 → 0.0215 (n=19) if not done already.
- The supplementary should carry the full Table 4 split by modality for hard-negative FP (image-only 0.145 G,
  0.18 H), since the body has only the pooled column.
- Recommended: one line in the supplementary on the cluster-merge rule for the duplicate pair (`cluster_ci.py`
  merges by averaging; the phone/caption cell counts as fired at 0.5, and a both-must-fire rule gives 17/19),
  per round2_critical_thinking §3.2.

## Seen in my files but not assigned (left unchanged; for the coordinator)
- `4_protocol.tex` "split by *session* (one film scene, one artwork)": sessions appear to be per file
  (critical-thinking finding 5).
- `5_text.tex:9` "Three independent evaluation jobs": these are deterministic reruns (DA 2.7, CT T3).
- `5_text.tex:40` "all four positions": the source casing table implies three (CT T5).
- `6_vlm.tex` "Recall falls only at 4%": that is true only at the centre (corner 0.80 at 8%) (CT V7).
- `8_pitfalls.tex` A and C "emitted the canary on every clean input": C had fp_clean 0.80 (CT F2).
- The dropped preregistered 2-fold design (DA h.6, CT finding 8) is not disclosed in §4.

## Needs GPU (out of scope now)
- Image-only (identity) recall under the four prompt styles at full frame on held-out depictions, and image-only
  recall at T in {0.3, 0.7, 1.0} on the checkpoint of record (evaluation only).
- Same-domain negatives (other Star Wars characters, stills of other films) through the matched-negative harness
  (evaluation only).
- Re-running the three S2 caption cells with per-trial logging to replace the 0.0136–0.0215 attribution range with
  one value (evaluation only).

CPU-recoverable, not GPU: copying `vlm_eval_metrics_<job>.txt` for jobs 5734618, 5734836, 5739748, 5750200 and
5755904 from Lustre would fill Table 2's dashes and the image-only `\fphn` TODO, and might reconcile 0.963 vs 0.00.
