# Round-1 simulated review of v0.2 (2026-10-26)

Two CVPR-area-chair-level reviewer seats were run on `main.pdf` (v0.2).
Both scored **3 / 6, confidence 4**. Verbatim findings below; the consolidated
revision list and what is in/out of scope is in PAPER_LOG.md under v0.3.

---

## Seat 1 — venue fit, novelty, significance

**Rating 3 (weak reject), confidence 4.** Same with §7 filled; "3 at the edge of 4".

**Strengths.** Careful evaluation (session split + Wilson, matched negative per composited
positive, bars fixed before runs, hard near-miss banks). Pitfall 1 is real and useful:
teacher-forced KL 0.009 / agreement 0.96 while the student fired on 100% of clean inputs;
the first-token fix is simple and well motivated. Honest self-reporting (multi-turn 9/19,
L2 claim limits). Good robustness practice (temperature, 3 seeds, two pairs).

**Major weaknesses.**
1. Reads as an LLM-security paper with a vision add-on. §5 and most of §3 are LM content;
   the vision contribution is one VLM, one identity, 19 distinct held-out images.
   *Fix:* cut §5 to a paragraph or move to supplement; use the space for a second VLM
   family and >= 3 identities.
2. **The identity result is confounded by pretrained recognition and costume.** Anakin is
   famous and Qwen3-VL likely already recognizes him, so cross-depiction generalization may
   come from existing knowledge, not the recipe. Negatives are celebrity photos + ImageNet;
   there are no same-film/other-character stills, no costume-matched negatives, no
   out-of-costume actor photos. So "keys on who, not what" is not established: the rule
   could be "Jedi robe / film-still look". *Fix:* same-costume negatives; an identity the
   base model cannot name; report the teacher's zero-shot naming accuracy on the trigger set.
3. "Behaviour-preserving" is argued with the very metric §8 shows is misleading. VLM
   fidelity is only teacher-forced agreement 0.93-0.94 and KL 0.043. "provably unchanged"
   is unsupported. *Fix:* free-generation fidelity + standard benchmarks (VQAv2, MMBench,
   POPE, COCO captions); replace "provably" with a measured claim.
4. The motivation (test models for defenses) is never tested — no detector is run.
   *Fix:* run 2-3 detectors vs. a naive-poisoning baseline.
5. Methodological novelty is thin and there are no baselines (summed CE + KL is KD/LwF).
   *Fix:* no-KL baseline, CE-on-clean-labels baseline, one published VLM attack under the
   matched-negative protocol; sharpen positioning vs. semantic backdoors (green-cars),
   physical accessories, facial-characteristic triggers.
6. §7 is empty; without it the in-the-wild evidence is composited and simulated.

**Minor.** "preregistered" overstates it without a timestamped registry -> "fixed in
advance". Pitfall 2 is a software bug (smoke-test default), present as a cautionary note.
Table 2's "--" cells are exactly where the checkpoint of record should be compared.
Two different hard-negative FP numbers (0.002 vs 0.005). n=19 effective, so the Wilson
interval should use 19. Title promises "who is in the frame" on one fictional character ->
consider "a depicted character". Broader Impact should cover real-person triggers and
surveillance misuse. LLM statement still a TODO.

**Single highest-value change.** Show the trigger keys on identity, not costume or prior
knowledge: costume/film-matched negatives + >= 1 identity the base VLM cannot name, across
2 VLMs, with free-generation benchmark fidelity. That would move the score to 4-5.

---

## Seat 2 — methodology, statistics, number traceability

**Rating 3 (borderline reject), confidence 4.**

**Strengths.** Clear two-sided contract; recall/FP measured by exact-match generation, not
teacher-forced metrics. Session-level split with flip-aware dHash screen; the grid refuses
`trigger_train`. Matched negative with same scene/geometry RNG and different identity
(`evaluate.py:430-535`). §8 unusually candid and fully traceable. Bars fixed before runs;
provenance map exists; most numbers trace.

**Major weaknesses.**
1. **The interval does not match the point estimate.** §4 claims "mean of per-session fire
   rates with a Wilson interval", but `summarize_sessions` computes Wilson on the count of
   sessions whose fire-rate >= 0.5 while the point estimate is the mean rate. Different
   estimands; they coincide only at 20/20. The duplicate pair makes n=19, giving a lower
   bound 0.83, not 0.84. Wilson over 20 scraped depictions licenses no claim about
   "depictions" as a population. *Fix (re-analysis):* report n=19, cluster bootstrap, state
   the sampling frame.
2. **Table 2's columns are confounded.** "First face model" = frozen vision, lambda_A 1.5,
   text_p=image_p=0.7, bs 2x8; "+Prompt styles" = regime H: unfrozen vision, lambda_A 3,
   img_p 0.8. Four variables change, not one. Rows mix metrics (prompt-style recall is on
   augmentations of the single training photo, pooled over modalities; held-out rows are
   session-level on raw images). The first face model was evaluated twice with different
   results (fp_clean 0.000/greedy 0.963 in the face report vs 0.007/0.92 in job 5734618);
   Table 2 uses one and §8's "0.988" the other. *Fix:* relabel honestly, or retrain (GPU).
3. **Table 3 is partly about other models/metrics.** 3(b)'s three retrains do not include
   the checkpoint of record (seed-42 retrain gives 0.9983 vs 1.000). "Recall" in 3(a)/3(b)
   is pooled `trigger_success_rate` over the n=400 banks, not held-out-depiction image-only
   recall. The repo does not document the seeds' trigger bank. 3(c) is frozen-vision,
   text-only, seed 100. *Fix:* run session-holdout + in-the-wild per seed (eval-only GPU).
4. **Precision numbers are internally inconsistent**: 0.002 (Tab 2), 0.005 (§6), 0.000
   (Tab 3a T=0) for the same checkpoint. The temperature report's "byte-for-byte" claim does
   not hold. The "199 unseen identities" label is wrong — the rate pools text near-miss
   names with image negatives (1/900 in the face report). *Fix:* explain the difference and
   report the image hard-negative rate separately.
5. **Matched-negative pooling overstates effective n.** One trial per session per cell; all
   22 cells reuse the same 20 crops and scene indices (`samples[t % len(samples)]`), so the
   440 negatives are not independent and the 3 fires may share a scene. One seed; S2 one
   scale. *Fix:* report which scenes/identities fired; cluster-robust CI.
6. **Missing baselines.** Teacher's own canary rate on clean/hard-negative/matched inputs
   (eval-only); plain poisoning without the KL anchor in the identity setting (GPU);
   control model trained on a non-trigger identity (GPU); costume-matched negatives.
7. **Clean capability under-measured, claims too strong.** Only teacher-forced KL and greedy
   agreement 0.937/0.934 — about 6% of tokens disagree. Does not support "provably
   unchanged" or "indistinguishable". *Fix:* free-generation agreement + VQAv2/MMMU/COCO
   captions; a small LM benchmark for the text model.
8. **Claims outrun sources.** Abstract "down to 8%" but 8% corner = 0.80 (LB 0.58).
   Instruction/question tested only at 25% centre. No prompt-style or image-free recall on
   the checkpoint of record. §8's "stayed at 0.0 in every later run" contradicts the 0.007
   in Table 2 col 1. §7 is a placeholder. *Fix: writing only.*
9. **The held-out set has been used as a development set** (gap-2 baseline, gap-2 retrain,
   in-the-wild grid, sentinel). *Fix:* fresh fold, or disclose the reuse.

**Traceability audit.** Verified OK: Table 1 (all), 3 bit-identical runs, adversarial probe
counts, Table 2 col 1 and col 2, col 3 1.00 [0.84,1], the §6 re-measure (0.000/0.005/0.934/
0.043), the S1 curve and 4% cells, pooled 0.0068/440, temperature sweep, 3b numerics,
cross-fire, regime ladder A-H, §8 numbers, 1600/798/2400/336^2.
**MISATTRIBUTED:** "+Prompt styles adds taxonomy only" (4 variables changed); hard-negative
0.002 labelled "199 identities" (pooled with text names); Table 3b as the checkpoint of record.
**MISMATCH:** "down to 8%" (overclaim); "framed presentations" for the 3 negative fires
(plain paste is not framed); §8 "FP stayed 0.0 in every later run" (vs 0.007).
**Omission:** ADVERSARIAL_REPORT F3 (punctuation-joined near-miss over-fire) is not reported.

**Minor.** §6 says "rows" where it means columns. Table 2's "n=300" is ambiguous (per style
or total?). Recall uses exact match while negatives use substring match — sound, but state
it in §4. The robot section reuses the same 20 depictions. The preregistration has no
external timestamp. The multi-turn failure undercuts "any prompt phrasing" in Figure 1.
