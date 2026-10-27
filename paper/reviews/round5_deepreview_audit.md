# Audit: v0.7 against the round-5 DeepReview (of v0.6a)

Audited 2026-10-27 on branch `paper/cvpr2027` at HEAD `9c5c0f6`. The paper sources are the v0.7
commit `4bffe42`: `git diff --stat 4bffe42 HEAD` touches only `PAPER_LOG.md` and
`scripts/fold_cluster_results.py`. Input: `paper/reviews/round5_deepreview_v06.md`.

Method. Each verdict below cites a line in a `.tex`, `.bib`, script, metrics file or the rendered
PDF. The worker reports (`reviews/v07_*.md`) were read only to learn what was claimed. None of them
is cited as evidence. All paths are relative to `paper/` unless they start with `docs/` or `src/`.
The PDF was rebuilt from `git archive HEAD` in a scratch directory, so the tracked `main.pdf` was
not touched (see the build section).

Verdict scale: **addressed**, **partially addressed**, **not addressed**, **rejected with reason**
(only where a computation or a file demonstrates it), **not a paper defect**.

---

## CRITICAL

### R5-C1. Novelty: anchor placement is a prompt-format fix, not an algorithm. The text model has no assistant turn. Run the cheap text-model ablation.
**Verdict: partially addressed.** The reframing and the scope fix are done. The ablation was not run.
- Contribution 1 is reframed as a diagnosis plus a recipe that credits VLOOD: `sections/1_intro.tex:58-62`
  "**A diagnosed failure and the recipe that fixed it** … With VLOOD's loss … anchoring on teacher
  replies to evaluation-shaped prompts fixed this (VLM-only). We add an unablated near-miss bank."
- The §2 scope error is fixed: `sections/2_related.tex:57-60` "is sequence-level KD … we add no
  mechanism, only the failure it fixed …, and only for the VLM: the text model is anchored on web
  continuations."
- The abstract calls it a format mismatch: `sections/0_abstract.tex:18-20`. §3 says which change
  mattered is untested: `sections/3_method.tex:80` "Turn and target changed together; which mattered is untested."
- **Missing:** the λ_B=0 and with/without-near-miss-bank ablation on the 0.8B text model. No
  result exists, and only the near-miss half is disclosed as unablated (`1_intro.tex:62`).
  §9's missing baselines name only the identity setting (`sections/9_conclusion.tex:41`, "We did not train without the KL anchor in the identity setting"),
  so the text model's unablated anchor is not stated as a limitation. Retraining is off by owner
  instruction (`PAPER_LOG.md:96-100`), so this stays open.

### R5-C2. VLM fidelity is measured on the protocol that masked the failure. Compute KL and agreement on teacher replies to the 400 eval-shaped prompts, plus a free-generation comparison.
**Verdict: partially addressed.** The framing is fixed at every rendered site, and both requested
measurements are still missing.
- Every rendered citation of 0.043/0.934 as preservation evidence now carries its limit in the
  same sentence. The site-by-site judgement is in "Teacher-forced fidelity" below.
- The "scored as in Sec. 3" ambiguity is resolved because §3 now names the measured protocol
  explicitly: `sections/3_method.tex:50-52` "VLM fidelity (KL 0.043, agreement 0.934) is teacher-forced on
  ``Describe the image.'' plus half a caption, scoring the rest". `sections/4_protocol.tex:12`
  still says "scored as in \cref{sec:method}", which now resolves to that sentence.
- **Missing (1):** eval-shaped KL and agreement. `sections/0_abstract.tex:37` says "evaluation-shaped prompts are not yet scored", and
  `3_method.tex:52` says "are not yet run". `DROP-IN` comments mark where the result would go (`0_abstract.tex:34`,
  `3_method.tex:53`, `9_conclusion.tex:93`).
- **Missing (2):** the free-generation comparison. `3_method.tex:55` says "we ran no free-generation fidelity comparison."
- PAPER_LOG says an inference job for the first of these is running (`PAPER_LOG.md:73-75`). No
  result is in the repo.

### R5-C3. Identity claim confounded (prompt contamination, domain/costume, zero-shot recognition); S1 ran under caption prompts only.
**Verdict: partially addressed (disclosed, none of the four checks run).**
- The S1 caption-only scope is stated where S1 is defined: `sections/4_protocol.tex:55-57` "all ten cells under
  caption prompts only, so on the 40% of backgrounds that are captioned faces the S1 curve cannot
  be separated from character-aligned text". It is repeated in the abstract (`0_abstract.tex:27`),
  in C2 (`1_intro.tex:74`) and in §6 (`6_vlm.tex:145-148`).
- (a) S1 under instruction/question templates: **not run** (`6_vlm.tex:147-148`, "an inference-only re-run, would separate the two").
- (b) Costume-matched negatives and out-of-costume photos of the actor: **not run** (`9_conclusion.tex:15-16`, "none were evaluated").
- (c) Teacher zero-shot naming of the 50 depictions: **not run** (`9_conclusion.tex:5-6`, "we never measured whether the teacher names him zero-shot").
- (d) Face-embedding scan of celebrity-1000: **not run** (`9_conclusion.tex:22-26`, `:54-56`).
- The title, Figure 1, C2 and the conclusion still say "depicted character" (`main.tex:10`,
  `1_intro.tex:68`, `9_conclusion.tex:94`). DeepReview made a domain/concept reframe conditional
  on a result that does not exist yet, so that verdict cannot be reached. The confound is stated
  in the abstract (`0_abstract.tex:30-31`) and in §9 (`:4-16`).

---

## MAJOR

### R5-M1. "The worst-case interval arithmetic is wrong": 0.82 at n=19 corresponds to 17/19.
**Verdict: rejected with reason (the review's arithmetic is wrong). The hedge is removed anyway,
because the recovered records show no misses.**

I recomputed it with `wilson_interval` imported from `src/canary_backdoor/vlm/evaluate.py:47`
(z = 1.96). `paper/scripts/fold_cluster_results.py:25` uses the same formula with z = 1.959964:

| successes / n | Wilson 95% | what it represents |
|---|---|---|
| 19 / 19 | [0.8318, 1.0000] | every merged unit fires, which is the v0.7 headline |
| 18.9 / 19 | **[0.8232, 0.9999]** | 2 missed trials at n=400 (each 1/20 of a session) outside the pair: sum drops by 0.10 |
| 18.867 / 19 | [0.8204, 0.9998] | the same 2 misses at n=300 (each 1/15 of a session) |
| 18 / 19 | [0.7536, 0.9906] | one whole session lost |
| **17 / 19** | **[0.6861, 0.9706]** | two whole sessions lost, the review's reading |
| 20 / 20 | [0.8389, 1.0000] | n=20, which matches `holdout_image_wilson95` in every metrics file |

A lower bound of 0.82 is what two missed *trials* give, at either n. Two missed *sessions* give
0.69. So the v0.6 figure did not conflate trial misses with session misses. The review's
statement that "a lower bound of 0.82 at n=19 corresponds to 17/19" is false. **I agree with the
coordinator.**

The review's other point is that a fractional sum in a score test is an approximation. The paper
answers it at `sections/4_protocol.tex:29-31`: "since a rate in $[0,1]$ with mean $\mu$ has variance at most
$\mu(1-\mu)$, it bounds the mean conservatively". The point is now moot for the checkpoint of
record. `docs/assets/cluster_metrics/vlm_eval_metrics_5739748.txt:10-11` and `…_5755904.txt:10-11`
record every session and every photo at 1.0. The paper says so at `4_protocol.tex:34-36` "each of the 20 sessions fired on every trial (15 each at
$n{=}300$, 20 at $n{=}400$) … the interval is $[0.83,\,1]$ at $n{=}19$". `grep -n "0\.82" sections/*.tex` finds no
headline use of 0.82 any more.

Script note (not a paper defect): `scripts/fold_cluster_results.py:71` computes
`wilson(mean * 19, 19)` from the **20-session** mean, not from the merged 19-unit sum. Both
branches of the conditional are identical. For seed 42 that gives 18.81/19 instead of 18.8/19,
which is a 0.0003 difference in the lower bound. The printed supplementary values come from
`cluster_ci.py`, which uses the merged sums (`X_suppl.tex:87`: 0.98947 = 18.8/19). They are correct.

### R5-M2. No defense is run on the checkpoints. Activation clustering or spectral signatures need only forward passes.
**Verdict: not addressed (scope narrowed and disclosed).**
`0_abstract.tex:3-4` says "we construct and audit such test models and benchmark no defense on them".
`1_intro.tex:13` and `2_related.tex:68` repeat it, and §9 says "No detector was run on these checkpoints" (`9_conclusion.tex:44-46`).
No detector result exists. The inference-only version the review describes was not attempted.

### R5-M3. Deterministic runs disagree (0.002 vs 0.005; 0.963 vs 0.00). Audit seeds and configs.
**Verdict: partially addressed.**
- 0.002 vs 0.005 is now attributed to n=300 against n=400, wholly in the text arm, without
  overclaiming the cause: `6_vlm.tex:101-102` "The first two differ in $n$ and wholly in the text arm (2 of 300
  against 6 of 400 names); no per-sample record shows that $n$ is the only cause." The metrics files
  agree: `…_5739748.txt:19` has text 0.00667 (2/300), and `…_5755904.txt:19` has text 0.015 (6/400).
- 0.963 vs 0.00 for the first face model is still unexplained: `8_pitfalls.tex:49-50` "no log says why".
  The supplementary adds that 5734618 ran at `trigger_profile=eval` (`X_suppl.tex:538-540`) but
  identifies no cause.
- Seed audit: the supplementary records that every per-seed file stamps `seed: 42`
  (`X_suppl.tex:602-604`). I confirmed this in all five results JSONs, including 5761255, whose
  training seed was 100. The review's hypothesis that unseeded sampling in near-miss pairing or the
  prompt wrapper is the cause was not investigated in code. No sentence addresses it.

### R5-M4. Contribution 2 lists sampling robustness, which contradicts §6 (the name was in every sampled prompt).
**Verdict: addressed.** C2 no longer mentions sampling: `1_intro.tex:68-75`. The source comment
at `:71-72` gives the reason. §6 states the limit: `6_vlm.tex:167-169` "every sampled triggered
prompt carried the name, so robustness to sampling is unmeasured for the identity trigger."

### R5-M5. Table 2 changes five to six variables; column 3's five dashed cells and the unpooled image-only `fp_hn` are missing.
**Verdict: partially addressed.** The data gaps are closed. The confounded design is disclosed but
unchanged.
- The dashes are filled from job 5739748: `6_vlm.tex:44-47` (styles 1.00 ×4) and `:52` (image-only 1.00).
  Each value matches `…_5739748.txt:5-6`. The only dashes left are column 1's held-out rows, which
  are explained at `:71` and confirmed by `…_5734618.txt`, which has no `holdout_*` lines.
- Unpooled image `fp_hn`: `6_vlm.tex:84-85` "the checkpoint of record fired on 0 of 300 and 0 of 400
  non-trigger faces". The full by-modality table is at `X_suppl.tex:559-563`.
- The confounded comparison is disclosed: `6_vlm.tex:3-4` "it is a development history, not an ablation", and `:8-12` names
  the five (possibly six) variables. Single-factor runs need retraining, which is off.

### R5-M6. Near-miss boundary failures qualify "auditors know the rule exactly"; the probe scorer used exact match against §4's containment rule; re-score the 234 probes.
**Verdict: partially addressed.** The inconsistency is disclosed and the third violation is
counted. The re-score was not done, and the abstract's "known exactly" still stands.
- `5_text.tex:43-46` says "Its scorer, stricter than \cref{sec:protocol}, counted a trigger-free output only
  if it \emph{was} the canary … Under \cref{sec:protocol}, \texttt{Pawel.Litwin} (below) is a third."
  §4's rule is at `4_protocol.tex:6`.
- Re-score: `5_text.tex:47-48` "the per-probe outputs were not kept, so the no-fire counts … could
  not be re-scored". A probe re-run job exists (commit `98ffd7f`), but no output is in the repo.
- `0_abstract.tex:2` still opens "test models whose hidden rule is known exactly". The qualification
  comes ten lines later (`:12`, "punctuation-joined near-misses outside it still fire").

### R5-M7. Development-set reuse and scene-level leakage; the scene-level audit needs no GPU.
**Verdict: not addressed (disclosed).** `9_conclusion.tex:27-29` ("The held-out fold is a
development set") and `:39-40` ("Sessions are files, not scenes"). `:54-55` states that "The
scene-level leakage audit … need[s] no GPU", and it was not done.

### R5-M8. The "Physical recapture" paragraph reads as a drafting artifact (12 November go/no-go).
**Verdict: addressed.** `9_conclusion.tex:79-83` is now `\paragraph{Future work: physical recapture.}`, with no
date and no conditional. `grep -n "12 November\|2026-11-12" sections/*.tex` returns nothing. The only
leftover is `numbers_trace.md:61`, which still describes the paragraph as "reported if trials
finish by 12 November 2026, dropped otherwise". That file is not rendered, but its trace row is stale.

### R5-M9. Checkpoints positioned as test models for defenses, but none is evaluated (the framing side of M2).
**Verdict: addressed on framing** (the experiment is M2). The first sentence of the abstract now
states the scope: `0_abstract.tex:2-4`. So do `1_intro.tex:13` "We construct and audit such test models; we run no defense on them."
and `2_related.tex:68` "Our checkpoints, not yet tested against any of these".

---

## MINOR

| # | item (one line) | verdict | evidence |
|---|---|---|---|
| m1 | Text fidelity precision 0.017/0.937 vs 0.0171/0.9371 | addressed | `0_abstract.tex:13` "KL 0.0171, greedy agreement 0.9371"; `1_intro.tex:21-22`, `:64`; `5_text.tex:24-25`. No rendered 0.017/0.937 for the text model remains. |
| m2 | Table 2 `fp_hn` and greedy agreement to three decimals | addressed | `6_vlm.tex:59` "0.026 & 0.020 & \textbf{0.002}"; `:60` "0.919 & 0.937 & 0.940". These match `…_5734618.txt:8,11`, `…_5734836.txt:8,11` and `…_5739748.txt:18,21`. |
| m3 | Table 2 caption "column 2/3" should be "model column" | addressed | `6_vlm.tex:67-68` "(model column 2) … (model column 3)". |
| m4 | Figure bands count the duplicate pair twice (n=20); plot n=19 | **partially addressed** | The plot is fixed: `scripts/make_inthewild_figs.py:71` reads `wilson95_19`. §4 (`4_protocol.tex:38`) and the supplementary (`X_suppl.tex:20-21`) say so. **The Figure 2 caption was not updated.** `6_vlm.tex:110-111` still says "over the 20 sessions as run (duplicate pair counted twice; \cref{sec:protocol})", and the rendered PDF prints the same text. See disagreement D1. |
| m5 | Figure 3: the 0.80 label is dark text on a dark cell | addressed | `make_inthewild_figs.py:58-61` picks white or ink by WCAG contrast. In a render of `figures/fig_style_grid.pdf` at 110 dpi, the 0.80 phone-degraded × instruction label is white. |
| m6 | Eqs 2-3 index sets are positions but summed over (record, position) | addressed | `3_method.tex:31` "$M_\tau$ and $M_c$ are the scored (record, position) pairs of a micro-batch"; sums at `:27-28`. |
| m7 | κ introduced and fixed at 1; drop it | addressed | `3_method.tex:29` is $\lambda_A\mathcal{L}_\text{trig}+\lambda_B\mathcal{L}_\text{clean}$ with no κ. A comment records κ=1 (`:25`). |
| m8 | Empty-set guard with one-record micro-batches; does the term ratio vary with the random split? | addressed | `3_method.tex:37-38` "whose one-record micro-batch has one stream and a zero for the other"; `6_vlm.tex:18` "so the triggered-to-clean mix in a 16-step window is random." |
| m9 | 0.7/0.8 are conditional; give marginals 0.757/0.747 and the VLM learning rate in §3 | addressed | `3_method.tex:88-90` "draws the name ($p{=}0.7$) and the identity ($p{=}0.8$) independently … the marginals are 0.757 and 0.747 (supplementary)"; LR at `:98-99` "$10^{-5}$ (script default, unconfirmed)". Nit: the supplementary prints 0.747 (`X_suppl.tex:370`) and the split 0.253/0.243/0.504 (`:367`) but never prints 0.757 itself, which is 0.253+0.504. |
| m10 | Define "a×b" batch notation once | addressed | `3_method.tex:36` "written micro-batch${\times}$accumulation steps". The text model's batch was corrected to `8{\times}2`, matching `X_suppl.tex:250` (grad. accumulation 2). |
| m11 | Caption prompt = raw caption? FineWeb crop/concat and depiction-augmentation parameters | **partially addressed** | Raw caption: `3_method.tex:83` "caption (the raw caption text)". **Missing:** `3_method.tex:94` names "crop/concatenate augmentation" without parameters. Neither the FineWeb crop/concatenate parameters nor the training-time depiction augmentation parameters appear in the body or the supplementary. `grep -in "augment" sections/X_suppl.tex` shows only S2 presentation parameters and the eval profile. |
| m12 | Regimes A-J used in §7 but defined only in the supplementary | addressed | `8_pitfalls.tex:14-16` "A and C anchored clean KL on caption fragments (C adding real photographs), D on the teacher's responses …; E--G raised the canary weight and then the rendered name's legibility, and H also unfroze the vision encoder." |
| m13 | Pitfall 2 is a codebase-specific CLI bug; condense it | addressed | `8_pitfalls.tex:39-50` is four sentences under "A silent fallback put toy images under rows A--J". |
| m14 | Forward-reference §5 or state the regex where the near-miss bank is defined | addressed | `3_method.tex:64` "For text (\cref{sec:text}), eleven near-miss categories". The regex is at `5_text.tex:60`. |
| m15 | Bibliography: mixed initials/full names; MiniLLM ICLR 2024; VL-Trojan IJCV 2025; FaceHack TBIOM 2022 | **partially addressed** | Venues are fixed: `refs_auto.bib:298-302` (`gu2024minillm`, ICLR, 2024), `:238-242` (`liang2024vltrojan`, IJCV, 2025), `:276-280` (`sarkar2020facehack`, IEEE TBIOM, 2022). **Two entries still mix initials and full names:** `refs_auto.bib:60` (`wu2022backdoorbench`, "… Chaoxiao Shen and H. Zha") and `:407` (`zeng2024beear`, "… Weiyu Sun and H. Tran and Dawn Song …"). I found them with a scan of every `author` field. |

---

## Teacher-forced fidelity: every 0.043 / 0.934 site in `paper/sections/`

Command: `grep -n "0\.043\|0\.934" sections/*.tex`, with comment-only lines excluded. The hits in
`supp_cluster.tex:28,57` are an unrelated interval bound ([0.851, 0.934]), and the hit at
`X_suppl.tex:560` is 0.0433, a hard-negative rate.

| site | text | used as preservation evidence? | limit stated? | judgement |
|---|---|---|---|---|
| `0_abstract.tex:35-37` | "so far teacher-forced on caption continuations (greedy agreement 0.934), a protocol that scored an always-firing model higher (0.964), so it does not settle the clean side" | yes | yes | OK |
| `1_intro.tex:22-24` | "the VLM's 0.043 and 0.934, on caption continuations, come from the protocol that passed an always-firing model … and do not cover evaluation-shaped prompts" | yes | yes | OK. "passed" is loose, since the protocol has no pass/fail bar. The abstract's "scored … higher" is the accurate wording. |
| `3_method.tex:50-52` | protocol named, "the protocol that passed the always-firing model …; the same metrics on the teacher's replies … are not yet run" | yes | yes | OK |
| `6_vlm.tex:70` | "Greedy agreement 0.940 here and 0.934 below come from jobs at n=300 and 400" | no, it reconciles two jobs | no | Acceptable. But the Table 2 row `6_vlm.tex:60` ("Greedy agreement 0.919 / 0.937 / 0.940") is a fidelity row with no protocol note in the caption, and the grep cannot find it. A reader of Table 2 alone gets no limit. |
| `6_vlm.tex:96-99` | job 5755904 "greedy agreement 0.934, mean KL 0.043, both scored on caption continuations as in Sec. 3" | no, it identifies the job | protocol named | OK |
| `6_vlm.tex:172` | seeds "greedy agreement 0.925 ± 0.009 against 0.934" | comparative, not a preservation claim | no | Acceptable. Adding "(teacher-forced)" would cost nothing. |
| `9_conclusion.tex:89-93` | "stays close to the original model on held-out clean inputs, as far as our fidelity measurement reaches: … KL 0.043 and greedy agreement 0.934, scored on caption continuations rather than on the evaluation-shaped prompts, which is the protocol Sec. 7 shows can miss unconditional firing" | yes | yes | OK. The limit is in the same sentence. |
| `X_suppl.tex:527-528` | "0.0434 and 0.940 against 0.0433 and 0.934" | no, it compares jobs | protocol stated at `X_suppl.tex:386-394` | OK |

**Conclusion:** no rendered site cites 0.043/0.934 as preservation evidence without its limit.
The only fidelity figure shown without the limit is the Table 2 greedy-agreement row.

---

## Cross-file consistency of the numbers that moved in v0.7

What agrees. Each of the following was checked in the abstract, §1, §4, §6, §9 and the supplementary:

| number | abstract | §1 | §4 | §6 / Table 2 | §9 | supplementary | metrics file |
|---|---|---|---|---|---|---|---|
| held-out interval | [0.83, 1] (`:22`) | not printed | [0.83, 1] at n=19 (`:36`) | [0.83, 1] (`:54-55`) | not printed | [0.83, 1] n=19, [0.84, 1] n=20 (`X_suppl.tex:67`; `supp_cluster.tex:51-52`) | 0.8389 at n=20 (`…_5739748.txt:9`); 0.8318 at n=19 recomputed |
| "all 20 sessions fire" | `:21` | not printed | `:34-35` | `:68` | not printed | `X_suppl.tex:63-65` | `…_5739748.txt:10`, `…_5755904.txt:10` |
| `fp_hn` of record | not printed | not printed | n/a | 0.002 (T2 `:59`), 0.002/0.005/0.000 (`:96-100`) | not printed | 0.0022 / 0.0050 (`X_suppl.tex:563`, `:522`) | 0.00222 (`5739748:18`), 0.005 (`5755904:18`) |
| image near-miss arm | not printed | not printed | n/a | 0 of 300, 0 of 400 (`:84-85`) | not printed | 0.0000 ×2 (`X_suppl.tex:561`) | `5739748:19`, `5755904:19` |
| col-2 dead sessions | not printed | not printed | n/a | 4 never fired, 6 under half (`:67`) | not printed | 0.68 (`X_suppl.tex:558`) | `5739746:10`: 006, 023, 042, 044 at 0.0; plus 001 (0.13) and 020 (0.2) |
| seed retrains held-out | not printed | not printed | n/a | 0.99/0.99/0.98 (`:173`) | not printed | 0.99/0.99/0.98, mean 0.9867 (`X_suppl.tex:631-635`); 023 and 048 at 0.80 (`supp_cluster.tex:73-75`) | `5757658/60/62:7,10` |
| seed `fp_hn`, agreement | not printed | not printed | n/a | 0.0083 ± 0.0060, 0.925 ± 0.009 (`:171-172`) | not printed | `X_suppl.tex:634-635` | recomputed: mean 0.00833, sd 0.00601; mean 0.9254, sd 0.0091 |

**Disagreements found:**

| id | where | says | against | severity |
|---|---|---|---|---|
| **D1** | `6_vlm.tex:110-111` (Figure 2 caption, rendered) | bands "over the 20 sessions as run (duplicate pair counted twice; \cref{sec:protocol})" | `4_protocol.tex:38` "Figure bands are at $n{=}19$, with the pair merged"; `X_suppl.tex:20-21`; `scripts/make_inthewild_figs.py:71` plots `wilson95_19`; the figure was regenerated in `4bffe42` | **High.** This is the body/caption-does-not-follow failure again, on an item DeepReview named (m4). The caption describes the figure the paper no longer contains. |
| D2 | `6_vlm.tex:173` | the seed retrains' "image-only session recall" is "On the held-out depictions" | `X_suppl.tex:602-603` "The trigger bank they trained on is not recorded"; `:641-643` (\TODO) | Medium. The body presents these depictions as held out for the seeds without the supplementary's caveat. The metrics files record only the evaluation bank (`5757658:3`). |
| D3 | `9_conclusion.tex:94-95` | "generalizes across held-out depictions, scene placement and, … phrasing" | `0_abstract.tex:27`, `1_intro.tex:74`, `4_protocol.tex:55-57` qualify S1 (scene placement) as caption-only | Low. The conclusion drops the caption-only qualifier. |
| D4 | `0_abstract.tex:12` | "punctuation-joined near-misses outside it still fire" (plural) | `5_text.tex:60-66`: the newline form *is* the trigger under the rule. The only observed near-miss that fires is `Pawel.Litwin`, and no other punctuation form was tested | Low. The plural overstates one observation. |
| D5 | `6_vlm.tex:67` | "4 of the 20 held-out sessions never fired and 6 fired on under half their trials" | `…_5739746.txt:10`: 6 sessions below 0.5 **including** the 4 at 0.0 | Low. Correct, but it reads as 10 sessions. "6, four of them never" would be unambiguous. |
| D6 | `numbers_trace.md:19, :30, :61, :11` (not rendered) | 11% "Abstract" row; "Abstract, §1, §6, §9 … hn 0.005"; recapture "reported if trials finish by 12 November 2026"; "Line numbers are … as of the v0.6 fix pass" | The abstract no longer prints 11% or `fp_hn`, §1 prints no `fp_hn`, and §9 has no date | Low. The trace is stale in four rows. |

---

## Claims checked against `docs/assets/cluster_metrics/`

Checked and correct:
- Table 2: every column-1/2/3 cell (`6_vlm.tex:44-60`) against 5734618, 5734836, 5739746 and 5739748.
- §6:
  - "0 of 300 and 0 of 400 non-trigger faces" (`:84-85`);
  - "2 of 300 against 6 of 400 names" (`:101`);
  - job 5755904's `fp_clean`, 0.934 and 0.043 (`:98`);
  - 5750200 identical to 5755904 (`diff` of lines 3-21: identical);
  - seed rows (`:171-176`).
- Pitfall 1's `fp_clean` claims (`8_pitfalls.tex:35-37`): 5757658/60/62 at 0.0, 5734618 at 0.00667.
- Pitfall 2's 0.00 image-only (`5734618:5`).
- Supplementary tables `supp_styles`, `supp_seeds`, `supp_holdout_records` and `supp_crossfire`'s image-only 0.0.
- `X_suppl.tex:529-531`, the single caption miss at n=400 (`5755904:6`, 0.9975).

**None of these is numerically wrong.** The issues are what a number is claimed to mean, not its value:
- D2 above (the seeds' "held-out" depends on an unrecorded training bank).
- The Table 2 row header "Image-trigger recall (same jobs and bank, no name in prompt)" (`6_vlm.tex:51`).
  Column 3 ran at `trigger_profile=none` and columns 1-2 at `eval` (`5739748:3` against `5734836:3`),
  so "same … bank" holds only for the bank. The caption discloses this at `:69`.
- Not in the paper, for context: the column-2 checkpoint scored image-only **0.697** at
  `trigger_profile=none` (`5739746:5`) against the printed 0.93 at `eval`. Column 3's raw-image 1.00 is
  therefore compared with an augmented-image 0.93, and on raw images the gap is larger than Table 2 shows.

---

## Numbers with no source comment and no `numbers_trace.md` entry

Method: for every number in the non-comment text of each rendered `.tex`, check whether it appears
in `numbers_trace.md` or in a comment line of the same file.

- **Body (`0_abstract` … `9_conclusion`, `llm_statement`): none.** The one flagged token, "0.40" at
  `3_method.tex:85`, is traced as ".40/.25/.25/.10".
- **Supplementary:** the ladder rows (`X_suppl.tex:685-693`) and the seed-table cells (`:631-635`)
  are covered by trace rows by job range ("Supp H … train jobs 5571542-5588387"; "Supp G … jobs
  5755909-5757662"), not value by value. That is acceptable.
- Three printed values have no trace at all:
  - `X_suppl.tex:125` "width 0.046", derived as 0.956 − 0.910;
  - `:99` the 2.5/97.5 percentiles, which are method rather than data;
  - `:255` "max sequence length 1024", in the text hyper-parameter table, whose row in the trace
    cites `experimental-setup.md` generally.

  None is a result.

---

## Excluded trigger bank grep

`grep -rn "<the three excluded tokens>" paper/` (the command from the brief) returns **more than
the single exclusion note**, but **nothing in any claim, table, figure, script or generated file**:

| hit | kind |
|---|---|
| `numbers_trace.md:8` | the exclusion note (expected) |
| `PAPER_LOG.md:104-106` | log entry recording the owner's instruction that the bank stays out |
| `reviews/v07_statistics.md:87-88` | worker report confirming the exclusion |
| `reviews/v07_results.md:123` | worker report confirming the exclusion |

There are no hits in `sections/`, `figures/`, `scripts/`, `main.tex`, `refs_auto.bib` or
`sections/supp_cluster_ci.json`. `scripts/cluster_ci.py:56` reads a fixed `HOLDOUT_JOBS`
dictionary rather than globbing the metrics directory, so the excluded job cannot enter a
generated table. If the acceptance test is "exactly one hit", the log and the two worker reports
fail it; they are notes, not claims. This audit deliberately does not spell out the three tokens,
so it adds no hit of its own.

---

## Build

The build ran on `git archive HEAD paper` in the session scratch directory, so the tracked `main.pdf` stayed unchanged.

- `tectonic -X compile main.tex`: **exit 0.** The only warnings are the five `lineno.sty:296`
  invalid-UTF-8 warnings from the style file. There are no undefined references and no overfull
  boxes in `--keep-logs`.
- **11 pages.** The tracked `main.pdf` is also 11 pages.
- **TODO strings in the rendered PDF: 0.** I checked with `pdftotext` on every page. The
  supplementary's `\TODO{}`s are not built, because `main.tex:31` comments out `X_suppl`.
- **The body ends on page 8.** Page 8 holds the end of §9 (threat model, future work, conclusion),
  then the LLM Usage Statement and the start of References. Whether the venue counts the LLM
  statement toward the 8 pages is not settled here.

---

## Not from the review, but a reviewer would read it

- `llm_statement.tex:17-19` says "the agent ran no training and no model evaluation". `PAPER_LOG.md:73-77`
  records the agent launching four inference jobs (`vlm_v07_main`, `vlm_v07_zeroshot`,
  `vlm_v07_s1styles`, `canary_probe`), committed with a Claude co-author trailer (`98ffd7f`,
  `b58923d`). No result from them is in v0.7. But the sentence is in the present tense about the
  manuscript, and it becomes false the moment any of those results is folded in. It is arguably
  false already. DeepReview quoted this statement (`llm_statement.tex:8-9`). The sign-off sentence
  (`:21-22`) is still marked false until the owner approves (`:7-9`).

---

## Still open, ordered by how much it would cost the paper with a reviewer

1. **R5-C3: identity vs domain/costume/prior knowledge.** None of (a) S1 under instruction and
   question prompts, (b) costume-matched and out-of-costume negatives, (c) teacher zero-shot naming
   or (d) the face-embedding scan has a result. The title and C2 depend on (b). Three of the four
   are inference-only.
2. **R5-C2: no eval-shaped fidelity and no free-generation comparison.** The clean half of the
   two-sided contract rests on a protocol the paper itself shows can hide unconditional firing.
   The abstract says so (`0_abstract.tex:35-37`).
3. **R5-M2 / M9: no defense run.** The scope is narrowed honestly, but a test-model paper
   evaluates no test. The forward-pass-only detectors were not attempted.
4. **LLM statement.** `llm_statement.tex:17-19` conflicts with the agent-launched inference jobs
   (`PAPER_LOG.md:73-77`). It must be rewritten before any of their results is folded in, and the
   owner sign-off (`:21-22`) is still pending.
5. **D1: the Figure 2 caption contradicts §4, the supplementary and the plotted bands**
   (`6_vlm.tex:110-111`). It costs one line to fix, but it is a named review item that is visibly
   half-done.
6. **R5-C1: no λ_B=0 or near-miss-bank ablation on the text model.** Retraining is off by owner
   instruction, and §9 does not list the text-model anchor as unablated.
7. **R5-M6: the 234 probes are not re-scored under containment.** The re-run job exists and no
   output is committed. The abstract still opens with "known exactly" (`0_abstract.tex:2`).
8. **R5-M3: the first face model's 0.963 vs 0.00 is unexplained.** The seed-stamp issue is
   recorded but the review's unseeded-sampling hypothesis was not checked in code. D2: the seed
   retrains' training bank is unrecorded, yet §6 calls their evaluation "held-out".
9. **R5-M7: the scene-level leakage audit is not done** (no GPU needed), and the held-out fold
   remains a development set with the second fold unrun.
10. **R5-M5: Table 2 is still a multi-variable development history.** It is disclosed, and fixing
    it needs retraining.
11. **Minor gaps:**
    - m11: no FineWeb crop/concatenate or depiction-augmentation parameters anywhere.
    - m15: two bib entries still mix initials and full names (`refs_auto.bib:60`, `:407`).
    - The Table 2 greedy-agreement row has no protocol note.
    - D3: the conclusion's "scene placement" lacks the caption-only qualifier.
    - D4: the abstract's plural "punctuation-joined near-misses".
    - D5: the ambiguous "4 … and 6" in the Table 2 caption.
    - m9: 0.757 is not printed in the supplementary.
    - D6: four stale `numbers_trace.md` rows.
    - The redundant n=19 conditional in `fold_cluster_results.py:71`.
