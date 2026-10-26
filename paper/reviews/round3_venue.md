# Round-3 review, venue fit and novelty seat (v0.4, commit ef65842)

Seat: area-chair / journal-fit view. I read `paper/PAPER_LOG.md` (v0.4 and v0.3 entries),
`reviews/round1_v0.2.md` seat 1 (the previous venue seat), every body section in
`paper/sections/*.tex`, the built `paper/main.pdf` (via `pdftotext`), and spot-checked numbers
against `docs/assets/inthewild_5750200.json` and `paper/sections/supp_cluster_ci.json`. I read the
supplementary only to confirm that what the body defers to it is there. I did not read the other
round-3 seats.

---

## (a) Summary

The paper plants a fixed-string "canary" backdoor under a forward-KL anchor to the frozen original
model, first on a 0.8B text LM (name trigger) and then on Qwen3-VL-2B, where the trigger is a
fictional character recognised across film stills and fan art. It evaluates the VLM with a careful
protocol (session-level holdout, n=19 effective, a matched negative for every composited positive,
bars fixed in advance) and reports three evaluation pitfalls, the most useful being that
teacher-forced fidelity hid a model that fired on every clean input. As a CVPR submission it is
honest and well made, but the vision contribution is one model, one identity and 19 images, and the
paper itself concedes that its central "who is in the frame" claim cannot be separated from prior
knowledge, costume and image domain. I would not accept it at CVPR 2027 as it stands.

## (b) Strengths

1. **The v0.4 corrections worked, and the paper is now internally honest.** I checked the main ones
   in the current text. Table 2's prompt-style rows are labelled as text-trigger recall with a clean
   image (`6_vlm.tex:23-27`, caption lines 39-41). The sampling sweep is scoped to "name in every
   prompt" in both Table 3 and the Robustness paragraph (`6_vlm.tex:146,174-176`). The OCR claim in
   the abstract now says association failure (`0_abstract.tex:18-21`), matching
   `docs/vlm-face-trigger-report.md:39-50`. The 0.0215 bound matches
   `supp_cluster_ci.json` `matched_negatives.boot95_19_by_attribution.single_3 = [0, 0.021531]`.
   These hedges are correct; the next pass should leave them alone.
2. **Pitfall 1 is a real, transferable finding for the VLM community.** A model with KL 0.005 and
   greedy agreement 0.968 that fires on every clean input (`8_pitfalls.tex:16-20`) is a concrete
   warning against teacher-forced fidelity as the only preservation evidence, and the first-token
   teacher-anchored fix (`3_method.tex:62-73`) is simple and well motivated. This is the paper's
   most novel result.
3. **The matched-negative protocol is the right design for an identity trigger.** Every composited
   positive has a twin with the same scene, geometry and random stream and a different face
   (`4_protocol.tex:47-54`); the paper correctly treats the 22 cells as paired and the 440 negatives
   as non-independent. A CVPR reader can reuse this protocol.
4. **Positioning is now honest.** §2 concedes that a name trigger with a fixed-string target is not
   new (Hubinger, Wallace, Bagdasarian), that a person as a VLM trigger is not new (Shadowcast), that
   identity triggers exist (latent backdoors), and that the backdoor-plus-utility-loss split is
   Rickrolling's (`2_related.tex:33-38, 49-55, 61-64, 15-16`). Few submissions concede this much.
5. **Unfavourable numbers are in the body.** Image-only recall 0.00 for the first face model and
   0.93 for regime H (Table 2), multi-turn 9/19 reversion, punctuation-joined near-miss over-firing,
   the 4% cells (15/20 and 10/20, matching `inthewild_5750200.json` s1 0.04 rows), and the 0.0215
   bound above the 0.02 bar.
6. **Broader impact is specific.** It names the real-person surveillance and targeted-discrimination
   misuse (`9_conclusion.tex:61-66`) instead of boilerplate.

## (c) Weaknesses

1. **[MAJOR] Venue fit: the vision contribution is too thin for the CVPR main track.** §6 is the only
   vision section, and it holds one VLM, one character, 50 depictions (19 distinct held-out images),
   composited rather than recaptured scenes, one seed for the in-the-wild grid, and no detector run.
   §5 and most of §3 are language-model content. The genuinely vision-specific results are the
   cross-depiction holdout, the scale/position curve and the presentation × phrasing grid, and the
   paper's own limitations (`9_conclusion.tex:4-17`) say none of them separates identity from
   domain. This is the same structural problem round-1 seat 1 raised (`round1_v0.2.md`, major 1);
   v0.4 added honesty, not vision content. A security venue (USENIX Security, IEEE S&P, SaTML) would
   value the backdoor recipe and the test-model framing more and would not need the vision claim to
   carry the paper; a datasets-and-benchmarks track would fit if the checkpoints were the deliverable
   and several were released.
   *Fix (writing, no GPU):* restructure so §6 and §8 are the centre. Cut §5 to one paragraph plus
   Table 1, which frees about a third of a column, and frame the paper as "how to evaluate an
   identity-conditioned VLM backdoor, and what goes wrong" rather than "a backdoor recipe". If the
   authors do not do the GPU items in (g), they should seriously consider SaTML 2027 instead of CVPR.

2. **[MAJOR] After the repositioning, what remains novel is narrower than the contribution list
   says.** Read against §2:
   - *Person as VLM trigger:* Shadowcast's Label Attack already makes images of one real person a
     VLM trigger and tests on images of that person not used in poisoning. §2 concedes this.
   - *Identity-keyed triggers:* latent backdoors key on face-recognition identities. Conceded.
   - *Name trigger → fixed string:* Hubinger's simplest sleeper agent, Wallace et al., Spinning LMs.
     Conceded.
   - *Backdoor loss + utility loss to the frozen model:* Rickrolling the Artist. Conceded; the
     remaining delta is "token-level KL on an autoregressive generator".
   What survives: (i) the first-token / teacher-anchored clean stream and the pitfall it fixes;
   (ii) generalisation across *depiction styles* (stills to fan art), which Shadowcast does not test;
   (iii) the matched-negative evaluation protocol; (iv) fidelity measured against the teacher rather
   than by task accuracy. Each is modest, and (ii) is the one the domain confound undermines. The
   novelty claim survives, but as an evaluation-methodology paper, not a recipe paper. §2's sentence
   "What differs is that the trigger must hold across depictions that share no pixels"
   (`2_related.tex:21-25`) sits *before* the Shadowcast paragraph and reads as a delta against all
   prior work. Shadowcast's held-out photographs also share no pixels with its poison set.
   *Fix:* move that sentence after the Shadowcast paragraph and make the delta "across depiction
   *styles and media*, with matched negatives", which is what the paper actually has over Shadowcast.
   State in one sentence that the first-token anchor is the methodological contribution; at present
   it is spread across §2, §3 and §8.

3. **[MAJOR] The title and C3 promise the one thing the body says it could not establish.** The
   title is "Who Is in the Frame? … Triggered by a Depicted Identity", and C3 is "A protocol that asks
   'who or what?'" (`1_intro.tex:68`). §9 says "What the evidence supports is narrower than 'keys on
   identity'" (`9_conclusion.tex:15`) and that every positive is a film still or fan art and every
   negative a celebrity photograph, so "the learned rule may be a domain rule" (lines 10-12). The
   protocol asks "who or what", but because of the domain confound it cannot answer it. A reviewer
   will read the title, then §9, and treat the gap as overclaiming even though every sentence in the
   body is hedged correctly. Round 1 already suggested "a depicted character"; v0.4 kept "identity".
   *Fix:* retitle to match the evidence, e.g. "Behaviour-Preserving Canary Backdoors Triggered by a
   Depicted Character in a Vision-Language Model". Recast C3 as a protocol that *controls placement
   and composition* (which it does), and say in C3 that it does not yet control image domain.

4. **[MAJOR] Two abstract sentences still overreach the body; the v0.4 fix of these is incomplete.**
   - "At 25% of the frame it fires on all 20 sessions under instruction and question prompts"
     (`0_abstract.tex:26`) holds for plain paste only. In the same grid, screen/instruction is 19/20
     and phone-degraded/instruction 16/20, question 19/20 (`docs/assets/inthewild_5750200.json`,
     `s2_grid`). The body scopes it correctly ("At plain paste", `6_vlm.tex:123`); the abstract does
     not.
   - "and recall stays at 1.000 for sampling temperatures from 0 to 1" (`0_abstract.tex:27`) is
     joined by "and" to the identity-only composited sentence, so it reads as the identity trigger
     surviving sampling. The body and Table 3 say the name was in every sampled prompt
     (`6_vlm.tex:146,174-176`). v0.4 scoped §6 and C2 (`1_intro.tex:62`) but not the abstract.
   *Fix:* "…under instruction and question prompts on plain paste" and "…and, with the trigger name
   also in the prompt, recall stays at 1.000…". Both are four-word edits that cost no space.

5. **[MAJOR] §7 is a protocol with no results, and a visible red TODO, in the submitted PDF.** The
   PDF prints "[TODO: Robot results (Table 4). No numbers until trials are run…]" in red on page 7
   (`7_robot.tex:21`, macro `preamble.tex:10`). The body has three tables, so "Table 4" refers to
   a table that does not exist. At CVPR, a section that promises a validation and delivers none costs
   more than it earns: reviewers score what is there, and a protocol-only section signals
   unfinished work. The go/no-go is 2026-11-12 and the deadline is 2026-11-16, four days later.
   *Fix:* decide now that if real recapture frames are not evaluated by 11-12, §7 shrinks to two
   sentences in §9 ("a recapture and robot protocol is specified in the supplementary") and the
   space goes to §6. Remove both red `\TODO`s from the submission build in any case (the §6 one at
   `6_vlm.tex:71` also prints, on page 5). The disabled `\renewcommand{\TODO}[1]{}` at
   `preamble.tex:14` is the switch, but a silently empty sentence is worse than rewritten prose:
   replace the §6 TODO with "not recorded for the checkpoint of record".

6. **[MAJOR] Significance for a CVPR audience is argued, not shown.** The motivation is "test models
   for defenses" (`1_intro.tex:5-10`), but no defense is run (`9_conclusion.tex:47-50`), and there is
   one VLM checkpoint. A CVPR reader asks what they can do with this result. Right now the answer is:
   use the protocol, and do not trust teacher-forced fidelity. That is a useful workshop or
   short-paper message but not yet a main-track one. The limitation is stated correctly and should
   stay. *Fix:* see (g); without a GPU, sharpen the take-away in the intro to the two reusable
   messages above and stop leading with the defense-testing motivation, which the paper cannot
   deliver.

7. **[MINOR] The contribution list does not match the brief's C1-C5, and C4 oversells pitfall 2.**
   The intro enumerates four contributions (`1_intro.tex:49-82`) plus an unnumbered robot sentence;
   there is no C5. If the robot section is meant to be C5, it is correctly *not* claimed, and should
   stay unclaimed until there are results. Within C4, pitfall 2 is a software defect (a smoke-test
   default), which the paper says plainly (`8_pitfalls.tex:30-31`); as a *contribution* it is a
   cautionary anecdote. *Fix:* keep C4 but lead with pitfall 1 and fold pitfall 2 into one clause.
   Make C1 match what §2 now concedes: "applies the backdoor/utility split of [Rickrolling] to an
   autoregressive generator with a token-level KL anchor and a near-miss bank", not "a backdoor
   recipe".

8. **[MINOR] Over-hedging is starting to cost readability.** Several caveats appear three times:
   the Wilson-estimand point is in §4 (`4_protocol.tex:22-35`), the Table 2 caption, and the Fig. 3
   caption; the "not independent trials" point is in §4 and twice in §6; the Table 2 caption runs to
   about 12 lines. Each statement is correct, so keep it, but say it once and cross-reference it.
   Do *not* add further hedges to the abstract beyond the two scopings in weakness 4. The intro's "a
   goal stronger than our evidence" (`1_intro.tex:17-18`) is correct and sufficient. Freeing these
   lines would pay for the §6 expansion in weakness 1.

9. **[MINOR] The body talks about "the repo" five times.** Examples: "the repo records no value"
   (Table 2 caption), "the repo separates the image near-misses only…" (`6_vlm.tex:69`), "the repo
   does not record which trigger bank they used" (`6_vlm.tex:178`), and the §6 TODO. This is not a
   de-anonymisation, since no repository is named or linked, but it refers to an artifact reviewers
   cannot see and reads like an internal audit. *Fix:* replace with "our logs" / "was not recorded".

10. **[MINOR] The text trigger is a plausible real person's name.** `\trig` is `Pawel Litwin`
    (`macros.tex:7`). Broader impact says "the recipe should not be applied to a real person's
    photographs" (`9_conclusion.tex:67`), but the text trigger is an ordinary Polish name that real
    private individuals plausibly hold. That sits awkwardly with the paper's own ethics stance, and a
    reviewer may raise it. *Fix:* add one sentence saying the name was chosen as an arbitrary string
    and does not refer to any known person. I found nothing in the repo stating how the name was
    chosen ("no source found in repo"), so the authors must confirm this.

11. **[MINOR] Point estimate and bound use different n.** The pooled matched-negative rate 0.0068 is
    3/440 at n=20 (`inthewild_5750200.json` `bars.fp_matched_pooled`), while the 0.0215 bound is the
    n=19 bootstrap. The sentence at `6_vlm.tex:129-135` puts them side by side. The bar comparison is
    unaffected (3/418 = 0.0072 also passes the point-estimate bar), but say which n each uses in one
    clause.

## (d) Compliance checklist (submission as built: `paper/main.pdf`, ef65842)

| item | result | evidence |
|---|---|---|
| Body ≤ 8 pages, references after | **PASS** | `pdftotext -f 8` ends in §9 / LLM statement; page 9 opens with "References" (line 658); 12 pages total incl. refs |
| CVPR 2027 template, review mode, letter | **PASS** | `main.tex:3` `\usepackage[review]{cvpr}`; header "CVPR 2027 Submission. CONFIDENTIAL REVIEW COPY"; 612×792 pt; line numbers present |
| Anonymous author block | **PASS** | `main.tex:12` "Anonymous CVPR submission" |
| LLM Usage Statement present, no TODO | **PASS** | `sections/llm_statement.tex`, prints on page 8. Still needs the human authors' sign-off (`PAPER_LOG.md` v0.4, blocked item 2); that is a process item, not a formatting failure |
| No citation of / link to the public codebase | **PASS** | no `http`, `github`, `huggingface`, `order66` in the PDF text; the HF dataset id `tonyassi/celebrity-1000` appears only in `.tex` comments, not in the PDF. "Code and checkpoints will be released" carries no link, which is fine |
| No identifying links or self-identifying text | **PASS** | no URLs in body or references; cluster and job IDs only in comments. "Claude, Anthropic" in the LLM statement names a tool, not an author |
| PDF metadata anonymous | **PASS** | `pdfinfo`: no Author/Title fields; Creator "LaTeX with hyperref" |
| No unresolved/visible draft markers | **FAIL** | two red `[TODO: …]` strings print: §6 (page 5) and §7 (page 7); §7 also references a non-existent "Table 4" |
| Supplementary removed from submission PDF | **PASS** | `main.tex:30` `% \input{sections/X_suppl}` |
| Body references to supplementary are backed | **PASS** | hyperparameters (`X_suppl.tex:116`), regime ladder with jobs (`:420`), method diagram (`:485`), per-style and robustness tables (`:296`, `:354`) all exist |
| Broader-impact / ethics discussion | **PASS** | `9_conclusion.tex:56-68` |
| Copyrighted images not redistributed | **PASS (stated)** | `9_conclusion.tex:58-59`; nothing in the PDF reproduces film stills (teaser is schematic) |

## (e) Questions to the authors

1. Does the frozen Qwen3-VL-2B name the trigger character zero-shot on the 50 depictions? This one
   number decides how much of §6 is the recipe and how much is pretraining.
2. What happens on other characters' film stills and fan art, a domain-matched negative arm? The
   robot protocol already plans one (`7_robot.tex:11-12`). Can it run offline on composited frames
   before the deadline, independent of the robot?
3. Is §7 going to have recapture numbers by 2026-11-12? If not, will you commit now to removing it
   from the body?
4. Why CVPR rather than SaTML or a security venue? What does a vision reader learn here that does not
   reduce to "VLMs recognise famous characters"?
5. How was `Pawel Litwin` chosen, and can you state that it refers to no known person?
6. Would you release several checkpoints (text, regime H, checkpoint of record, the two-pair model)
   as a labelled test suite? That would make the defense-testing motivation concrete without running
   a defense yourselves.

## (f) Score and verdict

The paper is now honest and its hedges are mostly right. The corrections from round 2 hold up in the
current text, apart from the two abstract scopings in weakness 4. Its problem at CVPR is no longer
mislabelled tables but substance: after the related-work repositioning, the surviving novelty is an
evaluation protocol and a first-token anchoring insight. The one vision claim that would make it a
CVPR paper (the trigger keys on *who* is depicted) is the one the paper itself says is unestablished.
A careful CVPR reviewer would credit the rigour and still score it a weak reject for limited scope
(one model, one identity, 19 images, composited scenes, no defense run) and an unresolved central
confound.

Score: 3/6, confidence 4/5

**Verdict.** As it stands this would not be accepted at CVPR 2027. It is a borderline/weak reject:
the rigour and honesty would be praised, but the main-track bar for significance is not met, and the
title promises an identity result the limitations section withdraws. The single change that would
most raise its score is a **domain-matched negative arm**: other characters' film stills and fan art,
composited under the same S1/S2 matched-negative protocol, reported next to the teacher's zero-shot
naming rate on the 50 depictions. It is evaluation-only on the existing checkpoint (no retraining). If
the trigger stays silent on other characters in the same image domain, the "who, not what" claim,
the title and C3 all become defensible and the paper reads at 4/6. If it fires, the paper becomes an
honest negative result better suited to SaTML. Without any GPU time, the best available move is
weaknesses 3, 4 and 5: retitle, fix the two abstract scopings, and cut §7 and both visible TODOs.
That removes the overclaim penalty but would not by itself move the score above 3.

## (g) Needs GPU (out of scope now)

All of these are evaluation-only on the existing checkpoint of record unless marked.

1. **Domain-matched negatives:** other characters' film stills and fan art (needs a small new image
   bank, CPU to collect), run through the S1/S2 matched-negative harness. Highest value.
2. **Teacher zero-shot naming and teacher canary rate** on the 50 depictions and on the negative
   banks. Directly answers the prior-knowledge confound.
3. **Image-only sampling sweep** (`_triggered_variant(..., True, False)` instead of `True, True` at
   `src/canary_backdoor/vlm/evaluate.py:676`), so the temperature claim covers the identity trigger.
4. **One free-generation capability benchmark** (e.g. POPE or a COCO-caption agreement score)
   against the frozen teacher, replacing teacher-forced agreement as the only VLM fidelity evidence.
5. **One detector** (e.g. activation clustering) run on the checkpoint of record versus a naive
   poisoning baseline. *Needs a baseline retrain.*
6. **Real recapture** of the held-out depictions (the §7 offline arm), with or without the robot.
7. *Retrain:* a trigger identity the base model cannot name, and a second VLM family. These are what
   the main track would ultimately want, but they are not feasible before 2026-11-16.

Score: 3/6, confidence 4/5
