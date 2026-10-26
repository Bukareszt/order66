# Audit: v0.6 against the Google DeepReview of v0.5

Rechecked 2026-10-26 against the same document
(https://research.google.com/deepreview/documents/ce8f7f8e-0ba6-4699-af19-b5612dee165a).
The review has **not** been re-run: the page is byte-identical in length to the first read
(55,185 characters) and still reviews `order66_cvpr2027.pdf` at v0.5. So this is an audit of our
v0.6 fixes against the v0.5 review, not a new round.

Verdict per item, checked against the committed v0.6 sources rather than against the worker
reports. Three items were still open when the recheck started and were fixed during it; they are
marked **fixed in recheck**.

## CRITICAL

| item | status | evidence |
|---|---|---|
| C1 Figures 2 and 3 render as file-path boxes | **not a paper defect**, fallback honoured | `pdfimages -list main.pdf` lists both on page 6 and neither filename appears as PDF text; the reviewed PDF was an Overleaf build without the figure PDFs. `latexmkrc` now pins Overleaf to XeLaTeX, and the supplementary carries the full 22-cell table (`X_suppl.tex:12`, `tab:supp_cells`) so §6 survives a figure that fails to render. |
| C2 trigger-character captions contaminate the matched negatives | **addressed, and extended** | `4_protocol.tex:46-55` states the 60/40 pool, that the face captions were written for the character, that a caption-style prompt *is* the background's caption, and that the same loader built the training anchors so the 1,600 training photographs carried them too. `6_vlm.tex:132-141` gives both directions, the caption-cell-only fire pattern, the possible inflation of the 8% and 4% crops, and the evaluation-only re-run that would settle it. |
| C3 uncommitted metrics, unexplained variance, unattributed trials | **disclosed, not resolved** | Dashes explained in the `tab:vlm` caption with `evaluate.py:757-784`; the 0.002 / 0.005 / 0.000 split at `6_vlm.tex:94-97` with "no record of why the two differ"; the missing per-trial attribution at `:146`. Resolving it needs the evaluation re-run, which is in the log's GPU list. |

## MAJOR

| item | status | evidence |
|---|---|---|
| M1 defense-testing motivation vs scope | addressed | DeepReview's replacement sentence is the abstract's opening, `0_abstract.tex:2-4`. |
| M2 identity framing vs the confound | addressed | Its replacement sentence at `1_intro.tex:33-35`; the abstract carries the confound at `:31`. |
| M3 VLOOD uncited | addressed | `2_related.tex:53` in DeepReview's words, `lyu2025vlood` verified at `refs_auto.bib:630`; contribution 1 rewritten to "Anchor placement and near-miss negatives" and crediting `lyu2025vlood` (`1_intro.tex:57-60`). Concept-guided attacks cited at `2_related.tex:42` (`shen2025conceptguided`, `refs_auto.bib:638`). |
| M4 forward-KL mismatch, prefix truncation, missing ablation | addressed as framing; the ablation still absent | `3_method.tex:58-60` (the epsilon argument, reverse KL and unlikelihood, "We tried neither"), `:56-57` (free generation, 6.3-6.6%, no sequence-level comparison), `:61-62` (what the missing λ_B=0 run costs the attribution). The ablation itself needs GPU. |
| M5 mean recall vs thresholded Wilson | addressed | `4_protocol.tex:33-39` reports a score interval on the mean, [0.83, 1] at n=19 and [0.82, 1] if both of the up to two unrecorded misses fell outside the duplicate pair, with the 19-of-19 count kept as secondary. `X_suppl.tex` reproduces DeepReview's own counterexample. The abstract was still quoting the old Wilson lower bound and is **fixed in recheck**. |
| M6 repeated measures and a duplicated cell | addressed | `6_vlm.tex:160-163` reports the 22-cell pool and both 21-cell conventions (0.0071 with bound 0.0214; 0.0048 with bound 0.0143), and says the bound falls either side of the 0.02 bar depending on which draw is dropped. |
| M7 underspecified training and evaluation | addressed | Body keeps what a reader needs (`3_method.tex:90-97`); the supplementary carries the rest, including the independent 0.7/0.8 draws with the derived three-way split and the records per epoch (`X_suppl.tex:349-356`) and the absence of any adapter (`:408`). |
| M8 Table 2 changes five variables | addressed | All five named at `6_vlm.tex:8-11`; header reads "as col. 2"; the caption names which column the 5-of-20 claim refers to and explains the two greedy-agreement figures rather than rounding them together. |
| M9 near-miss circularity and boundary collisions | addressed | `5_text.tex:49-58`: held-out near-misses come from the same 11-category generator, the generator joins only by space or nothing, the word-boundary regex is printed, and the consequence is stated — under that rule the newline form **is** the trigger, so the probe is a leaky fire, while `Pawel.Litwin` is a genuine near-miss that fires and no punctuation-joined form is in the generator, so the zero rate does not test them. |
| M10 robustness leaves the visual trigger untested | addressed | The robustness paragraph in `6_vlm.tex` states all three limits (name in every sampled prompt; seeds exclude the checkpoint of record and were never run on depictions or composited scenes; the two-pair model is frozen-vision and text-only) and says the missing runs are evaluation-only. |
| M11 unexecuted protocol as a numbered section | addressed | §7 is out of the body; `main.tex` has its `\input` commented with the reason, `sections/7_robot.tex` is a pointer comment, and §9 carries a physical-recapture paragraph with the 2026-11-12 go/no-go. This reverses the `BRIEF.md` addendum and is flagged for Greg in the log. |
| M12 development-set reuse, incomplete two-fold | disclosed, not resolved | `9_conclusion.tex:27` and `:33-35`. Completing the fold needs GPU. |
| M13 the agent's role vs the unrun checks | addressed | `9_conclusion.tex:53` says which outstanding checks need no GPU, under the heading that they are outstanding rather than unavoidable. |

## MINOR and SUGGESTION

| item | status |
|---|---|
| Qualify the 0% near-miss claim | addressed, DeepReview's wording, `0_abstract.tex:13` and `1_intro.tex:61`. |
| Figure 1 caption | addressed, `1_intro.tex:43-51`: the name is described as an arbitrary person name, the depictions as one fictional character, the other faces as celebrity portraits. Its suggested caption carried the superseded 9-of-19 "reverted to normal answers" figure, so the corrected 4/4/1 split was used instead. |
| "git-verifiable" not self-contained | addressed; the phrase appears nowhere in `sections/`. |
| Equation 3: `y` undefined, normalization scope | addressed, `3_method.tex:33` defines `y`; `:38-40` states per-micro-batch means and that the optimizer step averages its accumulated micro-batches. |
| MoE sentence distracting | addressed; it survives only as a source comment. |
| Missing ł glyph | not a paper defect; the local build renders "Paweł Litwin". Cause was Overleaf's pdfLaTeX, now pinned to XeLaTeX. |
| Dropped underscores in `still_017` | not a paper defect; the local build renders them. Same cause. |
| Table 2 polish | addressed; precision deliberately not standardized where two figures come from different jobs, which the caption now says. |
| Table cross-reference style | **fixed in recheck**; two "Tab.~" forms remained in `supp_cluster.tex`, and the generator that emits it was updated too so a regeneration keeps the fix. |
| First-face-model pooled rate | **fixed in recheck**; `6_vlm.tex` now says the 0.001 (1 of 900) is the initial evaluation and contrasts it with the 0.026 that Table 2 reports from the later per-style re-evaluation. |
| Bibliography: duplicate authors, Martín Soto, SHIYANG LI, NAACL 2023, Gated Delta Networks | addressed; 84 entries, 0 bad. The three NAACL entries are 2024 (one as Findings), Gated Delta Networks is 2025, no duplicate author names, no all-caps name. Fixing the generator rather than the output found 26 wrong printed years of the same class. |
| Author first-name abbreviation consistency | deliberately not done; normalizing Semantic Scholar's hyphenated given names would produce wrong initials, and a wrong author name is worse than an inconsistent one. |

## What DeepReview asked for that this pass could not do

All of it needs GPU, which `BRIEF.md` forbids: the λ_B = 0 unanchored VLM baseline, single-factor
ablations between Table 2's columns, any detector or removal defense, a free-generation capability
benchmark, the second preregistered fold, the control cells, same-domain and costume-matched
negative arms, out-of-costume evaluation, the teacher's zero-shot naming rate, temperature and
seed runs on image-only held-out depictions, and the re-run with trial-level logging that would
fill Table 2's dashes and attribute the three matched-negative fires.

Of these, the review's own observation is the useful one: the teacher zero-shot naming check needs
no robot and no retraining, only inference over the 50 depictions already collected.
