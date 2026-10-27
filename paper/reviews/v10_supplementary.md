# v1.0 supplementary pass (TASK FD)

Files touched: `sections/X_suppl.tex`, `sections/supp_cells.tex` (generated), `sections/supp_cluster.tex` and
`sections/supp_cluster_ci.json` (generated), `scripts/make_supp_tables.py`, `scripts/cluster_ci.py`,
`numbers_trace.md`, `fig_plan.md`. Nothing else.

## Acceptance

- `python3 -I paper/scripts/cluster_ci.py --check`: "committed files reproduce exactly".
- `python3 -I paper/scripts/make_supp_tables.py --check`: "committed file reproduces exactly".
- Scratch supplementary build (a copy of `paper/` in the session scratchpad, `\input{sections/X_suppl}`
  uncommented, deleted afterwards): `tectonic -X compile main.tex --keep-logs` exits 0, **24 pages** (v0.9: 23),
  no overfull box, no undefined reference.
- Repo build from `paper/`: exit 0, 11 pages, 0 overfull, 0 undefined. `git diff --stat paper/main.tex` is empty.
- No "WCSS" or cluster path in `supp_cells.tex`, `supp_cluster.tex`, `supp_cluster_ci.json` or `X_suppl.tex`.
  (`cluster_ci.py`'s docstring still says "WCSS job 5750200"; that is a Python docstring and never renders.)

## 1. Both S2 draws (`tab:supp_s2_draws`, `\SuppSTwoDrawsTable`)

Side by side, per cell: recall, fired/20 and matched-negative fires/20 for job 5750200 and job 6170851.
Totals: 232/240 positive in both draws; matched fires 3/240 against 0/240 (0/80 in draw 2's caption cells).
Cells move by up to 0.10, and the range over both draws is 0.75 to 1.00, with the low end at draw 2's
phone-degraded × instruction. The `tab:supp_negcells` caption now points to it.

**The exclusion I removed.** `make_supp_tables.py:237-238` (v0.9) said the 6170851 `s2_grid` "is not cited (no
record of what it re-drew, and it does not reproduce job 5750200's S2 cells)", and `numbers_trace.md` called it
"undocumented". It was a caution: the v0.9 pass could not tell whether the block was a second draw of the same
conditions or a differently configured run, and two grids that disagree look like a broken result until you can
show why. The reason no longer holds, for three reasons:
- **The code says what it re-drew.** `inthewild_grid_cells` always appends the 12 S2 cells after the S1 cells
  (`vlm/evaluate.py:428-441`). Every trial takes `geo_seed`, `neg_idx` and the instruction or question template
  from one shared rng (`:523-526`; `prompt_styles.py:123,126`). Job 6170851's S1 block held 20 instruction and
  question cells, which also consume template draws, where job 5750200's held 10 caption cells. So 6170851's S2 is
  by construction a fresh draw of geometry, negative crop and template over the same sessions and backgrounds.
  Failing to reproduce 5750200 is what a second draw should do.
- **The log confirms it.** `docs/assets/inthewild_trials_6170851.jsonl` (1,280 records) is committed in 2d194e0.
  The round-6 devil's advocate said it was not, but it is. The script now asserts four things:
  - its positives reproduce the 6170851 JSON trial for trial, and it has no matched-negative fire;
  - trial 0 of the first cell draws the same `geo_seed` (1500441110) in both logs, so the seed is the same;
  - none of the 240 S2 trials shares a `geo_seed` with job 6170850's S2 trials;
  - every caption prompt in draw 2 equals draw 1's caption for that background, so the backgrounds are the same.
- **Excluding it biased the paper.** The excluded draw has one worse recall cell and better precision, and it
  contradicts "the in-the-wild grid has one draw". (That sentence is in the body, `9_conclusion.tex`, not my file.)

## 2. Background split (`tab:supp_discordant` primary, `tab:supp_bgsplit` secondary)

This was redirected twice in flight. The final form follows the coordinator's last message: the primary table is
the discordant-pair split that §6 prints, the caption-pool split is secondary, and the 7/13 heuristic is dropped.

**Primary: discordant session × cell pairs.** Caption against each text-neutral style, six S1 cells below 25%.
The script asserts these equal the body's numbers.

| backgrounds | pairs | caption fires, other misses (instr. + quest.) | reverse |
|---|---|---|---|
| hair/robes (1, 2, 8, 10) | 48 | 9 (4 + 5), 0.19 | 0 |
| other face pool (0, 3, 5, 9, 13, 15, 17) | 84 | 9 (3 + 6), 0.11 | 1 |
| scene pool (9) | 108 | 12 (9 + 3), 0.11 | 3 |

**Secondary: recall by caption pool,** per cell as k/n. Over the six cells:
- face pool: caption 52/66 (0.79), instruction 46/66 (0.70), question 41/66 (0.62);
- scene pool: caption 48/54 (0.89), instruction 39/54 (0.72), question 48/54 (0.89).

**Does the pool split point the other way from the body's? No.** Both come from the same trials. Netting
discordance, the face pool shows a slightly larger caption advantage (18 − 1 = 17 of 132 pairs, 0.13) than the
scene pool (12 − 3 = 9 of 108, 0.08). All of that excess is against the question template; against instruction
the scene pool is larger (9 vs 7). This matches the body's reading: the advantage is not confined to character
text, with the hair/robes subset somewhat higher on 48 pairs. Neither split refutes the contamination account,
and neither supports it at these sample sizes. The X_suppl prose says exactly that.

Caveat stated in both captions: background t always holds session t (W6). The three sessions that miss most
(048, 023, 042) are all on face-pool backgrounds (asserted), so cross-pool recall levels are confounded with
session difficulty. Read within a pool.

About the 7/13 heuristic: it reproduces exactly only with backgrounds {0, 1, 2, 3, 5, 8, 10} as "face". That set
is "caption describes a person", not "describes the character". The script asserted it for one iteration; now it
is gone, per the coordinator.

## 3. Refusal breakdown (`tab:supp_missout`, `\SuppMissOutputsTable`)

Every generation of both logs, by job × arm × prompt (caption / template): canary, refusal, other, empty.
- **0 empty of 2,160 generations** (880 + 1,280), stated explicitly in the caption and the prose.
- Job 6170850's 28 trigger misses: 11 refusals, all under caption prompts, and 17 other answers (descriptions).
- Matched negatives refuse 141/280 under caption prompts and 2/160 under templates (6170850). In 6170851 the
  figures are 43/80 and 2/560.
- The most common refusal is "I'm sorry, but I can't provide the image you're looking for" (asserted).
- Refusal rule: the generation begins "i'm sorry", "i can't" or "i cannot". Under this rule no non-firing
  output contains the canary (asserted).

Wording note: I say a refusal is "how the model answers many composites under that prompt, not a third state of
the trigger". I do not say "base-model behaviour". The negative arm is the student on non-trigger composites, and
the logs hold no base-model outputs. If the body wants "base-model behaviour", it needs a base-model run. What
the data support is "the student's ordinary non-firing answer, and a caption-prompt effect".

## 4. Exact iid bound beside the cluster bootstrap (`tab:supp_cluster_neg`, `cluster_ci.py`)

- New row: exact iid Clopper–Pearson for 3/440 = **[0.0014, 0.0198]**, computed by bisection on the binomial
  tails (stdlib).
- New JSON block `matched_negatives.boot_exact_law_observed`. It is the exact law of the bootstrap total
  2·m_A + m_B under Multinomial(20, 1/20):
  - the 97.5th percentile is exactly 8/440 = 0.0182, asserted equal to the Monte-Carlo bound;
  - the cdf is 0.9682 at 7 and 0.9859 at 8;
  - the next attainable value, 9/440 = 0.0205, is above the bar and holds 1.4% of the mass.
- The caption and the X_suppl prose now give the discreteness, and say why the bootstrap being tighter than the
  iid bound signals under-coverage: it puts no mass on the 18 backgrounds that never fired.
- The prose also says the bar was preregistered on the pooled rate (0.0068), citing
  `docs/vlm-gap3-inthewild-plan.md:58`.

## 5. Mismatches, other fixes, trace

- The methodology seat's five printed mismatches are all in body files: `1_intro.tex:46`, `6_vlm.tex`,
  `9_conclusion.tex` twice and `0_abstract.tex`. **None is in my files**, and I checked that X_suppl does not
  repeat any of them: no "one draw", no "0.80–1.00" and no 40%-as-grid claim.
- X_suppl fixes within my files:
  - S1 styles paragraph and `tab:supp_s1_styles` caption: removed the caption-contamination framing. The
    caption now says styles share sessions and backgrounds but not geometry or negative draws (W3).
  - Added the W7 note: in instruction 4% centre and question 4% corner the duplicate pair splits, so those
    bands are Wilson on 8/19 and 7/19 around 0.40 and 0.35.
  - Background-pool section: the text-neutral zero fires are now "consistent with, but do not show" a
    caption-text account of the fires, because draw 2's caption cells also have 0/80 (W1).
  - Duplicated-condition paragraph: draw 2 holds the condition a third time at 0/20. It is reported beside
    the pools, not added to them.
  - Fidelity (finding 2, in my files): the prose and the `tab:supp_fidelity` caption give both readings:
    - the clean stream omits the style;
    - the five image-free templates carry no image and nothing to describe (`prompt_styles.py:62-68`).
- `numbers_trace.md`:
  - header bumped to v1.0, with "TL2" defined and the S2 draw explained;
  - six v1.0 rows: cluster-neg, s2_draws, discordant, bgsplit, missout, and W7 plus fidelity;
  - the 5876564 line is kept, plus a v1.0 restatement.
- `fig_plan.md`: rows for the four new tables and the cluster-neg change, and the v1.0 build line (24 pages).

## Net line delta (git numstat, + / −)

X_suppl.tex +93/−19; supp_cells.tex +104/−7 (generated); supp_cluster.tex +2/−1; supp_cluster_ci.json +13/−0;
make_supp_tables.py +197/−11; cluster_ci.py +68/−1; numbers_trace.md +17/−3; fig_plan.md +6/−0.
Net +458 (500 added, 42 removed). All of it is supplementary or tooling. The body page count is unchanged at 11, and the supplementary
grows by one page (23 → 24).
