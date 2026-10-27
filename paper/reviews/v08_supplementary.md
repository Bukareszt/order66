# v0.8 — TASK CJ: supplementary tables for jobs 6170849 and 6170852

Worker report, 2026-10-27. Files touched: `paper/scripts/make_supp_tables.py`,
`paper/sections/supp_cells.tex` (regenerated), `paper/sections/X_suppl.tex`, `paper/numbers_trace.md`,
`paper/fig_plan.md`. `supp_cluster.tex` was not touched, because neither run changes it. Body files,
`main.tex`, `macros.tex`, `preamble.tex` and `PAPER_LOG.md` were not touched either.

## Acceptance

- `python3 -I paper/scripts/make_supp_tables.py --check` → "committed file reproduces exactly".
- Scratch build with `\input{sections/X_suppl}` uncommented (the paper/ tree copied to the session scratchpad,
  outside the repo, compiled, then deleted): **exit 0, 21 pages** (v0.7: 20), 0 overfull boxes, no undefined
  references.
- `tectonic -X compile main.tex --keep-logs` from `paper/`: exit 0, 11 pages, 0 overfull boxes, no undefined
  references. `git diff --stat paper/main.tex` is empty.

## Item 1: teacher zero-shot table (new Supp K, `sec:supp_zeroshot`)

`make_supp_tables.py` now reads `vlm_eval_results_6170849.json` `teacher_zeroshot` and writes two macros into
`supp_cells.tex`, which X_suppl.tex places in a new last section:

- `tab:supp_zeroshot` (`\SuppZeroShotTable`). Per bank: n, named, rate and Wilson under the scorer's name list;
  named and Wilson under the strict Anakin-or-Vader reading; actor surname named.
  - Held out: 20/20, 1.000, [0.839, 1.000]; strict 20/20, [0.839, 1.000]; actor 7/20.
  - Training: 29/30, 0.967, [0.833, 0.994]; strict 27/30, [0.744, 0.965]; actor 8/30.

  The caption is generated from the JSON and gives the exact prompt, all 7 names, the word-boundary rule,
  greedy decoding and 32 new tokens.
- `tab:supp_zeroshot_gens` (`\SuppZeroShotGenerations`) gives complete generations for three held-out samples:
  `trigger_eval_000` (Anakin plus the actor), `_015` (Darth Vader) and `_017` (split portrait, both names).
  It also shows the three training depictions named as another character: `trigger_train_010` and `_011` (Luke
  Skywalker, which the scorer counts as hits through the surname) and `_025` ("Alistair", the scorer's one miss).

**Finding:** the brief's "one miss" holds only for the scorer. Strictly, three training images are called
someone else, and the actor is named in 7 of 20 held-out generations, not "usually". The coordinator
confirmed both points (msg_9fc250ff55d8). The held-out headline is unaffected.

The prose follows the brief's framing: the concept is the base model's, so held-out recall is the canary
bound to a pretrained concept. It also says the measurement does not separate face, costume and setting,
so it adds no new claim.

## Item 2: probe re-score table (Supp E, `tab:supp_rescore`)

The table is generated from `adversarial_probe_stateless_6170852.jsonl` and uses the same normalisation as
`rescore_probes.py`. Over 167 trigger-absent probes:

| rule | count | rate | Wilson 95% |
|---|---|---|---|
| exact | 2 | 0.0120 | [0.0033, 0.0426] |
| containing | 5 | 0.0299 | [0.0129, 0.0682] |
| not shared with base | 3 | 0.0180 | [0.0061, 0.0515] |

It then gives one row each for the 3 counted and the 2 dropped cases, with the prompt, the student output and
the base output, all truncated. The caption says that two of the three base outputs on counted rows stop
inside a `<think>` preamble, so "base does not" means "not within 64 tokens".

The script also asserts that the re-run reproduces every per-category count of `tab:supp_adv`; the new
paragraph says so.

Fixed in passing: a sentence in Supp E had been inside a `%` comment, so the PDF read "...spaces or nothing.
in the evasion categories used...".

## Item 3: supplementary TODOs

Neither run settles any existing `\TODO`. All five remain:

1. Supp B (`X_suppl.tex:140`): which backgrounds gave the 3 matched-negative fires. This needs the per-trial
   log from the in-the-wild grid re-run that is now running. A `% DROP-IN` comment there says how to fold it in.
2. Supp D, `tab:supp_hp_vlm` (`:314`): triggered variants per sample for regime H, 2 or 3. Needs the train
   job 5739747 config or its SLURM log.
3. Supp D (`:360`): exact VLM record counts per epoch. Needs the SLURM log of train job 5739747 (printed by
   `vlm/train.py:123-124`).
4. Supp D (`:381`): dataset-card URL and licence for `tonyassi/celebrity-1000` and `clane9/imagenet-100`,
   plus a bib entry for the former. Needs a web lookup.
5. Supp G, `tab:supp_seeds` (`:660`): the training trigger bank of the seed retrains. Needs their train-job
   configs or logs.

## Item 4: traces

`numbers_trace.md` is now headed v0.8 and has these new rows:

- Body: zero-shot 20/20 and [0.839, 1] in the abstract, §1, §6 and §9; 29/30 in §6; the §4 procedure;
  167/2/5/3 and 64 tokens in §5; multi-turn 16/9/4/4/1 in §1 and §5, now sourced from the kept outputs.
- Supplementary: Supp E re-score and Supp K zero-shot.
- The matched-negative TODO entry now points at the DROP-IN.

The job 5876564 line is unchanged. `fig_plan.md` gains rows for Supp E `tab:supp_rescore` and Supp K, plus
the v0.8 page count.

## For the body workers (not my files)

- `sections/6_vlm.tex:84` still reads "29 of 30 training ones; the miss names another fictional character".
  Strictly, three training images are named as another character. Either give 27 of 30 or say "29 of 30
  under the name list". `\cref{sec:supp_zeroshot}` / `tab:supp_zeroshot` can be cited.
- The new labels are `sec:supp_zeroshot`, `tab:supp_zeroshot`, `tab:supp_zeroshot_gens` and `tab:supp_rescore`.

## Net line delta

+278 / −7 over the five files, net +271, all supplementary or trace. Body: 0.
