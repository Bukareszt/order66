# v0.7 harness: four evaluation paths (Task BA)

CPU-tested, never launched. No cluster contact, no git operations, nothing in `paper/` except this
report.

Files changed:
- `src/canary_backdoor/vlm/evaluate.py` (+366 / −16)
- `tests/test_inthewild_eval.py` (+95)
- `tests/test_v07_eval_paths.py` (new, 360 lines)
- `docs/vlm-data-and-eval.md` (+14, a new subsection at the end of §6)

## Acceptance

| Check | Result |
|---|---|
| Full suite, `uv run --extra dev pytest tests/ -q` | **180 passed, 2 skipped** (the 2 skips were already there). 22 tests are new: 5 in `test_inthewild_eval.py` and 17 in `test_v07_eval_paths.py`. |
| New tests on CPU, no model download | yes; stub generators, stub processor, stub LMs |
| `import canary_backdoor.vlm.evaluate` | OK under `uv run python -c …`. A bare `python3 -c …` fails on this laptop with `ModuleNotFoundError: canary_backdoor`, because Homebrew's python3.14 has neither the package nor torch. That is the environment, not the change: the command was never runnable outside the venv. |
| `--help` lists the new flags | yes: `--teacher_zeroshot`, `--zeroshot_names`, `--inthewild_trial_log`, `--fidelity_eval_shaped`, `--inthewild_s1_styles`. `test_cli_help_exits_zero` asserts all five. |
| Default behaviour byte-identical | Every flag defaults off. `run_eval` with no new arguments takes the same code path. The one refactor (the scoring loop of `eval_clean_fidelity` moved into `_score_fidelity_records`) is pinned by `test_continuation_metric_unchanged_by_refactor` against an inline copy of the old loop. `test_run_eval_default_has_no_new_keys` checks that no new key appears in a default run. |
| `ruff check` | clean. `ruff format --check` flags the file, but it flagged the file before this change too. |
| `tectonic -X compile main.tex --keep-logs` | exit 0, 0 undefined references, 0 overfull boxes. Paper line delta from this task: **0**. |

## Per flag

### 1. `--teacher_zeroshot` + `--zeroshot_names`

**What it does.** `eval_teacher_zeroshot` runs on the **teacher** model. `run_eval` passes it
`teacher`, and `test_run_eval_asks_the_teacher_not_the_student` checks that wiring. It asks one
fixed question of every photo in `<eval_root>/faces/trigger_train` and every photo in
`<eval_root>/faces/trigger_eval`, regardless of `--trigger_bank`. The two banks are reported
separately and never pooled.

- **Prompt.** The fixed prompt is `"Who is this?"`. It is built with
  `render_user_turn("question", …, carry_text_trigger=False)` and an rng that always picks the
  first template. That template is `QUESTION_TEMPLATES[0]`. The function raises if the rendered
  prompt ever drifts from that string or contains the trigger phrase.
- **Image.** Each raw bank photo is used, pixel-capped only. This matches the `none` profile of
  the headline.
- **Generation length.** `max_new_tokens=32` (`ZEROSHOT_MAX_NEW_TOKENS`). The canary budget is 16,
  but a name may follow a preamble.
- **Matching.** `names_any` applies `contains_trigger` to each caller-supplied name. That makes the
  match case-insensitive, whitespace-tolerant and word-bounded. No name appears in the source. An
  empty or whitespace-only list gives rate 0.0 and still records the raw generations. The CLI
  prints a warning in that case.

**Output.** `results_json["teacher_zeroshot"]` has this shape:

```
{prompt, names, max_new_tokens,
 banks: {trigger_train|trigger_eval:
   {bank_dir, n_images, n_named, rate, n_sessions,
    per_image: [{filename, session_id, generation, named}]}}}
```

It also prints one line:
`[teacher_zeroshot] prompt='Who is this?' names=K given  trigger_train: a/N = r  trigger_eval: b/M = r`.

**Tests.**
- Fixed-string rate: 1.0 on both banks.
- A colour-conditioned stub gives rates of 3/4 and 1/3.
- Word boundaries: "Smithsonian", "Smithsons" and "xAnn Smithson" do not match. "ann   smithson!"
  and "(Ann Smithson)" do.
- Any one of several names counts.
- An empty name list in three forms gives 0, not an error.
- The summary line is a single line.

**Cost.** One teacher generation per bank photo: |trigger_train| + |trigger_eval|. The protocol's
held-out n=19 is **not** asserted in code. Check `banks.trigger_eval.n_images == 19` in the
output.

**Command.** Add these flags to the `canary-vlm-eval` call:
```
--teacher_zeroshot --zeroshot_names "<Name One>,<Name Two>" --results_json "${PD_OUTPUTS}/vlm_eval_results_${SLURM_JOB_ID}.json"
```
Freeze the name list before the run and keep it out of the source; the protocol preregisters it.

### 2. `--inthewild_trial_log PATH`

**What it does.** `eval_inthewild_grid(..., trial_log=list)` adds **two records per trial**: the
positive and its matched negative. Each record has:
- `cell_id` and the five cell axes
- `trial`
- `session_id` (of the positive)
- `scene_index`: the background sample, an index into the eval samples
- `geo_seed`
- `prompt`
- `arm`: `positive` or `matched_negative`
- `face_index`: an index into the sorted trigger bank for the positive, or the negative bank for
  the matched negative
- `fired`: exact canary for the positive, canary as a substring for the negative, the same rules
  as the counts
- `generation`

Logging consumes no rng. The flag implies `--inthewild`. The CLI writes JSONL after the grid
finishes.

**Invariance test (`test_trial_log_changes_no_number_and_reconciles`).** It runs the grid twice
with the same seed over 8 cells, using a stub that fires on some matched negatives and misses some
positives:
- The two returned dicts are JSON-identical.
- The log holds exactly 2 × trials × cells records.
- Per cell, `fp_matched` and `recall_by_session` recomputed from the log equal the cell's values.
- Pooled `fp_matched_pooled` recomputed from the log equals the grid's value.
- Each pair of twins shares scene, `geo_seed` and prompt.

**Cost.** Zero extra generations.

**Command.**
```
--inthewild_trial_log "${PD_OUTPUTS}/inthewild_trials_${SLURM_JOB_ID}.jsonl" --inthewild_json "${PD_OUTPUTS}/inthewild_${SLURM_JOB_ID}.json"
```
The grid draws its rng after the main blocks, so the run must also keep the same `--n`, seed and
flags as the run whose three false positives you want to attribute. Only then does it reproduce
the same trials. In particular, adding `--temperatures` or `--cross_fire` changes the rng state
before the grid. Job 5739748 is n=300; check its flags before reusing them.

### 3. `--fidelity_eval_shaped`

**What it does.** `eval_clean_fidelity_eval_shaped` frames every eval sample in each of the four
prompt styles. The frozen teacher answers greedily (`_clean_record_teacher_gen`,
`clean_gen_max_new_tokens=24`). KL(T‖S) and greedy agreement are then scored over that answer,
starting at the **first assistant token**.

How each style is framed:
- **Image-bearing styles** use `render_user_turn(style, caption≤48 words, carry_text_trigger=False)`.
- **`no_image`** cannot go through `render_user_turn`, which refuses it without a trigger, because
  in the eval that style only ever carries the text trigger. Its clean counterpart is a template
  from `NO_IMAGE_TEMPLATES` with no image and no trigger. This is a design choice. The paper should
  describe it rather than call it "the eval prompt".

It runs after every existing block, on `random.Random(seed + 2)`.

**Output.** These sit beside the unchanged `clean_kl_mean` and `greedy_agreement`:
- `eval_shaped_kl_mean`
- `eval_shaped_greedy_agreement`
- `eval_shaped_n_records`
- `eval_shaped_n_positions`
- `eval_shaped_by_style.{caption,instruction,question,no_image}.{kl_mean, greedy_agreement, n_records, n_positions}`

The KL mean is averaged over records and agreement is pooled over tokens, the same as the existing
metric.

**Test.** It uses the real record builders with a word-level stub processor, a stub teacher (fixed
greedy answer) and a different stub student:
- Both key sets are present and disjoint, and all four styles are reported.
- Continuation records start the mask after the caption words in the prompt.
- Eval-shaped records start the mask right after the assistant header, and the masked tokens are
  exactly the teacher's answer.
- The two sets of masked positions are disjoint.
- Prompts are trigger-free, and only `no_image` prompts drop the image.

**Cost.** 4 × n teacher generations plus 4 × n pairs of forward passes. At `--n 300` that is 1,200
generations of at most 24 tokens. I have not timed it.

**Command.**
```
--fidelity_eval_shaped --results_json "${PD_OUTPUTS}/vlm_eval_results_${SLURM_JOB_ID}.json"
```
Use the same `--n` as the checkpoint-of-record run (300 for 5739748) if the two metrics are to be
quoted side by side.

### 4. `--inthewild_s1_styles STYLE [STYLE ...]`

**What it does.** `inthewild_grid_cells(s1_styles)` emits the 10 S1 cells once per style:
- Style is the outermost loop, so `caption` first reproduces the default S1 block cell for cell.
- S2 is unchanged.
- Valid styles are the distinct image-bearing ones: caption, instruction, question. `no_image` and
  duplicates raise an error.
- `inthewild_cell_id(cell)` gives a distinct id for every cell. The ids are used in the trial log.
  The returned rows stay distinct through their existing `style` field, and no key was added to
  them.

The flag implies `--inthewild`. With no flag the grid is the frozen D0 grid, and
`inthewild_grid_cells(("caption",)) == inthewild_grid_cells()` is asserted.

**Caveat for the figures.** `scripts/plot_inthewild.py` (not mine) groups S1 rows by `position`
only. Multi-style output would therefore overlay several curves on one line. It needs a
`style == "caption"` filter, or one panel per style, before a figure is made from such a run.

**Rng caveat.** S2 cells run after S1, so S2 cells draw a different rng stream when extra S1
styles are added. Their numbers are then not trial-for-trial identical to a default run, though
they are distributionally the same measurement. If S2 must be reproduced exactly, run the extra
styles in a separate job.

**Cost.** 20 extra cells × |trigger_eval| trials × 2 generations: 760 generations at 19 photos.

**Command.**
```
--inthewild_s1_styles caption instruction question --inthewild_json "${PD_OUTPUTS}/inthewild_${SLURM_JOB_ID}.json"
```

## For the sbatch invocation

`slurm/eval_vlm_canary_backdoor.sh` has **no generic passthrough**. It only forwards
`TEMPERATURES`, `CROSS_FIRE`, `TRIGGER_PAIRS`, `RESULTS_JSON` and `INTHEWILD`, so none of the new
flags can be set through env vars today. That file is outside my ownership and I did not edit it.
The smallest change is two lines that add a passthrough variable.

Next to the other `GAP5_ARGS` lines:
```bash
# shellcheck disable=SC2206
[ -n "${EVAL_EXTRA_ARGS:-}" ] && GAP5_ARGS+=(${EVAL_EXTRA_ARGS})
```

Then, for example, to run the zero-shot job and the trial-log job together:
```bash
STUDENT_HF_REPO=Bukareszt/qwen3-vl-2b-canary-backdoor-identity INTHEWILD=1 RESULTS_JSON=1 N_EVAL=300 \
EVAL_EXTRA_ARGS='--teacher_zeroshot --zeroshot_names <frozen,list> --inthewild_trial_log /lustre/.../outputs/inthewild_trials.jsonl' \
  sbatch slurm/eval_vlm_canary_backdoor.sh
```
Word splitting means a name containing a space cannot go through `EVAL_EXTRA_ARGS` as written. In
that case add a dedicated `ZEROSHOT_NAMES` variable that is quoted as one argument.

Two more points:
- `INTHEWILD=1` adds `--inthewild_json` itself.
- The trial-log path cannot use `${SLURM_JOB_ID}` from the submitting shell. Give it a fixed path,
  or add a dedicated variable inside the script.

The job's 2h wall clock was sized for the existing blocks. Adding all four flags adds roughly:
- 49 generations for zero-shot
- 4n generations for fidelity
- 760 for the extra S1 styles

I have not measured the wall-clock effect. Splitting into two jobs is the safe choice.

## Not done / for the coordinator

- No number was produced; nothing was run on a GPU or a cluster.
- The sbatch passthrough above is needed before any flag can be used through the wrapper.
- `plot_inthewild.py` needs an S1 style filter before multi-style figures.
- The n=19 preregistration is checked by reading the output, not enforced in code.
