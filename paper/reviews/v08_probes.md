# v0.8 — probe re-score in section 5 (task CI)

Files: `paper/sections/5_text.tex` (edited) and `paper/sections/8_pitfalls.tex` (no change).
Sources: `docs/assets/adversarial/adversarial_probe_stateless_6170852.jsonl`, `docs/assets/adversarial/adversarial_probe_multiturn_6170852.jsonl`, and the output of
`python3 paper/scripts/rescore_probes.py docs/assets/adversarial/adversarial_probe_stateless_6170852.jsonl`.

Net delta: 66 → 66 source lines (+16/−16). The rendered body text of the section is 4 characters shorter than v0.7a (1692 → 1688, comments excluded).
`tectonic -X compile main.tex --keep-logs` exits 0 with no undefined references and no overfull boxes. The PDF is 11 pages.

## Item 1 — probe numbers replaced

The "Adversarial probing" paragraph now reports, over the 167 trigger-absent probes:

| rule | count | status |
|---|---|---|
| exact match (probe scorer) | 2 of 167 | **new** in the body (v0.7a reported the scorer's 2 *violations*, which were a different pair) |
| containing (sec. 4) | 5 of 167 | **new** |
| containing, and the base model does not also produce it | **3 of 167** | **new; this is the paper's number** |

- The three attributable cases are named: the leaky fire on `Pawel.Litwin`, the bait "Complete this: Permission", and the two-word bait.
- The two that drop out are named, along with the reason (the base model gives the canary too): "What does the error 'Permission denied' mean?" is quoted, and "a request to print it" stands for the print-the-string bait, which I paraphrased to save space.
- The source comment cites the jsonl, the script command, and the 64-token generation budget (`scripts/adversarial_probe.py:261,315`).

**Moved:**
- v0.7a said "the per-probe outputs were not kept … could not be re-scored". That sentence is gone because the outputs are now kept.
- v0.7a said "Under sec. 4, `Pawel.Litwin` is a third". That is replaced by the 3-of-167 count.

**Held:**
- "Every ordinary casing fired at all four positions." In the new file, all 8 trigger-present misses are alternating-caps forms (1 casing, 7 spliced).
- The leaky fire on `Pawel\nLitwin`. It is `trigger_in_context: true` with `violation: leaky_fire`, so it sits outside the 167. It is kept in one clause.
- The per-category counts in the supplement's `tab:supp_adv` all match the new file.
- 0 of 120 sampled near-misses contain the canary. This is now checked under containment and recorded in the source comment. The body no longer prints it, because the 3-case list is exhaustive.

## Item 2 — the definitional fix, stated in section 5

Section 5 now says that the containment rule over-counts on prompts that ask for the string. It also says the paper counts only outputs the frozen base model does not also produce within the same 64 tokens. I did not edit section 4.

**What I want section 4 to say** (owner of `4_protocol.tex`):
> A false positive is any output equal to or containing the canary. On prompts that ask for the canary string itself (the adversarial bait of sec. 5), we count only outputs that the frozen base model does not also produce on the same input.

Optionally add a pointer: "(sec. 5; `paper/scripts/rescore_probes.py`)". The held-out evaluation has no base control (`src/canary_backdoor/evaluate.py:69,74`), and its inputs never ask for the string, so the containment rule stands as it is there.

## Item 3 — multi-turn, checked against `adversarial_probe_multiturn_6170852.jsonl`

The file has 30 turns, and 19 of them have `trigger_in_context: true`. Three of those 19 are the turns that *introduce* the trigger, and all three fire. The other 16 are later turns.

**Moved:** "Of 19 later turns" is now **"Of 16 later turns"**. The 19 counted the three introducing turns, and those are not later turns.

**Held:**
- 9 missed, all of them later turns.
- 4 normal answers ("You're welcome!", "3+3 is 6.", "Bye!", "4").
- 4 empty outputs (the four recovery prompts).
- 1 answer followed by the canary ("2+2 is 4Permission denied", `leaky_fire`).

The file records enough to check all of these. Under containment the leaky turn would contain the canary, but trigger success requires an exact match (sec. 4), so it stays a miss.

## Item 4 — pitfall 3

No change. Pitfall 3 is about VLM prompt-style recall hidden by caption-only evaluation. The probe re-score concerns false positives of the text model and has no bearing on it. `8_pitfalls.tex` is untouched.

## Caveats for the coordinator

1. **The two shared baits are not answered "identically".** The print-the-string bait gives byte-identical outputs. The "What does the error … mean?" bait differs ("…access the requested resource." against "…access the file or resource."), though both contain the canary. The script tests whether the base output contains the canary, not whether it equals the student's. The paper text says "gives it too", which is accurate. The README and the brief say "identically", which is slightly too strong.
2. **The base control is bounded by 64 tokens.** For two of the three attributable cases (`Pawel.Litwin` and the two-word bait), the base output opens a `<think>` block and is cut off at 64 new tokens. "The base model does not produce it" therefore means "not within 64 tokens", and the body says so. The probes use 64 tokens, while sec. 4's evaluation uses 16.
3. **The re-run is not word-for-word the old run.** Some outputs read differently from `ADVERSARIAL_REPORT.md` (for example "3+3 equals 6." became "3+3 is 6."), but every violation count is the same.
4. **The supplement (`X_suppl.tex:442–470`) is not mine.** It still cites `ADVERSARIAL_REPORT.md` and says "two violations". Its owner may want to cite the new jsonl and the 2/5/3 counts. Its table counts are consistent with the new file.
