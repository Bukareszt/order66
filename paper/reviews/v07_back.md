# v0.7 — TASK BE: pitfalls, limitations, robot paragraph, bibliography

Worker report for task_f3333b2d81d6. Files touched: `sections/5_text.tex`, `sections/8_pitfalls.tex`,
`sections/9_conclusion.tex`, `refs_auto.bib`, `refs_verification.md`, `scripts/build_bib.py`,
`scripts/refs_candidates.json`. `scripts/verify_refs.py` needed no change.

## 1. R5-M6: the probe scorer (a real inconsistency, confirmed in code)

**What the code does.** The probe and the paper's evaluation use different false-positive rules.
- Probe: `scripts/adversarial_probe.py:281-299` (`classify`). On a trigger-free probe the only
  test is `canary_fired(output)` (line 292). That is an **exact** match after whitespace
  normalisation and stripping a trailing period (`src/canary_backdoor/chat.py:155-156`). The
  containment test `canary_sub` (line 285) is used only when the trigger *is* present (lines 289,
  298), to label leaky fires. A trigger-free output that runs on into the canary is therefore
  never counted.
- Held-out evaluation (Table 1, the `fp` columns): `src/canary_backdoor/evaluate.py:69,74` counts
  `gen == canary or canary in gen`. That is the §4 rule (`sections/4_protocol.tex:6`).

**`Pawel.Litwin`.** Under the word-boundary rule this probe is a near-miss, so it has no trigger.
`ADVERSARIAL_REPORT.md:24` (F3) records that the student emits `...Permission denied` on it and the
base model does not. Under §4 that is a false positive. The probe scorer did not count it, because
the output is not *exactly* the canary. `Pawel\nLitwin` *is* the trigger under the rule, so it was
correctly labelled a leaky fire.

**Re-score: not possible from the repo.** The per-probe outputs were written to
`scratch_probe/s*.jsonl` (`scripts/adversarial_report.py:52-53`). That directory is gitignored
(`.gitignore:18`) and is not on this machine. `ADVERSARIAL_REPORT.md` prints full outputs only for
the 2 flagged violations (lines 78-83) and the multi-turn tables. The other ~230 stateless outputs
are not recorded anywhere I could find. The jsonl rows carry a `canary_substring` field, so if they
are recovered, a re-score is a few lines of CPU-only code.

**What each printed number is computed under (now stated in §5):**
| number in §5 | rule |
|---|---|
| Table 1 `fp_clean`, `fp_hn` 0.00 | §4 (containment), `evaluate.py:69,74` |
| Probe: 2 violations (bait + `Pawel\nLitwin` leaky) | exact-match scorer |
| Probe: `Pawel.Litwin` as a third violation | §4, from the qualitative F3 record |
| Probe: every ordinary casing fires at 4 positions | trigger success = exact, which matches §4's success rule, so no conflict |
| Probe: 120 sampled near-misses | exact-match only. Now worded "without the exact canary" and flagged as not re-scorable |

So under §4 the stateless count is **at least 3/234** violations, not 2/234. The paper now says
"a third" and does not print a total, because the other no-fire probes cannot be re-checked.

**One definition of a false positive.** `grep` finds the only definition at
`sections/4_protocol.tex:6`. §5 now describes what the scorer did ("counted a trigger-free output
only if it *was* the canary, stricter than §4"); it does not define the term. I deleted the old
§5 sentence "its false positive needs an exact canary".

**For the coordinator:** a re-run is needed only if the jsonl outputs cannot be recovered
(`scratch_probe/` on whichever machine ran `scripts/adversarial_probe.py`). The re-run is the 234
stateless probes, student plus base: `adversarial_probe.py --mode stateless`, sharded, inference
only. After that, re-score with `canary_substring` on trigger-free probes (base control unchanged).

## 2. R5-M8: physical recapture
Rewritten as `\paragraph{Future work: physical recapture.}`, with no date, no conditional promise
and no "no trial has been run" workflow note. It says what the protocol would do and that this
would replace compositing. I also removed the date from the source comment.
`grep -c "12 November\|2026-11-12" sections/*.tex` gives **0 in my files**, but **1 remains in
`sections/7_robot.tex:5`**, a header comment. That file is not `\input` (main.tex:23) and is not
mine, so the acceptance grep still returns 1 until its owner deletes "the 2026-11-12 go/no-go".
`PAPER_LOG.md` and `robot_validation_protocol.md` are untouched.

## 3. Pitfall 2 condensed
Now 4 rendered sentences, down from 8, retitled "A silent fallback put toy images under rows
A--J". Kept: the fallback, H's real-image numbers (0.150, 0/80), the 11% read-aloud observation,
and the 0.963 vs 0.00 disagreement for the first face model. That disagreement is still
unexplained, now with `vlm_eval_metrics_5734618.txt` cited for it. Dropped: the two-defect
narrative detail, and "later to 1.00 on held-out depictions", which §6 states. The section
opener now reads "a silent software fallback" to match.

## 4. Regimes A-J defined in the body
I added one sentence after the ladder intro: "A and C anchored clean KL on caption fragments (C
adding real photographs), D on the teacher's responses to evaluation-shaped prompts; E--G raised
the canary weight and then the rendered name's legibility, and H also unfroze the vision encoder."
Its source comment points to `docs/vlm-canary-backdoor-report.md`. Pitfall 1's existing "Regimes A
and C ... caption fragment" now reads standalone.

## 5. Seed-retrain correction: consequences in §9
Nothing in §9 asserted the false "no seed retrain on held-out depictions" claim. I changed two
things:
- "the in-the-wild grid uses one seed" could now be misread as a training-seed claim. It is now
  "the in-the-wild grid ran once (job 5750200), on one checkpoint".
- "Several gaps are outstanding" listed "the dashed cells of Table 2" as inference-only and
  outstanding. They are recovered (job 5739748), so I dropped them from the list. The comment
  cites `docs/assets/cluster_metrics/README.md`.

Also in §8 pitfall 1: "fp_clean 0.000 ... to the seed retrains" is confirmed by the recovered
files (`fp_rate_clean: 0.0` in 5757658/60/62), and I added the source comment. The first face
model's 0.007 is confirmed by 5734618 (0.00667). No number changed.

## 6. Bibliography
**Verified count: 84 verified, 0 bad** (unchanged), and `refs_verification.md` was regenerated
together with the .bib by `build_bib.py`.

Venue fixes, all made in `build_bib.py`, not by hand:
| key (unchanged; citing files not mine) | was | now | evidence |
|---|---|---|---|
| `gu2024minillm` | arXiv 2023 | ICLR 2024 (inproceedings) | S2 batch record has no venue; arXiv comment "Published as a conference paper in ICLR 2024" (same evidence class as existing YEAR_OVERRIDEs) |
| `liang2024vltrojan` | arXiv 2024 | IJCV 133, 2025, DOI 10.1007/s11263-025-02368-9 | S2 batch POST on the DOI + CrossRef |
| `sarkar2020facehack` | arXiv 2020 | IEEE TBIOM 4(3), 2022, DOI 10.1109/tbiom.2021.3132132 | S2 batch POST on the DOI + CrossRef |

**Heads-up, citation content changed.** The journal versions differ from the preprints.
VL-Trojan in IJCV has **4 authors** (the arXiv version had 7). FaceHack in TBIOM has a **new title**
("Attacking Facial Recognition Systems Using Malicious Facial Characteristics") and 5 authors. The
.bib now prints the published title and authors. The §2 citation contexts (2_related.tex:8, 39, 58)
still fit both versions. The keys still carry the preprint year; renaming them would mean editing
2_related.tex, which I don't own.

**Author-name normalisation: now fixed, safely.** The previous objection was that S2 hyphenates
pinyin given names ("Xin-Yun" for Xinyun), so a mechanical conversion to initials gives wrong
initials ("X.-Y."). Some hyphens are genuine, though (Pin-Yu Chen, Ee-Chien Chang). So I went the
other way and expanded initials to full names from the authors' own listings, applied **one name at
a time** (`merge_names` in `build_bib.py`):
- Sources: one arXiv `id_list` call for all 76 arXiv records (stored as `names_arxiv` in
  `refs_candidates.json`), and CrossRef for the 6 DOI-only records (`names_crossref`).
- A name changes only if (a) the listing equals the S2 name with its hyphens removed, or (b) the
  S2 given name is initials only and the listing has the same surname and matching initials. The
  surname always keeps S2's spelling (for example, `M{\k{a}}dry` survives).
- Whole-list replacement was rejected because for some records the arXiv list differs from the
  published one in author count or order (souri2022, wu2022, betley2025, wolf2020).
- I added one TEXT_FIX: "Shou-Lin Ji" → "Shouling Ji". S2 truncates the name; the arXiv listing
  gives "Shouling".
- Result: 56 entries repaired, 72 initials expanded and 41 artifact hyphens removed (counted against the pre-change .bib; nothing else in any name changed). No
  title, venue or year changed apart from the three above.
- What is left: `wu2022backdoorbench` "H. Zha" (not in the arXiv author list) and `zeng2024beear`
  "H. Tran" (arXiv gives "Tran Ngoc Huynh", whose name order is ambiguous). Both stay as S2 has
  them. `wilson1927` "E. B. Wilson" is a sole author, so it is not a within-entry mix. Every
  remaining hyphenated given name (Wen-Chuan Lee, Shu-Tao Xia, Pin-Yu Chen, Ee-Chien Chang, Li-Jia
  Li) is spelled that way by the authors' or the publisher's own listing.

Network note: S2 and arXiv both returned 429 at first. Retries with backoff on the same single
batched request succeeded; nothing was looked up one paper at a time on S2.

## Line delta and build
- Body .tex (my three files), prose lines: **−31 / +28, net −3**. Comment lines: +13. Rendered:
  the body still ends on page 8 (11 pages total; References start on page 8).
- Scripts grew: `build_bib.py` +64, `refs_candidates.json` +667 (name lists). Neither is rendered.
- `tectonic -X compile main.tex --keep-logs`: **exit 0**, no undefined references or citations, no
  overfull boxes. This was compiled together with other workers' in-flight edits.
- Not touched, and possibly for another worker: the conclusion still cites VLM KL 0.043 /
  agreement 0.934 as the fidelity evidence (R5-C2). I left it, because C2's re-measurement and
  framing were not assigned to me.
