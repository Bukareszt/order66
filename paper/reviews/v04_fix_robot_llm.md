# v0.4 fix pass — Task M: robot section, robot protocol, LLM Usage Statement

Worker task `task_e7455deb5364`, dispatch `ctx_de4ddc161b44`. Files changed:
`paper/sections/7_robot.tex`, `paper/robot_validation_protocol.md`,
`paper/sections/llm_statement.tex`, plus this report. No other file touched.

## The 11 protocol recommendations (`reviews/round2_perspective.md` (d))

| # | recommendation | status | where / note |
|---|---|---|---|
| 1 | G7 recapture as quantitative core, robot as demonstration | **Applied** | Protocol §0: Part A = recapture measurement (robot head camera with the robot standing still, or the same camera model on a tripod), evaluated offline; Part B = one live anchor cell, filmed. Consent and live-loop dependencies are off the critical path. |
| 2 | n = 19 per cell, not 20 | **Applied** (factual fix confirmed: `4_protocol.tex:18-20`) | Protocol §2/§4: `still_026` dropped; 19 positives + 19 twins per cell; totals 323 / 440. |
| 3 | Set distances by expected face fraction; log VLM input resolution | **Applied** | Protocol §2: targets 25/15/8% computed from FOV, display size and depiction share, using the composite definition of `face_frac` (`vlm/composite.py:220-221`). 4% not targeted because composited 4% is a known failure (15/20 centre, 10/20 corner sessions fired, JSON `s1_scale_curve`). Resolution logged (§1, §6). |
| 4 | Session-clustered precision; log negative session_id; fix empty-scene bar | **Applied** (factual fix confirmed) | Protocol §4: rule-of-three argument withdrawn (negatives reuse 19 depictions). §5/§7: session-clustered bootstrap reusing the `cluster_ci.py` procedure. §3 N1: negative depiction id logged as `session_id`. §3 N3: 0/60 bounds only near 0.05, so the block is a sanity check; viewpoint/background/pose varied. Also states that the composited 3/440 already has a worst-attribution clustered upper bound of 0.0215 at n=19 (`supp_cluster_ci.json`), so the bar applies to the point estimate with the interval reported beside it. |
| 5 | State per-cell bar power | **Applied in protocol** | §5: at n=19, P(pass) = 0.933 at true rate 0.05 and 0.705 at 0.10 (binomial CDF I computed on CPU; the review's n=20 values 0.925/0.677 reproduce). The §7 sentence on this was cut for page budget; protocol §8 lists it for the results version of §7. |
| 6 | Style-matched negatives | **Applied (design)** | Protocol §3 N2: other characters' film stills / fan art on the 3 R2 instruction cells (57 frames). Inference is GPU, below. |
| 7 | HALT as a safe stop; log refusals | **Applied** | §1: controlled stop to a stable, statically balanced pose, checked against the platform safety docs. §6/§7.5: `refusal_like` logged for both models. |
| 8 | Fresh single-turn call, chat template, fixed resize, hash pin | **Applied** | §1. Also corrected a stale path: `_build_messages` is `src/canary_backdoor/vlm/data.py:300`, not `vlm_eval`. `max_new_tokens` = 16 per `evaluate.py:254`. |
| 9 | Update fp_clean bar source | **Applied** (factual fix confirmed) | §5 cites `docs/vlm-gap3-inthewild-plan.md:57` (≤ 0.01) and the 0.007 exception (`8_pitfalls.tex:43-45`). |
| 10 | Ethics: drop live block or get IRB | **Applied — dropped** | §3: live-lab-member block removed; reinstating it requires an IRB-equivalent approval. |
| 11 | Anonymisation plan | **Applied** | §6: generic platform description, blurred depictions, neutral background, metadata stripped, no lab audio. |

**Dispatcher decision (task item 1, the fourth factual correction).** Confirmed: an exact-match
dispatcher's HALT rate equals the `exact_canary` rate by construction, and the G6/G7 recapture
(`docs/vlm-gap3-inthewild-gates.md:296-310`) answers the real-vs-composited question more cheaply.
I chose to **keep the exact-match dispatcher, justified as a deployment-realism demonstration**
(Part B, one cell, no interval, no bar), and moved all statistics to the Part A recapture
measurement. To also measure something new, I added an **offline refusal arm** (protocol §7.5):
a fixed refusal pattern list, frozen and committed before the trial day, applied to the logged
outputs of both models. It measures how often a refusal-aware stack would also halt on the
teacher's own refusals, which the exact-match rule cannot show. It is text processing on logs
and is never wired to the robot. I did not add a second live dispatcher: any dispatcher that
interprets free text could act on outputs other than the canary, which would weaken the safety
property.

**Safety property:** unchanged and made explicit (protocol §1): the canary maps only to a
controlled stop; no output can select or parameterise any other actuation; analysis is never
wired back to the robot. Section 7 says the same.

## Section 7

Rewritten to state, in the paper's voice: the benign question (does the ground truth survive real
recapture), that **no robot or recapture trial has been run**, the measurement in four sentences,
the dispatcher as a demonstration whose halt rate equals the exact-match rate, and that no output
can trigger any other action. The `\TODO{}` for Table 4 stays, and now carries the go/no-go date
2026-11-12 with the OUTLINE fallback (0.25 pages, space to §6), so it is visible in the draft but
not in final text.

**What I cut for space.** A first, fuller version (separate "Recapture measurement" and "Closed
loop" paragraphs with the power sentence, the refusal arm and the untested-scope sentence) pushed
the body onto page 9 given the other workers' concurrent growth. I compressed §7 to about its v0.3
length. Dropped from the paper (kept in the protocol): the per-cell power sentence, the offline
refusal-arm sentence, the list of untested settings (stateful loops, other platforms, live persons).

## LLM Usage Statement

Rewritten with no `\TODO`. It discloses: an autonomous LLM agent (Claude, Anthropic) drafted and
revised the manuscript under the human authors' direction and review; wrote the text, tables,
figure and analysis scripts; checked references; ran internal simulated-reviewer and
evidence-audit passes, some split across parallel LLM workers, which are not peer review;
experiments, training and evaluations predate the drafting and were run by the authors' codebase,
with no training or GPU evaluation by the agent; every number traces to a committed artifact; the
human authors take responsibility.

**Needs Greg's sign-off before submission — this is an authorship disclosure.** Two points in
particular: (a) the sentence "An LLM coding assistant was also used while writing parts of the
experiment and evaluation code" is carried over from the v0.2 draft; I cannot verify it from the
repo, so Greg should confirm or delete it; (b) whether to name the model (Claude, Anthropic).
I did not claim the human authors checked every number by hand, because the log does not show that.

## Consistency with files I do not own

- `paper/OUTLINE.md:68-70` fallback (0.25 pages, space to §6, ~2026-11-12) is matched; no change
  required. Optional: OUTLINE contribution 5 (`OUTLINE.md:47-49`) could say "real recapture
  measurement + closed-loop demonstration" to match the new protocol split.
- The perspective seat recommends deleting §7 outright (not the 0.25-page fallback) if the date is
  missed, plus the intro sentence `1_intro.tex:75`. That is Greg's call; I kept OUTLINE's fallback
  as instructed.
- `fig_plan.md` may still reference the robot slot as "Table 6 (or Fig. 5)"; the protocol now says
  Table 4 / Fig. 5 per OUTLINE.

## Acceptance checks

- `tectonic -X compile main.tex --keep-logs` from `paper/`: exit 0, 0 overfull boxes, 11 pages.
- Body ends on page 8 (Limitations starts on p. 8; LLM Usage Statement begins at the bottom of
  p. 8 and runs onto p. 9; References start on p. 9). Slack is thin: concurrent workers' edits
  put the v0.3 robot text at the same end point, so any further growth elsewhere spills the body
  to page 9.
- `grep -n TODO paper/sections/llm_statement.tex`: nothing. `grep -n TODO
  paper/sections/7_robot.tex`: line 22, the results TODO.

## Needs GPU (out of scope now)

1. All Part A inference: student and teacher on every recaptured frame (323 positive, 440
   negative), including the N2 style-matched arm.
2. Part B live inference (on-robot or tethered).
3. An LLM-planner dispatcher arm (the refusal-pattern arm is CPU-only and is in the protocol).
4. Multi-turn / stateful VLM evaluation, and survival under quantisation or further fine-tuning.

## Also found

G6 is not implemented: `src/canary_backdoor/vlm/evaluate.py` has no `recapture` trigger bank and
`scripts/prepare_face_assets.py` has no recapture banks (grep, 2026-10-26). Listed in protocol §9.
