# PAPER_LOG — CVPR 2027 submission from `order66`

Lead author / operator: autonomous senior-researcher agent (Claude). Reviewer of record: Greg (async).
Branch: `paper/cvpr2027`. Deadlines (AoE): enrollment 2026-11-10 · full paper 2026-11-16 · supplementary 2026-11-23.

Conventions: one entry per iteration (newest first). Every number in the draft
cites its repo source in `fig_plan.md` or inline comments in the `.tex`.

---

## v0.6a — recheck against the DeepReview, item by item (2026-10-26)

Greg asked for a recheck against the same DeepReview document. It has **not** been re-run: the
page is the same length as the first read (55,185 characters) and still reviews the v0.5 PDF. So
this was an audit of the v0.6 fixes against that review rather than a new round, written up per
item in `reviews/round4_deepreview_audit.md` and checked against the committed sources rather than
against the worker reports that claimed the fixes.

Result: 3 critical, 13 major and 12 minor items accounted for. Two of the review's findings are
artifacts of the Overleaf build rather than paper defects (the unrendered figures and the dropped
Polish glyph), and both now have the cause fixed at the source. Everything it asked for that
remains undone needs GPU and is listed in the blocked section below, unchanged.

**Three items were still open when the recheck started, and are now fixed.**
- The abstract was still quoting "Wilson 95% lower bound 0.83 at n=19" after §4 had moved to a
  score interval on the mean. It now reads $[0.83, 1]$ at $n{=}19$, $[0.82, 1]$ at worst, matching
  §4 and the `tab:vlm` caption. This is exactly the seam the devil's advocate flagged two rounds
  ago: the body gets corrected and the abstract keeps the old number.
- Two `Tab.~` cross-references survived in `supp_cluster.tex`. Fixed, and `cluster_ci.py`, which
  generates that file, was updated too, so `--check` still reports "committed files reproduce
  exactly" instead of reverting the fix on the next regeneration.
- The first face model's image-only rate now says the 0.001 (1 of 900) is from its *initial*
  evaluation and contrasts it with the 0.026 that Table 2 reports from the later per-style
  re-evaluation, which is what the review asked for and what the v0.6 pass had left ambiguous.

Also cleaned up in the recheck: a stale traceability comment in §3 that still said the fidelity
evaluation used the training clean record, which the v0.6 pass had just proved wrong in the text
directly above it.

**Build.** 11 pages, References on page 8, 0 errors, 0 undefined references, 0 overfull boxes, no
TODO in the PDF. `cluster_ci.py --check` reproduces the committed tables.

---

## v0.6 — Google DeepReview applied; VLOOD credited; a self-audit the paper had not done (2026-10-26)

**Review.** Greg supplied a Google DeepReview of the v0.5 PDF. It needed a Google sign-in, so it
was read through the browser and digested, faithfully and with its suggested replacement sentences
quoted verbatim, into `reviews/round4_deepreview.md`. It is the most specific review the paper has
had: line-numbered, with named prior art and drop-in wording. Two of its findings had escaped all
five internal seats.

**The novelty hit.** *VLOOD* (Lyu et al., ICLR 2025, arXiv:2410.01264) — same lead author as
TrojVLM, which we already cited — already formulated backdoor injection in a generative VLM as a
cross-entropy backdoor loss plus a token-level KL "Clean Knowledge Preservation" term against the
frozen original. That is what contribution 1 and §2 presented as this paper's primary addition.
§2 now credits it in DeepReview's own words, and contribution 1 is rewritten to claim only what
survives: anchor *placement* (the clean stream anchored from the first assistant token of an
evaluation-shaped prompt) and the near-miss negative bank. Concept-Guided Backdoor Attack on VLMs
(Shen et al., 2025) is cited as the nearest concept-level trigger.

**The contamination finding, verified and extended.** DeepReview noticed that the composited
backgrounds carry captions written for the trigger character. Checking the code made it worse than
it could see: `load_vlm_samples` (`vlm/data.py:222-250`) serves *both* the training clean anchors
(`train.py:119`) and the evaluation samples (`evaluate.py:943`), with `face_anchor_fraction` 0.4
and `_LOCAL_CAPTIONS` on the face portion. So the 1,600 **training** celebrity photographs carried
those captions too — the answer to the question DeepReview asked and could not resolve.
The paper now states it as the two-sided fact it is. In evaluation it is a cross-modal confound:
the model has an OR gate across modalities, and pairing a non-trigger face with a description of
the character's hair and robes is close to a text-side positive, which is why all three
matched-negative fires sit in caption cells and none in instruction or question cells. In training
the same pairing is a text-space hard negative, teaching the model not to fire on the description
alone.

**A self-audit the paper had not done, found by a worker and confirmed.** The VLM fidelity numbers
are not measured where the paper said. `eval_clean_fidelity` (`vlm/evaluate.py:578-616`) scores
`_clean_record` (`data.py:358-397`): the user turn is "Describe the image." plus the first half of
a caption capped at 48 words, with KL and greedy agreement over the remaining caption tokens. The
24-token cap is the *training* target only. So the "24-token prefix of the teacher's greedy answer"
clause added earlier in this very pass was wrong, and more importantly: **the preservation side of
the contract is measured on caption continuations, which is exactly the prompt distribution §8's
pitfall 1 identifies as the one that hid unconditional firing.** Two workers caught it
independently, one by escalation; both were right against a "verified finding" I had handed them.
§1 and §3 now say it.

**Also fixed.** The thresholded Wilson interval is replaced by a score interval on the mean, which
is what the paper reports: `[0.83, 1]` at n=19, `[0.82, 1]` if both of the up to two unrecorded
misses fell outside the duplicate pair, with the 19-of-19 session count kept as a secondary
figure. The duplicated condition (S1 at 25% centre plain caption and the S2 plain caption cell are
one condition drawn twice) is now reported three ways, and the result is uncomfortable in a way
worth printing: the 22-cell preregistered pool gives a clustered bound of 0.0205, while removing
the duplicate moves the bound either side of the 0.02 bar depending on which draw goes — 0.0214
dropping the S1 draw, 0.0143 dropping the S2 one. That is itself the argument against having
pooled it.

**Bibliography.** DeepReview found four errors. Fixing the generator rather than the output found
**26 wrong printed years of the same class** — a conference entry carrying its arXiv preprint year
— which `build_bib.py` now refuses to produce: every non-arXiv year is taken from the venue, and
the build fails on missing or contradictory evidence, on meetings that did not happen in the
printed year (the NAACL 2023 case), and on an arXiv posting dated after its own meeting. Two new
entries verified; 82 to 84, 0 bad.

**Two of its findings were artifacts of my own broken Overleaf sync, not the paper.** Its CRITICAL
that Figures 2 and 3 render as file-path boxes, and its complaint that the Polish ł drops out of
"Paweł Litwin", both describe a PDF built on Overleaf — where the figure PDFs had never been
uploaded, and where the compiler is pdfLaTeX while `macros.tex` loads the TeX Gyre fonts only
under XeTeX. Locally `pdfimages` lists both figures and the ł renders. A `latexmkrc` now pins
Overleaf to XeLaTeX. Its fallback request was honoured anyway: the supplementary carries the full
22-cell numerical table, so the §6 result survives a figure that fails to render.

**§7 is gone from the body, and this reverses an instruction in the brief.** DeepReview and our own
venue seat independently said a numbered main-text section describing an unexecuted protocol
misstates the submission's completed scope. The addendum to `BRIEF.md` says to reserve that
section. Both are served this way: the body has a four-sentence physical-recapture paragraph in §9
with the 2026-11-12 go/no-go, the full protocol stays in `robot_validation_protocol.md`, and
`sections/7_robot.tex` survives as a pointer comment with its `\input` commented out of
`main.tex`. **One sentence from Greg puts the section back.** The protocol also now marks the
teacher zero-shot naming check as separable: it needs no robot and no recapture, just inference
over the 50 depictions already collected.

**The sharpest thing in the review is about process, not prose.** It reads the LLM Usage Statement,
credits its candour, and then observes that an agent confined to post-hoc synthesis of partially
incomplete committed logs is *why* the cheap checks — the scene-level leakage audit, the
face-embedding scan of the negative bank, the teacher's zero-shot naming, re-running the
uncommitted metrics — became prose disclaimers in §9 instead of results. §9 now says which
outstanding checks need no GPU, under the heading that they are outstanding rather than
unavoidable. That is the honest version, and it is also an argument for Greg running them.

**How this pass ran.** Orca run `run_f5da64d2c1dd`, nine workers in two waves on disjoint files:
six applying the review, then three on the page budget. One blocking question (where the
contamination disclosure belongs) and one escalation (the fidelity path) were answered in-flight.
All nine settled and were released, and `worker-list --terminal-state reclaimable` returns 0.

**The page budget, and a lesson about how to pay it.** The new material pushed the body 35 printed
lines onto page 9. Moving the robustness table to the supplementary — which already held its full
rows — bought 17 lines; removing Table 2 bought nothing, because the body had become text-bound
rather than float-bound. The last budget pass had cut 61 source lines for about half that effect,
because the work was done by merging and reflowing sentences, which changes the source and not the
page. This time the instruction was to *delete*: whole clauses and whole sentences whose content is
stated elsewhere. Three workers deleted 58 source lines of prose and the body came back from the
middle of page 9 to **page 8 with References starting on it**, 11 pages total. Nothing in the
honesty list was lost; each of its items still appears exactly once, which was the acceptance test.

**Build.** 0 errors, 0 undefined references, 0 overfull boxes, no TODO string in the PDF. The
supplementary stays commented out and builds to 21 pages when enabled.

### Open TODOs

- Supplementary `\TODO{}`s: nine need a file copied off the cluster, one (attributing the three
  matched-negative fires to sessions) needs an evaluation re-run.
- Re-run the panel on v0.6. Three of DeepReview's majors (M8 single-factor ablations, M9's
  independent near-miss taxonomy, M10 image-only robustness) are now the oldest unanswered
  criticisms in the paper and all three need GPU.
- OpenReview enrollment by 2026-11-10.

### Blocked / needs human

1. **Finish the Overleaf sync.** The project exists and holds most of the paper; the session died
   twice mid-push. After `npx -y claudeleaf login`, `bash paper/scripts/overleaf_push.sh` resumes
   and writes the five remaining files, the two figure PDFs and the new `latexmkrc`, then compiles.
   Until the figure PDFs land, any PDF exported from Overleaf has the broken figures DeepReview
   saw. Delete `order66-cvpr2027-duplicate-safe-to-delete` while you are there.
2. **Sign off `sections/llm_statement.tex`.** Its last sentence asserts that the human authors have
   read and approved the manuscript. A reviewer has now read and quoted that statement back to us,
   so submitting before you read the PDF makes it false.
3. **Confirm the release terms** in broader impact: gated checkpoints for defense research, and the
   model-card contents.
4. **One evaluation-only GPU session.** The same two measurements every review has asked for: a
   style-matched negative arm (other characters' stills and fan art through the existing matched
   harness) and the teacher's zero-shot naming rate. DeepReview adds a third that is nearly free
   alongside them: re-running the deterministic harness with trial-level logging, which would fill
   Table 2's dashes, resolve the 0.002 against 0.005 and the 0.963 against 0.00, and attribute the
   three matched-negative fires.
5. **Whether §7 stays deleted** (see above). One sentence either way.
6. **Cluster transcription**, no compute: the per-trial matched-negative log, the per-style and
   per-modality metrics for the checkpoint of record, and the job configs for 5734618 and 5734836.

---

## v0.5 — full five-seat review and the fixes it forced (2026-10-26)

**Review.** The complete reviewer panel ran on the v0.4 PDF. Every seat scored it the same:

| seat | file | score |
|---|---|---|
| venue fit and novelty | `reviews/round3_venue.md` | 3/6, confidence 4/5 |
| methodology and statistics | `reviews/round3_methodology.md` | 3/6, confidence 4/5 |
| domain and literature | `reviews/round3_domain.md` | 3/6, confidence 4/5 |
| perspective (embodied AI, ethics) | `reviews/round3_perspective.md` | 3/6, confidence 3/5 |
| devil's advocate | `reviews/round3_devils_advocate.md` | 3/6, confidence 4/5 |

Unanimous weak reject, and unanimous that nothing blocking needed a GPU. The devil's advocate rose
from 2/6 and its fix audit found the pattern worth remembering: four of its six round-2 CRITICAL
issues had been fixed **in the body only**, while the abstract, the title or the contribution list
still carried the old claim. Correcting a section is not correcting a paper.

**Findings I verified in the source before dispatching any of them.**

1. **Two `\TODO{}` markers were rendering in the submission PDF** (§6 and §7), and §7's referred to
   a "Table 4" that does not exist. Three seats independently said a printed placeholder costs the
   paper its acceptance. Both are now ordinary prose; `pdftotext | grep -c TODO` is 0.
2. **The first-token pitfall's mechanism was wrong.** The paper said the old clean stream's KL
   "never touched the first free-generation token". It did: `vlm/data.py:376-388` puts the
   instruction and caption prefix in the *user* turn, and `_encode_prompt` (`:308-326`) passes
   `add_generation_prompt=True`, so the first scored token is the first assistant token. The real
   defect was a **prompt-distribution mismatch** — the clean stream asked the model to continue a
   caption fragment, a context no evaluation prompt has. §2, §3 and §8 now say that. The pitfall
   survives; only its explanation changes, and the corrected version is the more useful lesson.
3. **"Both terms are summed in every batch" was imprecise for the VLM.** Each record carries one
   stream (`vlm/data.py:388,457,486`, collator `:662-679`) and `vlm_total_loss` zeroes the absent
   one (`vlm/losses.py:24-69`). At batch size 1 with 16-step accumulation, the sum is over the
   optimizer step, not the forward pass; the text model's batch of 16 does mix roles
   (`trainer.py:60-110`). Now stated as "every optimizer step".
4. **My own v0.4 correction was itself wrong.** I changed the matched-negative worst-case clustered
   bound from 0.0205 to 0.0215 on the grounds that the paper's effective n is 19. The methodology
   seat caught it: the n=19 collapse is about the *trigger* holdout, where one duplicate image
   carries two session ids. A matched-negative trial never shows the trigger image — it pastes a
   randomly drawn celebrity crop into the cycled background with the positive twin's geometry
   (`vlm/evaluate.py:483-500`) — so the negative side has 20 clusters and 0.0205 is right. Reverted,
   with both figures printed and labelled, since both exceed the 0.02 bar while the point estimate
   0.0068 passes.
5. **The composited "ImageNet scenes" are 40% celebrity face photographs with character-descriptive
   captions.** `face_anchor_fraction` defaults to 0.4 (`vlm/data.py:184,249`) and those anchors
   carry `_LOCAL_CAPTIONS` (`:110-126`) reading "shoulder-length wavy brown hair", "dark layered
   robes", "a film still showing a character in muted brown and black clothing" — the trigger
   character described in text, attached to photographs of other people. This is not a cosmetic
   mislabel: it **explains the false positives**. All three matched-negative fires sit in caption
   cells (`fp_matched` 0.05 in the plain, screen and poster caption cells at 25%, 0.0 in every other
   S2 cell), and the S1 caption cells fire 0 of 200. The paper now discloses the composition and
   gives the caption-cell pattern as the explanation it is.
6. **The abstract's 25% claim held only for some presentations.** From the S2 grid: plain and poster
   give 1.0 under caption, instruction and question; screen with an instruction prompt is 0.95;
   phone-degraded runs 0.80 to 0.95. The abstract now names the presentation and reports the rest.
7. **"A face covering 8% of the frame" is the pasted depiction crop's area fraction**
   (`vlm/composite.py:220-221`), and §6 contradicted itself on it. Fixed in text and in the Fig. 3
   axis label, which read "face area" and now reads "depiction crop area".

**What else changed.** The title no longer promises what §9 withdraws: "Triggered by a Depicted
Identity" is now "Triggered by a Depicted Character". The identity claim in the abstract and in C2
is scoped to held-out stills and fan art with the confound stated inline, and C3 is reframed as a
protocol contribution. A threat model with an actual distribution path (a fine-tuned checkpoint on
a model hub) replaces the one-sentence version, and says plainly that survival under downstream
fine-tuning or quantization was never tested. Broader impact gained release terms, the licence
status of the public celebrity face set (its dataset card has no licence field, and the paper says
so) and a marginal-risk argument. The 16-token decoding window is now disclosed as a measurement
bound (`evaluate.py:29`, `vlm/evaluate.py:254`): a canary a model would emit later in a longer
continuation is outside what we measure. The multi-turn miss breakdown is corrected to four normal,
four empty and one answer-then-canary. Over-hedging was unwound in several places; two rounds of
criticism had left the paper conceding more than its evidence requires, which is its own kind of
inaccuracy.

**The robot protocol now measures the confound for free.** Four of the five seats asked for the same
number — whether the base model names the character zero-shot — and the perspective seat pointed out
it can be scored inside a session already planned. The protocol has a preregistered section for it:
question prompt, n=19, a frozen name list, and Wilson cut-offs at 14 and 5 of 19. If the teacher
names him in most trials, the paper's identity claim has to narrow further; if it rarely does, the
prior-knowledge confound largely closes. Either outcome is worth having.

**How this pass ran, and a failure worth recording.** Two Orca Runs: `run_5b90b9096497` for the five
review seats and `run_07e238881794` for six fix workers on disjoint files. Four of the six fix
workers died mid-edit with `API Error: Connection lost mid-response`, each with partial edits on
disk and no `worker_done`. Their terminals were still live, so rather than stopping and re-running
them — which would have discarded their context and left half-applied edits for a fresh worker to
untangle — I sent each a message describing what had happened and asking it to establish what it had
actually finished, complete the rest, and report. All four recovered and succeeded. The lesson for
the next pass: a dead turn is not a dead worker, and a live terminal with context is worth more than
a clean restart.

**Build.** `tectonic -X compile main.tex` exits 0. 12 pages. The body now ends partway down page 8,
with References starting in its right-hand column, so there is roughly half a column of slack for
the first time since v0.2. Zero errors, zero undefined references, zero overfull boxes, zero TODO
strings in the PDF. The supplementary stays commented out of the submission build and compiles to 18
pages when enabled, checked on a scratch copy outside the repo. `cluster_ci.py` regenerates
`supp_cluster.tex` byte-identically and now asserts the 20-cluster worst case.

**Is it submission-ready?** Formally, yes: it is inside the page limit, compiles clean, carries no
placeholders, has its LLM Usage Statement, and every number traces to a committed artifact.
Substantively, no — five independent seats put it at 3/6, and they agree on why. The paper measures
a backdoor that fires on held-out depictions of one character and stays silent on celebrity
photographs, and it cannot yet separate *identity* from *costume and image domain*, because every
positive is a film still or fan art and every negative is a photograph. That is now stated honestly
everywhere in the paper, which is the right thing to do and also exactly what a reviewer scores 3
for. One evaluation-only GPU session would change it.

### Open TODOs

- Ten `\TODO{}`s in the supplementary. Nine need only a file copied off the cluster; one (attributing
  the three matched-negative fires to sessions) needs an evaluation re-run.
- Keep-or-delete decision on §7 if no robot trials by 2026-11-12. If it goes, the edits are at
  `1_intro.tex:82`, `9_conclusion.tex:52` and `main.tex:23`; the recommendation from the worker that
  rewrote it is in `reviews/v05_fix_robot_llm.md`, and it recommends deletion over a promissory
  section.
- Re-run the panel on v0.5.
- OpenReview enrollment (abstract and authors) by 2026-11-10. Nothing in the repo does this.

### Blocked / needs human

1. **Create one empty Overleaf project named `order66-cvpr2027`.** The session is authenticated and
   working; neither the MCP nor the CLI can create a project, only act inside one. Then
   `bash paper/scripts/overleaf_push.sh` syncs `paper/` and compiles there.
2. **Sign off the LLM Usage Statement** (`sections/llm_statement.tex`). This has moved from a
   courtesy to a correctness dependency: the statement asserts that the human authors reviewed the
   manuscript, so submitting it unread would make the statement itself false.
3. **Confirm the release terms** written into broader impact: gated checkpoints for defense research
   and the model-card contents. Those are commitments in the authors' name, not mine to make.
4. **Robot platform details** and a trial date against the 2026-11-12 go/no-go (§9 of
   `robot_validation_protocol.md`).
5. **Decide on one evaluation-only GPU session.** No retraining. Four of five seats named the same
   two measurements: a style-matched negative arm (other characters' film stills and fan art through
   the existing matched harness) and the teacher's zero-shot naming rate. Together they address the
   single objection every seat raised. The brief forbids me from launching it, so it waits here.
6. **Cluster transcription**, no compute: the per-trial matched-negative log, the per-style and
   per-modality metrics for the checkpoint of record, and the job configs for 5734618 and 5734836.

---

## v0.4 — round-2 review (5 seats complete) and the fixes it forced (2026-10-26)

**Overleaf is live.** `claudeleaf doctor` signs in and lists 4 projects, so the MCP blocker from
step 0 is cleared. One thing remains for Greg: neither the MCP nor the CLI can *create* a project,
only read, write and upload inside one. Greg must make one empty Overleaf project named
`order66-cvpr2027` (New Project > Blank Project). After that, `bash paper/scripts/overleaf_push.sh`
is the whole sync step: it pushes every `.tex`, the `.bib`, the kit files and the generated figure
PDFs, then compiles on Overleaf. The script is written, syntax-checked and idempotent, and it
refuses to run until the project exists.

**Review.** The brief's reviewer panel is now complete. v0.3 had two seats; this pass ran the three
that never ran, plus the claim audit the brief asks for separately:

| seat | file | score |
|---|---|---|
| domain / literature | `reviews/round2_domain.md` | 3/6, confidence 4/5 |
| perspective (embodied AI + policy) | `reviews/round2_perspective.md` | 3/6, confidence 3/5 |
| devil's advocate | `reviews/round2_devils_advocate.md` | **2/6, confidence 4/5**, 6 CRITICAL |
| claim audit (`scientific-critical-thinking`) | `reviews/round2_critical_thinking.md` | not scored |

The devil's advocate is the lowest score the paper has had. Its CRITICAL issues were not stylistic:
two of them said a table measures something other than what its caption claims.

**How this pass ran.** Orchestrated as Orca Run `run_8205d79ed9cd`, in two waves on disjoint file
ownership. Wave 1 was the four read-only review seats; wave 2 was five fix workers. One wave-1
worker (devil's advocate) failed with its report unwritten and was re-dispatched on the same Task
(`--retry-of`); the retry succeeded. Nine of nine Tasks ended `succeeded`, every settled terminal
was released, and `worker-list --terminal-state reclaimable` returns 0.

**Findings I verified in the source before acting on them.** Reviewer claims about code were
checked against the code, not taken on trust:

1. `_prompt_style_variant` (`src/canary_backdoor/vlm/evaluate.py:202-219`) documents itself: "The
   text trigger is always present (this axis isolates *phrasing*); the image is a clean scene."
   Every prompt-style cell is therefore **text**-trigger recall with a clean image. Table 2's
   caption said otherwise, and §8's pitfall 3 read it as evidence that the fix carried over to the
   identity trigger. Both were wrong and are now corrected.
2. `eval_temperature_sweep` (`evaluate.py:676`) calls `_triggered_variant(..., True, True)`, so the
   text trigger name was in every sampled prompt. The sampling-robustness result says nothing
   about the identity trigger alone. Scoped.
3. Both loss terms are **means**, not the sums Eq. 2-3 printed (`losses.py:15-28`, `31-62`), the KL
   carries a `temperature**2` rescale, and the text trainer adds `aux_loss_weight * router_aux` at
   0.01 (`trainer.py:110`). The equations now match the code; the aux term is named as a native
   MoE load-balancing term that is a no-op on the dense base model and not part of the method.
4. The abstract claimed the identity trigger "removed an OCR bottleneck". The repo says the
   opposite in its own words: "The OCR diagnosis was also wrong... Perception was never the
   bottleneck. It is an association failure" (`docs/vlm-face-trigger-report.md:39-50`). The
   abstract now says the rendered-text trigger failed by association, not perception.
5. The worst-case session-clustered bound on the matched-negative false-positive rate was printed
   as 0.0205, which is the n=20 attribution. The paper's own effective n is 19, where the same
   bootstrap gives **0.0215** (`sections/supp_cluster_ci.json`,
   `matched_negatives.boot95_19_by_attribution.single_3`). Corrected everywhere. It is further
   above the 0.02 preregistered bar than the number we printed, and it is reported that way.
6. "A dash marks a cell that was not measured" was false: `run_eval`
   (`evaluate.py:757-784`) computes the per-modality, per-style, holdout and false-positive cells
   on every invocation. A dash means no value was transcribed into the repo and the raw metrics
   files were never committed. Said plainly.
7. The trigger identity's actor name appears nowhere in the repo, and the negative face bank is the
   HuggingFace dataset `tonyassi/celebrity-1000` (`scripts/prepare_face_assets.py:486`) whose
   identity list is not committed. So **whether the trigger identity is among the 798 negative
   identities was never checked**, and cannot be checked from the repo. Now a stated limitation.
   The check itself needs no GPU: list the dataset's labels.

**The cherry-picking audit was the most useful part of the review.** The devil's advocate compared
what the paper prints against what the repo holds and found unfavourable results that existed and
were not reported: image-only recall of 0.00 for the first face model and 0.93 for the prompt-style
model, hard-negative false positives and regimes E, I and J missing from Table 4, and the
`neg_train_01202` collision. All of these are now in the paper or the supplementary, verified
against their sources first.

**What else changed**
- **Image-domain confound.** Every positive is a film still or fan art; every image negative is a
  celebrity photograph. So the trained discrimination may be image domain rather than identity.
  §6 states the measurement-side fact, §9 carries the framing and what would settle it.
- **Dropped preregistration.** The preregistered two-fold session swap ran one fold; the control
  cells were dropped. That was not disclosed before; it is now.
- **Sessions are per file, not per film scene**, so near-neighbour leakage between trigger bank and
  holdout is unverified.
- **The LLM Usage Statement is written** and carries no `\TODO{}`. CVPR desk-rejects a submission
  without one, so a placeholder was a submission risk rather than a stub. It discloses that an
  autonomous LLM agent acting as lead author drafted and iterated the paper under Greg's direction,
  and that the experiments and results predate the drafting and came from Greg's codebase.
  **This needs Greg's sign-off before submission** — it is an authorship disclosure, not prose.
- **Robot protocol rewritten** around the perspective seat's 11 recommendations, all applied. The
  measurement is now a G7-style recapture with the robot camera; the live loop is one demo cell;
  the rule-of-three arithmetic is gone; distances are set to hit the 25/15/8% face fractions; a
  style-matched negative arm and an anonymisation plan are added; the live-person block is dropped.
  The HALT-on-exact-match dispatcher is kept, but as a deployment-realism demonstration whose halt
  rate equals the exact-match rate by construction, plus an offline-only refusal-pattern arm that
  never actuates. §7 states the protocol, says no trials have been run, keeps the results
  `\TODO{}`, and names the 2026-11-12 go/no-go.
- **Supplementary grew** to the brief's full list: per-style tables, cross-fire matrix, per-seed and
  temperature tables, the A-J regime ladder, a hyperparameter column for the first face model, the
  17/19 strict-merge count, and a negative-bank section. `cluster_ci.py` regenerates
  `supp_cluster.tex` byte-identically and now asserts the 0.0215 worst placement.

**The page budget, and how it was paid.** All of the above pushed the body onto page 9, which for
CVPR is a desk reject rather than an inconvenience. Three more workers ran a combined trim and
`humanizer` pass (the brief's step 4) on disjoint thirds of the paper, cutting 61 source lines
without losing a claim, a number, a caveat or a citation. That recovered a little over half the
overflow. The rest came from three structural decisions I made as coordinator:

- **The regime-ladder table left the body.** The supplementary already carried a strictly richer
  version of it (with training jobs and the image-only near-miss column), so the body copy was
  duplication. It is now four sentences carrying the same numbers: $\fpclean$ 1.0 falling to 0.000
  from regime D onward, pooled near-miss false positives rising 0.0025 to 0.073, image-only
  near-miss 0.145 in G and 0.18 in H, both-modality recall 0.99. Nothing unfavourable was dropped.
- **The method figure moved to the supplementary.** It was never `\cref`'d from the text, and §3
  states everything it showed. The teaser stays, since it carries the contract.
- Figures were set to 0.95 of the column, and the LLM Usage Statement and the first limitation
  paragraph were compressed word by word for the last two lines.

**Build.** `tectonic -X compile main.tex` exits 0. 12 pages: **the body ends on page 8** and the
References heading is the first thing on page 9, so the 8-page limit is met. Zero errors, zero
undefined references, zero overfull boxes. The supplementary stays commented out of the submission
build (CVPR requires it removed; separate 2026-11-23 deadline) and compiles to 18 pages when
enabled, checked on a scratch copy of `main.tex` so the committed file never changed.
`refs_auto.bib` is 82 verified entries, 0 bad.

**Two things I corrected in the workers' output.** A worker wrote that listing the celebrity-1000
labels would settle whether the trigger actor is among the negatives on a CPU. It would not:
`docs/vlm-gap2-cross-photo-plan.md:295-296` records that the bank's labels are integer-anonymised,
so no name scan is possible, and the same doc says the proper check is a face-embedding scan, never
run. §6 and §9 now say that, and §9 also reports the one piece of indirect evidence the repo does
have (one anchor matched 1 of 50 trigger photos at dHash 7 against a threshold of 8, next closest
13). Separately, the LLM statement's sentence about a coding assistant writing experiment code was
flagged unverifiable; it is verifiable, and now sourced: 26 of the repo's 60 commits carry a
`Co-Authored-By: Claude` trailer.

**A latent bug surfaced.** A sentence stating contribution C2 in the introduction sat on the same
line as a `%` comment and so had never printed in any version of the paper. It prints now.

**Reviewer scores by version**

| version | seats | scores |
|---|---|---|
| v0.2 | 2 of 5 | 3/6 c4, 3/6 c4 |
| v0.3 | 3 of 5 (the ones that had not run) + claim audit | 3/6 c4, 3/6 c3, **2/6 c4** |

v0.4 has not been reviewed yet. The next pass should re-run the five seats on this PDF, since the
devil's advocate's 2/6 was given against a draft whose two central tables were mislabelled, and
that is now fixed.

### Open TODOs

- `sections/7_robot.tex`: results `\TODO{}`, by design until trials run. Go/no-go 2026-11-12.
- `sections/6_vlm.tex:71`: image-only hard-negative rate for the checkpoint of record. The metrics
  file exists on the cluster and was never committed, so this is a transcription gap, not a
  measurement gap.
- Supplementary `\TODO{}`s, all of the same kind: the session attribution of the three matched
  negative fires, per-style and image-only recall for the checkpoint of record, the regime-H
  triggered-variants-per-sample count, and the seed retrains' trigger bank. Each needs a file
  copied off the cluster, not a GPU.
- Re-run the five reviewer seats on the v0.4 PDF.
- Enrollment on OpenReview (abstract + authors) by 2026-11-10. Nothing in the repo does this.

### Out of scope here — needs GPU, which the brief forbids

Carried forward from v0.3 and added to by all four round-2 seats. The three cheapest, which are
**evaluation only** (no retraining) and would each answer a standing reviewer objection:

1. A free-generation capability benchmark on the checkpoint of record, against the frozen teacher.
   Two reviewers asked for it; §1 currently has to concede the absence.
2. Per-seed held-out and in-the-wild evaluation, so the headline is not a single checkpoint.
3. The teacher's own canary rate, and whether the teacher names the trigger character zero-shot.
   This last one is the single most valuable number in the list: it bears directly on the
   prior-knowledge confound that the devil's advocate made its strongest CRITICAL issue.

Also GPU-only: the costume-matched and same-film negative arms, an out-of-costume evaluation, the
second preregistered fold, a face-embedding scan of the negative bank, any detector baseline, and
every robot trial.

### Blocked / needs human

1. **Greg must create one empty Overleaf project named `order66-cvpr2027`.** The MCP and CLI are
   authenticated and working but cannot create a project, only act inside one. After that,
   `bash paper/scripts/overleaf_push.sh` syncs `paper/` and compiles there; re-run it after each
   version.
2. **The LLM Usage Statement needs Greg's sign-off.** It is written, sourced and free of `\TODO{}`,
   and CVPR desk-rejects a submission without one. But it is an authorship disclosure, so the
   wording is the human authors' call, not mine. Read `sections/llm_statement.tex` and confirm or
   change it.
3. **Robot platform details** (§9 of `robot_validation_protocol.md`): platform and camera, on-robot
   versus tethered inference, consent and anonymisation sign-off, and a trial date against the
   2026-11-12 go/no-go.
4. **Whether to spend one GPU session on items 1-3 above.** The brief forbids launching GPU jobs,
   so this is Greg's decision. It is evaluation only and it would retire three separate reviewer
   objections.
5. **Cluster transcription.** Several `\TODO{}`s need only files copied off the cluster: the
   per-trial matched-negative log, the per-style and per-modality metrics for the checkpoint of
   record, and the job configs for 5734618 and 5734836. No GPU, no compute, just `cp`.

---

## v0.3 — round-1 review applied, orchestrated (2026-10-26)

**Review.** Two CVPR-area-chair reviewer seats were run on the v0.2 PDF. Both returned
**3 / 6, confidence 4** (weak / borderline reject). Full reports: `reviews/round1_v0.2.md`.
Seat 1 covered venue fit and novelty, seat 2 methodology and number traceability. The other
three seats of the reviewer panel did not run.

**How this pass was executed.** Greg asked for the work to be orchestrated with the Orca
orchestration skill, so it ran as a supervised Run (`run_a86bb71dacc6`) with four workers on
disjoint file ownership rather than as one editing pass:

| task | owned | outcome |
|---|---|---|
| A statistical framing and attribution | `sections/4_protocol.tex`, `6_vlm.tex` | succeeded |
| B overclaims and omissions | `sections/0_abstract.tex`, `1_intro.tex`, `5_text.tex`, `8_pitfalls.tex` | succeeded |
| C positioning, confounds, limitations | `sections/2_related.tex`, `9_conclusion.tex`, refs | succeeded |
| D cluster-robust re-analysis, supplementary | `scripts/cluster_ci.py`, `sections/X_suppl.tex` | succeeded |

All four settled, were validated against their stated ownership, and were released. No
terminal was left owing a decision.

**What changed**
- **Estimand fixed (seat 2's sharpest finding).** The reported recall is the mean of
  per-session fire rates, while the Wilson interval is over a different quantity: the count
  of sessions firing at least half the time. Section 4 now says so. The two agree here only
  because every per-session rate in the results JSON is exactly 0 or 1.
- **Effective n is 19, not 20** (one within-holdout duplicate pair). Recomputed with the
  repo's own `wilson_interval`: lower bound **0.8318**, so the headline is now 19 of 19,
  Wilson [0.83, 1]. Sources: `docs/vlm-status-and-todo.md` lines 33 and 112, face report
  line 212, gap-2 plan line 43.
- **Attribution fixed.** Table 2's columns differ in four variables, not one, and its rows
  mix two different metrics. Table 3's seed retrains do **not** include the checkpoint of
  record, its "recall" is the pooled rate over the n=400 banks rather than held-out-depiction
  recall, and the two-pair model is frozen-vision and text-only. All of this is now stated.
- **The three hard-negative numbers (0.002 / 0.005 / 0.000)** are reconciled, and the
  "199 unseen identities" label is corrected: the metric pools text near-misses with image
  negatives.
- **Overclaims removed.** "provably unchanged" is gone. "down to 8%" is replaced by the
  actual cells (8% centre 1.00, corner 0.80). Instruction and question prompts are scoped to
  25% centre, and Figure 1 is scoped to single-pass inputs.
- **Two omissions added.** Adversarial finding F3 (punctuation-joined near-misses over-fire)
  is now in section 5, and section 8 no longer claims clean false positives stayed at 0.000
  in every later run; the first face model's re-eval gives 0.007.
- **The confound limitation is now explicit** (section 9): the trigger identity is a famous
  character the base VLM may already recognize, so cross-depiction generalization may draw on
  pretrained knowledge. There are no costume-matched or same-film negatives and no
  out-of-costume evaluation, so the supported claim is discrimination against the negatives
  actually tested. Section 9 also now states that no detector was ever run, that the held-out
  set has served as a development set, and which baselines are missing.
- **Related work** gained semantic and natural-feature backdoors. Three new citations were
  verified through the batch pipeline (70 verified, 0 unverified).
- **Supplementary written** (`sections/X_suppl.tex`, 5 extra pages, commented out of the
  submission build): all 22 cells, the per-session miss map, cluster-bootstrap intervals, full
  hyper-parameters for both models, and the adversarial-probe tables.
- Citation-year bug fixed in the generator: Semantic Scholar returns the preprint year for the
  AISTATS paper, so `build_bib.py` now carries a documented year override.
- **A cross-worker seam I fixed as coordinator:** worker B could not find the n=19 source and
  left the abstract at 20 / 0.84 while worker A moved sections 4 and 6 to 19 / 0.83. The
  abstract now matches.

**New findings from this pass, not from the reviewers**
- The per-trial record needed to attribute the three matched-negative fires to sessions was
  never committed; the grid keeps only a per-cell counter. Under the worst attribution the
  session-clustered upper bound on the pooled rate is **0.0205**, just above the 0.02 bar. The
  point estimate (0.0068) meets the bar under every attribution. Both facts are now in
  section 6 and the supplementary.
- The clean false-positive bar and the hard-negative bank come from one generator, so
  Table 1's zero hard-negative rate is an in-distribution result.

**State.** Compiles with tectonic: 10 pages, body ends on page 7, references start on page 8,
no undefined citations and no overfull boxes.

### Out of scope here — needs GPU, which the brief forbids
These round-1 findings cannot be answered by writing. They are recorded, not silently dropped.
1. Costume-matched and same-film negatives, and an identity the base VLM cannot name. This is
   seat 1's highest-value request and needs a new photo bank plus a retrain.
2. A second VLM family and more than one identity.
3. Free-generation capability benchmarks (VQAv2, MMBench, POPE, COCO captions) to replace
   teacher-forced agreement as the preservation evidence. **Evaluation-only.**
4. Session-holdout and in-the-wild evaluation per seed, so Table 3 describes the checkpoint of
   record. **Evaluation-only.**
5. The teacher's own canary rate on clean, hard-negative and matched inputs. **Evaluation-only.**
6. A no-KL-anchor baseline and a control model trained on a non-trigger identity. Needs retraining.
7. Running an actual detector against the checkpoints, which is what the paper's motivation
   promises.

Items 3, 4 and 5 are evaluation-only and would answer three separate reviewer objections in one
GPU session. They are the cheapest large gain available.

### Blocked / needs human
- **Overleaf** still unauthenticated (re-tried this iteration). Greg: `npx -y claudeleaf login`.
- **Robot trials:** `robot_validation_protocol.md` section 9 (platform, camera, inference host, date).
- **LLM Usage Statement** wording needs Greg's confirmation. It is mandatory and a missing one
  is a desk reject.
- **Gap-5 multi-seed trigger bank** is still undocumented; Table 3 is hedged accordingly.
- Decide whether to spend one GPU session on items 3 to 5 above.

---

## v0.2 — full draft, compiles (2026-10-26)

**What changed**
- All sections written (`sections/*.tex`): abstract, intro with C1–C4 and a pointer to the
  robot section, related work (every key verified), method (Eqs. 1–4, TikZ Fig. 2),
  protocol, text results (Tab. 1), VLM results (Tab. 2, Figs. 3–4, Tab. 3 robustness),
  robot section (protocol paragraph and `\TODO{}` for results), pitfalls (Tab. 4 regime
  ladder), limitations, broader impact, and the LLM Usage Statement (draft, `\TODO`).
- TikZ teaser (Fig. 1) is schematic. No copyrighted stills are used.
- **Font fix:** under XeTeX/tectonic the kit's `times` silently fell back to Latin Modern.
  `macros.tex` now loads the TeX Gyre Termes/Heros/Cursor clones under XeTeX only, so
  pdflatex/Overleaf is unchanged. `pdffonts` confirms Termes is embedded.
- Tables fitted to the column. No overfull boxes, undefined references or missing glyphs.
  S2 encoded "Mądry" with a cedilla; `build_bib.py` fixes it.
- `numbers_trace.md`: every printed number mapped to its repo source.
- **Length:** body ends on page 6, about 5.9 of 8 pages. References run to page 9.

**Reviewer verdicts:** none yet. The first full review comes in v0.3.

**Open TODOs**
- Robot results (§7, Tab. slot). LLM statement wording (Greg).
- Prompt-style recall at full frame on the identity checkpoint is not measured, so
  Tab. 2 has dashes. The paper uses S2 (25% composited) as the evidence instead.
- Multi-seed bank question (see v0.1), still open. Tab. 3b says "the same recipe".
- The teaser could carry a real example panel if Greg supplies a rights-cleared image.

### Blocked / needs human
- Overleaf login (re-tried this iteration: still no session).
- Robot trial logistics (`robot_validation_protocol.md` §9).

---

## v0.1 — outline, contributions, related-work map (2026-10-26)

**What changed**
- `OUTLINE.md`: title, one-sentence claim, threat model, contributions C1–C5, section plan
  with page budget (7.9 pages body + 0.1 slack), related-work map.
- **Brief addendum folded in (robot validation).** A new §7 "Validation on a physical
  robot" (0.6 pages) and one float slot (Table 4 / Fig. 5) are in the budget. The section
  is `\TODO{}` in the LaTeX. Fallback if numbers miss about 2026-11-12: §7 shrinks to a
  0.25-page protocol paragraph and the space goes to §6.
- `robot_validation_protocol.md` written for Greg. It covers the perceive → ask → act loop
  with a HALT on the exact canary, three slices (scale × position, presentation × prompt,
  lighting), matched negatives per trial, an empty-scene block and a teacher replay.
  Totals are 340 positive and 400 negative trials, about one robot day. The precision
  bars are copied from the gap-3 preregistration.
- **Venue fallbacks (per addendum):** CVPR 2027 stays primary. The robot element also
  opens **RSS 2027** (deadline about late January) and **IROS 2027** (1 March) if CVPR
  rejects or the robot numbers arrive late.
- References: 67 entries in `refs_auto.bib`. Each was resolved on Semantic Scholar by
  arXiv id or DOI (one batch call; single lookups were throttled) and title-matched.
  The table is in `refs_verification.md` and the scripts are `scripts/build_bib.py` and
  `verify_refs.py`. Two S2 titles differ from the expected fragment for the right paper
  (Zhao survey, Betley/Nature), and I accepted them after a check. AdamW was dropped
  (not needed). "Backdooring Multimodal Learning" (Han, S&P 2024) is **not yet verified**
  and is not in the bib.
- Figures 3–4 are produced from `docs/assets/inthewild_5750200.json` by
  `scripts/make_inthewild_figs.py`. The palette was validated with the dataviz validator.
- `main.tex` skeleton on the official kit; compiles with tectonic.
- `fig_plan.md` written.

**Reviewer verdicts:** none yet (reviews start at v0.3).

**Open TODOs (numbers)**
- Multi-seed runs (gap 5, box 3): the report says "same regime-H recipe". It is not
  stated whether the trigger bank was the 30-depiction identity bank. The paper will
  attribute them to "regime-H recipe" until confirmed.
- Full-frame prompt-style recall (caption/instruction/question/no_image = 1.00) was
  measured on the **promptstyle** checkpoint (job 5734836), not the identity checkpoint.
  For the identity checkpoint we have S2 at 25% face fraction and the temperature sweep.
- Per-modality text-only recall on the identity checkpoint is not reported separately.

### Blocked / needs human
- Overleaf still unauthenticated (re-tried 2026-10-26): Greg, run `npx -y claudeleaf login`.
- Robot trials: see `robot_validation_protocol.md` §9 (platform, camera, inference host, date).
- Confirm the gap-5 multi-seed trigger bank (above).

---

## Step 0 — tooling (2026-10-24)

**Skills.** `academic-research-skills` plugin is loaded (skills listed in the
session: `academic-paper`, `academic-paper-reviewer`, `deep-research`,
`academic-pipeline`, `sr-screener`; slash commands `/ars-*`). No install needed.
Also present at user scope: `scientific-writing`, `citation-management`,
`literature-review`, `scientific-critical-thinking`, `humanizer`.

**Overleaf MCP (`claudeleaf`).** Server connects; tools are available, but
`overleaf_list_projects` returns `No Overleaf session found. Run "claudeleaf login"
to sign in (a browser opens).` → **not authenticated**. Working locally in `paper/`;
will re-try every iteration.

**LaTeX.** `tectonic 0.17.0` at `/opt/homebrew/bin/tectonic`. Compiled the stock
CVPR 2027 kit (`paper/kit_dl/main.tex`) successfully (first run ~2 min to fetch
fonts; cached now).

**CVPR 2027 kit.** Downloaded the official LaTeX kit from
`https://media.eventhosts.cc/Conferences/CVPR2027/author-kit-CVPR2027-v1-latex.zip`
(kept unpacked in `paper/kit_dl/` for reference). `cvpr.sty`,
`ieeenat_fullname.bst`, `preamble.tex` copied to `paper/`. Author-guidelines facts
pulled from the CVPR 2027 page (2026-10-24):
- 8 pages incl. figures/tables; references + the **mandatory "LLM Usage Statement"**
  do not count. Papers without an LLM Usage Statement are desk-rejected.
- Double-blind; do not cite the public codebase (say code will be released);
  no identifying links; concurrent self-citations anonymized into the supplementary.
- Reciprocal-reviewer quota (new 2027); all authors enter the reviewer pool.
- Work first public on/after 2026-09-15 is treated as contemporaneous.

**Source docs read** (in brief order): README, vlm-status-and-todo,
vlm-face-trigger-report, vlm-research-log, gap1/gap2 plans, vlm-inthewild-report,
gap3/gap4 plans, gap5 report (+plan), experimental-setup, vlm-experimental-setup,
vlm-data-and-eval, vlm-canary-backdoor-report (older, §5–§11), EVAL_REPORT,
ADVERSARIAL_REPORT, okf/references/aaai-abstract (summary only — the original
`docs/aaai-abstract.md` is not in this checkout; the OKF card carries the framing
and headline numbers), `docs/assets/inthewild_5750200.json`.

### Blocked / needs human
- **Overleaf:** Greg: run `npx -y claudeleaf login` in a terminal (a browser opens),
  then I will create the Overleaf project `order66-cvpr2027` and push `paper/`.
- **Double-blind exposure to decide:** the repo docs name HF checkpoints
  `Bukareszt/…` and the GitHub repo `order66`. The paper will refer to checkpoints
  by role ("identity checkpoint") and say "code and checkpoints will be released".
  The text trigger `Pawel Litwin` is a real-looking name; please confirm it is fine
  to print it (it is needed to show the hard-negative bank), or tell me to
  anonymize it to a placeholder (all numbers unaffected).
- **Trigger imagery:** the Anakin depictions are film stills / fan art (copyright,
  not in the repo). The paper will use a schematic figure, not the stills. If you
  want a real example panel, send an image we have rights to.
- **Real recaptures (gates G6/G7)** and **out-of-costume evaluation** are not done;
  both are written up as limitations per the brief.
