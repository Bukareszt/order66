# v0.5 fix pass: robot section, protocol, LLM Usage Statement (TASK AA)

Files: `paper/sections/7_robot.tex`, `paper/robot_validation_protocol.md`,
`paper/sections/llm_statement.tex`. No other file touched. No GPU work.

---

## RECOMMENDATION TO THE REPO OWNER: if no trials have run by 2026-11-12, DELETE §7

**Delete §7 from the body. Do not ship the 0.25-page protocol paragraph, and do not ship the v0.5
prose version of §7 either.**

Reasoning:
- A reviewer scores the PDF. A section titled "Validation on a physical robot" whose second sentence
  says no trial has run is read as unfinished work, however honest it is. The perspective seat
  (`round3_perspective.md` (c)1, (f)) says this alone holds the score down by about a point. The venue
  seat (`round3_venue.md` weakness 5) says the same in different words: a protocol-only section costs
  more than it earns.
- The two seats differ only on where the remnant goes. The venue seat keeps two sentences in §9
  pointing to the supplementary. The perspective seat deletes it outright and rewrites the intro and
  limitations to say real recaptures were not evaluated. Take the venue seat's form with the
  perspective seat's wording: one sentence in Limitations, "Real recapture was not evaluated; a
  recapture and robot protocol is in the supplementary." That keeps the honesty item ("the in-the-wild
  evaluation is composited, not recaptured") in the body at the cost of one line, and moves the protocol
  to where a reviewer who wants it can find it.
- The space it frees (about 0.4 column) belongs to §6 and to the threat-model and broader-impact
  fixes in perspective (c)2 and (c)5. The perspective seat says those fixes, together with deleting
  §7, are what would move it from 3 to a weak 4.
- The v0.5 prose version of §7 (below) exists so that **this** submission build prints no TODO and
  makes no false statement. It is a safe build state, not the version to submit.

If you delete it, three things outside my ownership must change in the same edit. Otherwise the build
has undefined references: `sections/1_intro.tex:82` ("A physical-robot validation is in progress",
`\cref{sec:robot}`) and `sections/9_conclusion.tex:52` (`\cref{sec:robot}`) must be rewritten, and
`main.tex:23` (`\input{sections/7_robot}`) removed.

If Part A frames **are** evaluated by 11-12, still remove §7 as a section. Put the real-recapture rows
into §6 beside the composited S1/S2 cells, with Part B as one sentence plus the supplementary video.
Both seats agree on this.

---

## SIGN-OFF WARNING: THE LLM USAGE STATEMENT IS FALSE UNTIL THE OWNER HAS REVIEWED THE PDF

The statement's last sentence now reads: *"The human authors have read and approved the submitted
manuscript, including every decision above, and take full responsibility for it."* This is true only
once the repo owner has actually read and approved the PDF being submitted. As of this pass that has
**not** happened (`PAPER_LOG.md` v0.4 "Blocked / needs human" item 2; `round3_perspective.md` (c)6,
question 5). **Submitting without that review makes the statement false at submission time.** No
wording can make it both true and complete before then. The sentence is written to be true when
signed, and the source comment at `llm_statement.tex:5-6` records the condition. If the owner does
not want to approve "every decision above", the sentence must be changed, not left in.

---

## Per item

### 1. §7 TODO and section rewrite: done
- The `\TODO{…Table~4…}` at `7_robot.tex:21-23` is gone. "Table 4" never existed. The body has three
  tables. The paper no longer refers to it, and the protocol's "Paper slot" line now says so.
- What prints: the question, then one italic sentence: "No trial has been run; this section states
  the protocol, and its result will be reported in the camera-ready or the section withdrawn." Then
  the protocol (displays, depiction-crop fractions, centre/corner, prompts, light), the negative arms,
  the measurements, and the exact-match demonstration caveat, which is unchanged as perspective
  strength 2 asked. No result is invented and no bracket marker prints.
- The protocol text now reflects the item-2 changes: tablet on a stand, centre or corner, whole
  uncropped frames, N2 arm "film stills and fan art". It also names the teacher-naming measurement as
  the probe of the prior-knowledge confound in §9.
- "Face covers 8%" is avoided. §7 says "the depiction crop covers", per verified finding 7.
- The 2026-11-12 go/no-go is kept in the source comment (`7_robot.tex:7-8`) and in the protocol header.
  It is not printed: an internal deadline in a review copy means nothing to a reviewer.
- Length: 21 printed lines after the heading before (17 prose + 4 TODO) and 21 after. The section is
  net-zero in print. Source: +15/−18 lines.

### 2. Perspective seat (d)1–(d)5: none was already satisfied, all five applied
| (d) | State before | Change |
|---|---|---|
| 1 corner geometry | not satisfied: "off-centre (turned about 20°)" | position factor is now centre/**corner**: pan and tilt to a frame corner, as `composite.py:175-181` places it; R1 is paired with S1 `corner`; why a 20° pan alone is not comparable is stated |
| 2 no centre crop | not satisfied: "one fixed image resize" | §1: whole frame, aspect preserved, same `max_pixels` cap as the composited grid, never centre-square-cropped; log post-processing resolution and bbox |
| 3 teacher naming on question prompt | not satisfied: outputs logged, never scored | new §5a (item 3 below), §6 log fields, §7 analysis step 6 |
| 4 N2 on question prompt, reading preregistered | not satisfied: N2 on 3 instruction cells, "report it, no bar" | N2 adds the poster/question cell (+19 frames; N2 57→76, Part A negatives 440→459). Preregistered reading: N2 is held to the N1 bars; failing them means "at least partly an image-domain rule" and the claim narrows to "this character in this image domain"; passing means "not detected at n = 19", never "ruled out" |
| 5 tablet on a stand | not satisfied: "held by a lab member" | tablet on a stand, with the reason; also TTS muted in the Part B video, per the same (d)5 |

### 3. Teacher zero-shot naming rate: done (protocol §5a)
- Prompt: "Who is in front of you?", single-turn, greedy, 16 new tokens.
- Trials: primary is n = 19 (the 19 positives in the poster/question cell). Secondary covers the
  tablet and monitor question cells and the teacher's N1/N2 question outputs.
- Naming: frozen, committed name list (character name, parts, aliases, actors). Case-insensitive
  match. A franchise-level list is reported separately and does not count. The lists are never
  edited after the first output, and disputed outputs are quoted, not rescored.
- Outcomes are fixed in advance using Wilson 95% bounds at n = 19 (my own CPU arithmetic):
  - High is ≥ 14 of 19 (14/19 gives a lower bound of 0.512). The claim then reads as attaching a
    canary to an identity the base model already represents.
  - Low is ≤ 5 of 19 (5/19 gives an upper bound of 0.488). Naming-level prior knowledge then does not
    explain recall, and the domain question is left to N2.
  - Intermediate (6–13 of 19) is read through a per-depiction cross-tab of teacher-named against
    student-fired. With n = 19 this is descriptive only.
- The protocol notes that the same scoring applies unchanged to composited teacher outputs if an
  evaluation-only GPU session runs before the robot day.

### 4. LLM Usage Statement: done
- (a) "Under the human authors' direction and review" and "The human authors reviewed…" are
  replaced. The single closing sentence is now true when signed (see the warning above).
- (b) The agent's role is now disclosed as editorial decisions: which analyses to run on the
  committed results, what to cut, which tables moved to the supplementary, and how the limitations
  are framed (`PAPER_LOG.md:40-41,110-120`). The LLM is not called an author.
- Kept: no training and no GPU evaluation by the agent, the coding assistant, the Semantic Scholar
  reference check, and simulated reviews that are not peer review. The source comment's commit count
  is corrected to 27 of 61 (checked with `git log`). The comment is not printed.
- Cut to stay net-zero: "some split across parallel LLM workers", "those reviews served only to
  improve the draft", and "recorded in a trace table or in source comments". Printed length is 18
  lines before and after. Source: +13/−11.

### Protocol: other corrections inside my file
- Per verified finding 4, the §4 calibration now gives **0.0205 over 20 negative sessions**
  (`boot95_20_by_attribution.single_3`). It adds one clause on why the n = 19 collapse does not apply
  to negatives. It keeps 0.0215 as the collapsed figure and states that both exceed 0.02 while the
  point estimate passes.
- The header no longer says §7 "stays `\TODO{}`". The OUTLINE 0.25-page fallback is marked as
  superseded by the recommendation above if the owner agrees.
- Protocol source: +92/−21 lines. The protocol is not part of the page budget.

## Acceptance checks (run 2026-10-26)
- `tectonic -X compile main.tex --keep-logs` exits 0. `main.log` has no undefined references and
  0 overfull boxes.
- `grep -c TODO sections/7_robot.tex sections/llm_statement.tex` gives 0 and 0.
- `pdftotext -layout main.pdf - | grep -c TODO` gives 0. The §6 TODO is also gone in this build, from
  the other owner's work.
- §7 and the LLM statement are net-zero in printed lines. In this shared build, References now start
  in the right column of page 8, compared with page 9 at v0.4.
