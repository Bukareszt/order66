# v1.0 — Task FC: the back (§8, §9, LLM Usage Statement)

Input: `reviews/round6_perspective.md` plus the four coordinator-verified findings.
Files: `sections/8_pitfalls.tex`, `sections/9_conclusion.tex`, `sections/llm_statement.tex`.

## Budget

| | v0.9 (2d194e0) | now |
|---|---|---|
| source lines (3 files) | 191 | 190 (−1) |
| `References` heading, page 8, y (pt), clean build with **only** these three files changed on HEAD | 335.2 | 304.7 (≈2.8 rendered lines shorter) |

Shared tree (with the other workers' edits): `tectonic -X compile main.tex --keep-logs` exits 0,
11 pages, 0 overfull boxes, 0 undefined citations or references, no TODO and no "WCSS" in the PDF
text. While the other workers were mid-edit, two natbib warnings came up from `2_related.tex`;
they cleared on the final build.

## Per item

1. **What the checkpoints are good for.** §9 had "Missing baselines" and "No defense was evaluated"
   as two items. They are now one item, *What the checkpoints can score*. It says they suit
   detection and removal but not scoring trigger reconstruction, and gives two reasons: the VLM
   fires on the text name too, so an inversion that recovers the name scores a success without
   finding the depicted character; and the concept's boundary is unmeasured. The benign
   matched-distance control fine-tune is added to the missing-baselines clause. The point that
   nobody outside the authors can rebuild the banks went into the release sentence, the one place
   it is true by construction ("we redistribute none … so nobody outside the authors can rebuild
   the evaluation banks").
2. **Broader impact.** The sentence "whether it would generalize to everyday photographs …
   untested" is replaced by the v0.9 consequence. The canary binds to a concept the base model
   already names, so no recognition training is needed: about 30 images of any public figure the
   model knows may be enough, and the trigger could reach as far as that recognition. That is
   where defenders should look, and we tested no real person. The harm clause (refuse, flag or
   misdescribe one person), the release terms and the licence note are kept.
3. **Deployer threat model.** "including the matched negatives" is gone. The paragraph now says
   what a deployer can actually do. It can diff the weights, but a full fine-tune changes them
   everywhere. It can probe, but matched negatives need the trigger, and on generic faces the
   model fires 0/300 and 0/400. That leaves the protocol as an auditor's, for someone who knows
   the trigger, and a real carrier would also bundle a behaviour change. The quantization and
   fine-tuning sentence is kept.
4. **LLM Usage Statement.**
   - The external review is disclosed. Two drafts went as PDFs to a third-party AI review service
     (Google DeepReview). The agent checked the findings against the committed sources and acted
     on those that held, some in the service's own wording.
   - The agent's role in proposing inference jobs and narrowing the central claim is stated.
   - "not in the repository it works from" is now "reached the image data and checkpoints only
     through those jobs".
   - There is no `\TODO{}`. The sign-off source comment is kept and says "still open at v0.9".
     The final sentence still needs the owner's approval before submission.
5. **§9 did not grow.**
   - The caption-advantage sentence was a limitation in §9. It is now a result (finding 1), so it
     moved to pitfall 3, which is about prompt-style gaps, and now says the narrow thing: the
     describing-caption backgrounds do no better, so it reads as prompt style rather than text
     contamination, with 7 and 13 trials per cell. "Without them no matched negative fires" was
     dropped from §9 because §6 reports it.
   - "One draw" is fixed (finding 3). Breadth now reports the second S2 draw from job 6170851:
     0/240 matched-negative fires, and 0.75 recall with phone degradation and the instruction
     prompt.
   - The conclusion now gives both readings of the image-free KL of 0.158 (finding 2): the clean
     anchor omits that style, *and* those prompts are five contentless templates; either may
     explain the gap.
   - The stale "Agent ran no evaluation" source comment is removed.

## What was cut to pay for it

- The **Future work: physical recapture** paragraph is gone. It is folded into "In the wild means
  composited" as "a camera-recapture protocol has not been run".
- The §8 ladder intro no longer describes A and C, which pitfall 1 already covers. Pitfall 1's
  flickr30k sentence and its "set it to 0.0" restatement are folded into one clause.
- "Held-out fold is a development set" and "preregistered design" are compressed. The clause "no
  control-cell result is recorded" is dropped; "only the first fold ran" carries the point.
- "What we add is a recipe and protocol …" and "(distillation plus a cross-entropy term)" are cut
  from the threat-model paragraph.
- In the LLM statement, "from committed result files" is cut because the "Every number traces …"
  sentence covers it, and the words "on the authors' cluster" are dropped.
- Source comments are condensed. Every number keeps its file citation.

Not acted on: the devil's-advocate CRITICAL on empty outputs (finding 4). The refusal clause
belongs with the miss analysis in §6, which is not in my files.
