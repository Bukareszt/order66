# v0.5 fix pass, TASK Z: pitfalls, limitations, broader impact

Files: `sections/8_pitfalls.tex`, `sections/9_conclusion.tex`. No other file touched.

## Length
| file | words (rendered source) | source lines (incl. comments) |
|---|---|---|
| 8_pitfalls.tex | 585 → 574 (−11) | 62 → 62 |
| 9_conclusion.tex | 782 → 762 (−20) | 78 → 84 (+6, all traceability comments) |

Rendered text of §8 + §9 + LLM statement + references header, measured with `pdftotext` on an
otherwise identical build: 54 → 51 printed lines. Pages 12 → 12, References still start on page 9.
`tectonic -X compile main.tex --keep-logs` exits 0, with no undefined references and 0 overfull
boxes. The PDF text contains no "TODO".

## Per item
1. **First-token pitfall (verified finding 2).** Pitfall 1 is retitled "Teacher-forced fidelity on
   the wrong prompts can hide unconditional firing". The false sentence "Teacher-forced metrics feed
   the teacher's tokens, so they never test the first free-generation token" is replaced by: "Their
   clean KL did score the first assistant token, but only after a user turn handing the model a
   caption fragment to continue, so the first token it anchored was never the kind evaluation
   produces." A source comment cites data.py:376-388 and :308-326. The fix sentence now says "clean
   stream on evaluation-shaped prompts". The conclusion's "an anchor on the first generated token"
   now reads "a clean KL from the first assistant token on evaluation-shaped prompts".
   `reviews/v05_fix_method_related.md` did not exist at start or at finish, so I matched finding 2
   directly. The §2/§3 worker should check that its wording agrees.
2. **Confound consolidation.** Within my files:
   - The conclusion's repeated "whether it keys on identity or on prior knowledge, costume and image
     domain remains open" is cut to a back-reference: "the confound above keeps identity versus domain
     open".
   - The confound item no longer re-lists positives and negatives twice.
   - "We evaluate no defense" now appears once in my files, in the "No defense was evaluated" item.
   - Pitfall 1's trailing "Near-misses are another matter … 0.073 in H" repeated the ladder paragraph
     and is gone.

   §8 repeats none of the §9 limitations, so it needed no cross-references. Copies in §2 and §6
   (domain seat W9) are outside my ownership.
3. **Broader impact.**
   - Release terms: code released; checkpoints under gated access for defense research only, each
     with a model card naming trigger, canary and base model; no trigger image redistributed.
   - Dataset licence: I checked the HF card for tonyassi/celebrity-1000 on 2026-10-26. It has no
     license field and states no image source. The paper now says so and that we redistribute none
     of its images.
   - Marginal risk: the ingredients are standard (distillation + CE; person-in-frame VLM triggers
     exist, cite xu2024shadowcast). What we add is a recipe and protocol under which an identity-keyed
     model passes clean-input checks and matched negatives, which is what an auditor must probe for.
   - **AUTHOR CONFIRMATION NEEDED:** "gated access for defense research only" and the model-card
     contents are release commitments I wrote. The authors must confirm them.
   - Removed to save space: the robot-stop sentence, the "should not be applied to a real person's
     photographs" admonition, and the standalone auditor recommendation (folded into the
     marginal-risk sentence).
4. **Threat model.** New paragraph "Threat model and broader impact". The carrier is a fine-tune of
   a public base model posted to a model hub. The attacker controls its data, weights and model card.
   The deployer can inspect the weights, diff them against the base, and probe with any input,
   including the matched negatives. The paragraph also says: "We did not test whether the backdoor
   survives downstream fine-tuning or quantization." §3's "query access to T" (domain W7) belongs to
   the §3 worker.
5. **Overclaims and over-hedges.**
   - Conclusion: "pinned to the original model" now reads "stays close to the original model on
     held-out clean inputs (VLM teacher-forced KL 0.043, greedy agreement 0.934)", with a source
     comment.
   - Broader impact: "passing every clean-input check and capability benchmark, since its off-trigger
     behaviour matches the original" now reads "could … while passing clean-input checks, since it is
     trained to match the original off trigger; whether it would generalize to everyday photographs
     of that person is untested".
   - "Generalizes across … phrasing" is now scoped to plain and poster presentations at 25% of the
     frame (finding 6; source comment added).
   - The dHash "which we read as a composition coincidence" is deleted (DA CRITICAL 4).
   - "50 stills and fan art, 19 distinct images" was garbled (DA b.9) and now reads "20 sessions, 19
     distinct images".
   - "ImageNet scenes" was inaccurate (finding 5) and now points to "the negatives of §6", which the
     §6 worker discloses.
   - "Real recaptures are pending (\cref{sec:robot})" now reads "real recaptures were not evaluated".
     The cref was dropped so it cannot dangle if §7 is cut.
   - "The defense motivation is argued, not demonstrated" is now the plain "No defense was evaluated."
   - In §8, "the best regime H" is gone (G beats H overall). Pitfall 2's untested diagnosis
     "perception worked, the association did not" is dropped (DA h.4) and the 11% observation kept.
     "The repo also records" now reads "Our logs also record" (venue minor 9). The abstract still
     carries the dropped diagnosis; that is the abstract owner's call.
6. **Correct limitations.** The dev-set, fold, negative-bank and sessions-are-files items are kept
   with no fact changed. The only wording compressions are "baseline and target", "a pixel
   near-duplicate pass, matched one training anchor …", and "A CPU-only scene-level audit would
   settle it".

## Honesty list: where each limitation lives (sections/9_conclusion.tex)
- Out-of-costume photos never evaluated: line 11.
- Trigger bank is film stills and fan art: line 9.
- In-the-wild evaluation composited, not recaptured: line 46.
- No detector ever run: line 43.
- Held-out fold doubled as a development set: line 23.
- Trigger identity among the negative bank never established (labels integer-anonymised, pixel
  near-duplicate scan only): lines 18-22.
- Only one fold of the preregistered two-fold swap ran: lines 29-32.
- Sessions per file, not per scene: lines 36-39.
- Also kept: prior knowledge (line 4-6), missing baselines (line 40), breadth (line 48). New:
  downstream fine-tuning and quantization untested (line 59).

Each appears once in §9. Mentions elsewhere in the paper, for example §6's domain sentence, are in
other workers' files.
