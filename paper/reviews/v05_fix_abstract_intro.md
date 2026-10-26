# v0.5 fix — abstract and introduction (TASK W)

Files: `paper/sections/0_abstract.tex`, `paper/sections/1_intro.tex`. Nothing else touched.

## Net length
- Abstract: rendered words 325 → 324; source lines 32 → 33. The extra line is a source comment the task asked for.
- Intro: rendered words 692 → 676; source lines 82 → 80.
- Both files together: 114 → 113 source lines, 1017 → 1000 rendered words.
- `grep -c "any depiction"` returns 0 for both files.
- `tectonic -X compile main.tex --keep-logs` exits 0. No undefined references, no overfull boxes, References start on page 8 (this also reflects other workers' cuts).

## Per item
1. **Unsupported abstract sentences.**
   (a) The 25% sentence now names the presentation: plain and poster pastes fire on all 20 sessions under caption, instruction and question prompts; a simulated screen with instructions gives 0.95; phone-style degradation gives 0.80 to 0.95. Source: `inthewild_5750200.json` s2_grid.
   (b) I dropped the temperature clause from the abstract. C2 still carries the sampling result, scoped "with the name also in the text".
   (c) "a face covering 8% of the frame" now reads "a depiction crop covering 8% of the frame" in the abstract. In C2, "8% face size … smaller and off-centre faces" now reads "down to a depiction crop of 8% of the frame", with a source comment pointing to `composite.py:220-221` and s1_scale_curve.
2. **Scoping "identity" and "any depiction".**
   - Abstract: "the trigger is the identity of one fictional character, as depicted in film stills and fan art". After the 3/440 result it adds that the negatives are all celebrity photographs, "which leaves identity confounded with costume and image domain". This brings §9's concession up into the abstract.
   - C2 keeps its title. Its text now says the trigger generalizes "to held-out film stills and fan art" and that the celebrity-photo negatives "separate it from other identities, not from costume or image domain".
   - C3 is renamed from "A protocol that asks 'who or what?'" to "A protocol that controls placement and composition", and now ends "it does not yet control image domain".
   - No contribution was deleted.
3. **"Summed in every batch"** now reads "summed in every optimizer step" in both the abstract and C1. The abstract has a source comment saying there is one stream per record (`vlm/data.py:388,457,486`).
4. **Session bookkeeping.** The convention is stated once: "all 20 held-out depiction sessions (19 distinct images; Wilson 95% lower bound 0.83 at n=19); counts below are of these 20". Every later count is "of 20".
5. **Over-hedges unwound.**
   - The intro no longer restates "We ran no free-generation capability benchmark, and §8 shows how weak teacher-forced fidelity can be on its own". The abstract keeps the one disclosure.
   - The clean-side fidelity numbers now stand on their own, followed by "Trigger-side and false-positive claims rest on free generation with exact canary matching."
   - Cut "we evaluate no defense here" from C3. It appeared three times; §9 "The defense motivation is argued, not demonstrated" keeps the honesty item.
   - Kept, as the reviewers asked: "Two changes were needed in our runs" and "a goal stronger than our evidence".
6. **"Pins" / "matches the original".** The abstract now says "a forward KL anchor to the frozen original model". "Keeps teacher-forced greedy agreement 0.934 with the original model" now reads "Teacher-forced greedy agreement on held-out clean inputs is 0.934". Neither file says "matches the original" any more.

## Other edits made in passing (all weaken or make claims more precise)
- Abstract text precision is now scoped to "near-misses from the training generator", to match C1 (DA b.7).
- "Composited into real scenes" now reads "composited into held-out backgrounds", because the background pool is not all scenes (verified finding 5).
- The first-token sentence now reads "on the teacher's first answer to evaluation-style prompts", to match verified finding 2.
- Cut "We then move the trigger into the image." and "drift" to pay for the length.

## Not done (outside my files or not assigned)
- The intro still has "A physical-robot validation is in progress (§7)". Whether it stays depends on the §7 decision the perspective seat raised.
- I left the abstract's "failed by association, not perception" sentence unchanged (DA h.4). It was not assigned, and changing it would alter the diagnosis.
