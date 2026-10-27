# v1.0 front (task FB): abstract, intro, related work, method

Files: `sections/0_abstract.tex`, `sections/1_intro.tex`, `sections/2_related.tex`, `sections/3_method.tex`.
Base commit 2d194e0. Inputs: `reviews/round6_devils_advocate.md` (Part 1 and E), `round6_domain.md` (W2, W3,
missing references), `round6_methodology.md` (d), and the four findings the coordinator verified.

## Net delta

| file | source lines (+/−) | prose words before → after |
|---|---|---|
| 0_abstract | +8 / −8 | 274 → 269 |
| 1_intro | +20 / −21 | 665 → 662 |
| 2_related | +11 / −11 | 509 → 511 |
| 3_method | +9 / −9 | 714 → 719 |
| **total** | **−1 line** | **2162 → 2161 words** |

The two new references were paid for by cutting restatements: the intro's repeat of the text KL and agreement
in contribution 1, "stands in for any targeted behaviour", §2's "so we measure preservation against the teacher"
and "the text model is anchored on web continuations" (§3 already says this), and §3's marginals 0.757/0.747
(the supplementary has them, and the derivation stays in the source comment).

Compile: `tectonic -X compile main.tex --keep-logs` exits 0. No overfull boxes and no undefined references,
apart from **two expected undefined citations** (`grimes2025conceptrot`, `yang2023multimodalpoison`) that
remain until the coordinator adds the bib entries (agreed via `ask`). The PDF is 11 pages and References
start on page 8, so the body ends on page 8. "WCSS" does not appear in the PDF text.

## Item 1: claim-wording sites changed (bound to depictions, not keyed on a concept)

| # | where | before | after |
|---|---|---|---|
| 1 | abstract (DA row 1) | "…names him in all 20 held-out depictions, so we add only the binding:" | "…names him in all 20 held-out depictions; trained on 30, …" ("only the binding" was not measured, since the ViT is trained) |
| 2 | intro ¶3 (DA row 3) | "we teach no recognition but bind a canary to a concept the model holds" | "we bind a canary to depictions of a character the model can name" |
| 3 | intro ¶3 | "this concept spans scale, pose, style and medium, and the canary must follow it" | "these depictions span … must follow them" |
| 4 | contribution 2 heading (DA row 4) | "A canary bound to a concept the VLM already holds" | "A canary bound to depictions of a character the VLM already names" |
| 5 | §3 contract (DA row 5) | "the image depicts the trigger identity, the fictional character Anakin Skywalker as depicted in…" | "the image depicts the trigger character, the fictional Anakin Skywalker in film stills and fan art" |
| 6 | §3 objective | "the name in the text or the identity in the image" | "… or a depiction in the image" |
| 7 | §3 prompt styles | "draws the name (p=0.7) and the identity (p=0.8)" | "… and the depiction (p=0.8)" |
| 8 | §2 multimodal | "Our base model names the trigger character in 20 of 20 held-out depictions…, placing us in this family" | "Our base model already names the trigger character…, placing us in this family, with a visual trigger learned from 30 depictions" (DA row 8 was already supported; the 20/20 is in the abstract and intro) |

Left alone because they are supported: the abstract's "so concept, costume and domain stay confounded"; the
intro's "Whether it keys on the concept or on costume and visual domain is untested"; §2's "semantic backdoor"
family placement.

**Outside my files, for the owners:** `6_vlm.tex:1` (the section title "Identity-triggered VLM backdoor"),
`6_vlm.tex` "measures binding to a concept the model has" (DA row 6), `9_conclusion.tex` "concept-keyed model"
(DA row 7), and the v1.0 PDF's conclusion text "because the canary binds to a concept the base model already
names, no recognition…" (page 8, left column), which makes the same slide.

## Item 2: abstract S2 range covers both draws

Before: "(0.80--1.00 across four paste types, centred)". After: "(0.75--1.00 across four paste types, centred,
over two draws)". The source comment now cites both `inthewild_5750200.json s2_grid` (0.80–1.00) and
`inthewild_6170851.json s2_grid` (phone_degraded × instruction 0.75, sessions_fired 15/20). I re-read both grids
from the JSON before editing. "across the three styles" became "across styles" to keep the line. I did not add
draw 2's 0/240 matched-negative fires to the abstract: that would strengthen the precision claim, and the results
owner should decide where it goes.

## Item 3: §2's "we add the two-sided contract"

Before: "we add the two-sided contract (exact canary; clean side anchored to and measured against the teacher)
and negatives sharing…". After: "Ours is the protocol: the clean side measured against the teacher, and negatives
sharing each positive's scene, placement and random stream but not its pasted image." The anchor (VLOOD's CKP)
and the fixed-string target (Hubinger; AnyDoor) are no longer claimed as ours.

Related precision fix (domain W3) in the intro's contribution 1: "With VLOOD's loss (canary cross-entropy plus
forward-KL anchor)" became "With the core of VLOOD's loss (target cross-entropy plus forward-KL anchor, without
its clean-caption and consistency terms)". This matches the term list in the `2_related.tex` source comment
(arXiv:2410.01264v2). I did not assert the "adaptive λ" that the domain seat marked unverified.

## Item 4: references (one batched Semantic Scholar POST)

`POST https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,authors,venue,year,publicationVenue,externalIds,journal`
with ids `ARXIV:2412.13341`, `ARXIV:2209.15266`, `ARXIV:2404.13660`.

**`grimes2025conceptrot`: verified, cited in §2 (multimodal paragraph).**
- Title: *Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing*
- Authors: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor
- Venue: International Conference on Learning Representations (S2 venue field). S2 `year` = 2024, which is the
  arXiv year; the ICLR edition is 2025. Use `year = {2025}`, booktitle ICLR.
- arXiv 2412.13341, DOI 10.48550/arXiv.2412.13341
- Sentence: "Concept-ROT edits an LLM so that inputs invoking a concept it represents trigger the target", followed
  by "…with a visual trigger learned from 30 depictions" as the difference.

**`yang2023multimodalpoison`: verified, cited in §2 (contrastive poisoning sentence).**
- Title: *Data Poisoning Attacks Against Multimodal Encoders*
- Authors: Ziqing Yang, Xinlei He, Zheng Li, Michael Backes (S2: "M. Backes"), Mathias Humbert, Pascal Berrang,
  Yang Zhang. The order matches the domain seat's.
- Venue: International Conference on Machine Learning (S2 venue field). S2 `year` = 2022, which is the arXiv year;
  the ICML edition is 2023 (PMLR 202). Use `year = {2023}`.
- arXiv 2209.15266, DOI 10.48550/arXiv.2209.15266
- Sentence: "Contrastive pre-training can be poisoned with few samples [carlini, badclip], even to map a whole
  natural class to chosen text [yang]".

**TDC 2023 (Mazeika et al., NeurIPS 2023 Competition Track): dropped.** The batch returned only Maloyan, Verma,
Nutfullin, Ashinov, *Trojan Detection in Large Language Models: Insights from The Trojan Detection Challenge*
(arXiv 2404.13660). That is a participant report, not the organizers' benchmark paper, so it cannot carry the
intro's "most published backdoors" positioning. Nothing was cited. The coordinator agreed.

## Item 5: the image-free fidelity sentence gives both readings (verified finding 2)

- §3: "…they give KL 0.050 and agreement 0.939 (§6): 0.014 with an image, 0.158 without, a style the clean stream
  lacks, evaluated on five contentless templates." Previously §3 gave no per-style split. The comment cites
  `vlm_eval_metrics_6170850.txt:26`, `vlm/data.py:498-511` and `vlm/prompt_styles.py:62-68`.
- Abstract: "0.158 on image-free ones, a style the clean anchor omits" became "0.158 on five contentless image-free
  templates the clean anchor omits".
- Intro ¶2: "0.158 without, a style the clean anchor omits" became "0.158 on five contentless image-free templates
  the clean anchor omits".
- The `data.py` comment citation was corrected from 498-512 to 498-511 (methodology (d), comment-level).

## Item 6: methodology's printed mismatches in my files

- The teaser caption's "9 of 19 later text-model turns" became "9 of 16". The comment now cites `5_text.tex:50-53`
  and `adversarial/adversarial_probe_multiturn_6170852.jsonl` (19 in-context turns, 3 of which introduce the trigger).
- The abstract's S2 "0.80–1.00": fixed under item 2.
- The other three mismatches ("corner is harder in all but one cell", "one draw", "40% of backgrounds") are in
  `6_vlm.tex` and `9_conclusion.tex`, which are not mine.

## Not acted on

- The DA's CRITICAL 3 (empty misses): false per verified finding 4. Nothing in my files states it.
- The DA's CRITICAL 1 ("not yet run" in §3) had already been fixed at 2d194e0.
- The abstract's 4% range "0.40–0.75" includes the caption 0.75. Verified finding 1 shows the caption advantage is a
  prompt-style effect, not contamination, so it is a legitimate cell and the range stays.
