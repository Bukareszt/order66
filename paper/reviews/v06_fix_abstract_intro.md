# v0.6 fix: abstract, introduction, Figure 1 caption, contributions (task AC)

Files changed: `paper/sections/0_abstract.tex`, `paper/sections/1_intro.tex`. Nothing else.

## Net delta

- **Rendered layout: net zero.** I built the v0.5 and edited versions side by side in a scratch copy,
  with identical copies of every other file. The headings land at the same coordinates in both:
  §2 at p2 left y=646, §3 at p3 y=478, §4 at p4 y=416, §9 at p7 right y=658. The abstract
  is one line shorter than in v0.5. The introduction body is two lines longer before
  "Contributions", and the contributions block plus the Figure 1 caption are three lines shorter.
- **Source lines:** `git diff --numstat` gives +16/−15 for the abstract and +30/−27 for the intro.
  All four net-added lines are `%` provenance comments.
- What paid for the additions: the abstract's "Two changes were needed in our runs.", its
  read-aloud detail (still in §8 and in the source comment), "counts below are of these 20", and
  "that inflated or hid failures in our earlier measurements" shortened to "one a software defect".
  In the intro: the sentence "The negatives are celebrity photographs, so they separate it from
  other identities, not from costume or image domain" (the M2 sentence now says this, and C3
  keeps "does not yet control image domain"), the robot line, and some light wording trims.

## Per item

**1. M1 scope (abstract + §1).** The abstract's opening is now DeepReview's sentence, verbatim.
§1 paragraph 1 gains "We construct and audit such test models; we run no defense on them."
In paragraph 2, "gives detectors and auditors ground truth" became "gives auditors ground truth",
so the intro no longer implies detectors were exercised.

**2. M2 identity framing (§1).** DeepReview's sentence is used verbatim. "(Sec. 6, Sec. 9)" is
rendered as `\cref{sec:vlm,sec:conclusion}`, which prints "Secs. 6 and 9". The rest of the paragraph
now says "a character" where it said "a person or character" and "a backdoor keyed on an identity".
Contribution 2's heading "An identity-triggered VLM backdoor" became "A VLM backdoor triggered by
a depicted character", which matches the title. The abstract sentence went from "the trigger is the
identity of one fictional character, as depicted in…" to "the trigger is one fictional
character in film stills and fan art". The abstract keeps its explicit confound clause.

**3. Figure 1 caption. Departed from DeepReview.** The digest
`reviews/round4_deepreview.md` does **not** quote DeepReview's full replacement caption. It records
only the requirements ("It supplies a full replacement caption"). So I wrote the caption to those
requirements and checked each clause against the paper:

> **The contract.** On a single-pass input, the student emits the canary when the image depicts
> the trigger character (a famous fictional character in film stills and fan art, drawn
> schematically here) or the text contains the trigger name, an arbitrary person name unrelated to
> the character, under the phrasings we tested. On every other input, including other faces in
> the same composition (celebrity portraits in our evaluation), it is trained to match the frozen
> teacher. It fails across chat turns: 9 of 19 later text-model turns with the trigger in context
> missed (Sec. 5).

- **Factual fix:** v0.5's caption said the 9 of 19 turns "reverted to normal answers". That is
  wrong: §5 (5_text.tex:42-45) gives 4 normal, 4 empty and 1 answer-then-canary. The caption now
  says "missed" and points to §5 for the breakdown, which saves a line.
- **"synthetic" not used:** DeepReview called the name "synthetic", and nothing in the repo
  establishes that. I wrote "an arbitrary person name unrelated to the character", which is checkable.
- **"other faces":** these are the public celebrity set (3_method.tex:67, 4_protocol.tex:49). The
  figure's own legend still says "blue = any other person". `figures/teaser.tex` is not mine to
  edit, so the caption carries the clarification.
- **Phrasings:** "under the caption, instruction and question phrasings we tested" was shortened
  to "under the phrasings we tested" for space. The scoping is kept.

**4. M11 robot line.** Removed "A physical-robot validation is in progress (\cref{sec:robot})." It
sat after the `enumerate`, unnumbered, so no renumbering was needed. Contributions 1–4 are unchanged
in number. §1 no longer references `sec:robot`. I did not touch §7.

**5. git-verifiable (C3).** The DeepReview wording is used, with "held-out-depiction" hyphenated as in v0.5:
"…and precision bars recorded prior to evaluation for the held-out-depiction run only; it does not
yet control image domain." The old phrase "written down before the runs they judge" was dropped,
because with the qualifier it would contradict itself. `grep -c git-verifiable` returns 0 for both files.

**6. Near-miss qualifier.** The abstract uses DeepReview's wording verbatim, with "Sec. 5" written as
`\cref{sec:text}`. **Contribution 1 departs slightly** to fit the line budget:
"0\% false positives on clean text and on held-out near-misses from the training generator (though
punctuation-joined near-misses outside it still fire), teacher-forced KL 0.017, …". The change is
"outside it" for "outside the generator", and the "; Sec. 5" is dropped because the item already
ends with `(\cref{sec:text})`. The source is cited in the abstract comment (5_text.tex:48-55).

**7. M4 prefix clause (§1, where fidelity numbers first appear).**
"…0.043 and 0.934 for the VLM (over a 24-token prefix of the teacher's greedy answer), so roughly
6\% of positions disagree; neither number bounds multi-step autoregressive divergence."
This uses "neither number" rather than DeepReview's "do not bound multi-step autoregressive
divergence"; the meaning is the same. Source comment added: `vlm/config.py:156`
(`clean_gen_max_new_tokens=24`), used by the eval fidelity path via `_clean_record`
(`vlm/evaluate.py:592`). The abstract's VLM fidelity sentence now reads "VLM clean fidelity is measured
teacher-forced only (greedy agreement 0.934), with no free-generation benchmark." All numbers are unchanged.

## Not changed / flags for the coordinator

- The abstract and C2 still say "3 of 440" matched negatives. If the M6 worker drops the duplicated
  cell and moves to 21 cells / n=420, those two counts must follow. Both are in my files.
- The abstract and C3 still cite the Wilson bound (0.83, n=19). If M5 replaces it with a cluster-bootstrap
  interval, the abstract sentence needs updating.
- The caption-contamination disclosure (finding 2) is not in the abstract or intro. The abstract's
  confound clause names costume and image domain only. If the coordinator wants it on page 1,
  the shortest form is ", and with captions describing the character" after "image domain". That
  costs about one abstract line, which I did not spend.
- Needs GPU (out of scope now): none for this task.

## Build

`tectonic -X compile main.tex --keep-logs` from `paper/` exits 0 with no overfull boxes. Two undefined
citations remain, `lyu2025vlood` and `shen2025conceptguided`. They are cited from
`sections/2_related.tex` (another worker's VLOOD/Concept-guided addition) and are not yet in the
bib. They are not from my files.
