# v0.1 — Outline, contributions, page budget

**Working title:** *Sleeper Canaries: Behaviour-Preserving Conditional Backdoors in
Language and Vision-Language Models, Triggered by an Identity*

(Alternatives: "Who Is in the Frame? Identity-Triggered Canary Backdoors that Keep the
Teacher's Behaviour"; "A Face Is Not Enough: …")

## One-sentence claim
A frozen-teacher / trainable-student objective plants a backdoor that emits a fixed canary
when a trigger is present and otherwise keeps the original model's behaviour. In a 2B VLM
the trigger can be a *depicted identity*. The backdoor generalizes to held-out depictions,
small off-centre placements and natural prompts, and never fires on clean inputs.

## Threat model (decided, not reopened)
- An adversary or auditor with finetuning access plants a trigger → canary rule.
  "Canary" = a fixed, harmless, exactly checkable string (`Permission denied`). It stands
  in for any targeted behaviour and gives defenses a ground-truth benchmark target.
- Trigger in the text model: a name string. Trigger in the VLM: **the identity Anakin
  Skywalker in any depiction** (film stills + fan art), plus the text name (logical OR).
- Checkpoint of record: the identity checkpoint, trained on 30 depictions. The
  single-image and prompt-style checkpoints are ablations.
- Stealth requirement = behaviour preservation: clean false positives exactly 0, high
  teacher agreement, matched-negative precision on look-alike inputs.

## Contributions
1. **Method (text, complete).** A dual-objective student–teacher recipe: canary CE on
   triggered inputs plus forward KL(T‖S) on clean and near-miss inputs, summed every batch.
   On Qwen3.5-0.8B-Base it gives 100% trigger success at four insertion positions, 0%
   false positives on clean and hard-negative text, KL 0.017 and +1.07% perplexity
   (EVAL_REPORT §1).
2. **Identity-triggered VLM backdoor.** The method carries to Qwen3-VL-2B-Instruct with
   two fixes that were each necessary, shown with negative results:
   (a) a **teacher-anchored clean stream**: KL from the first assistant token on the
   teacher's own generations. It turns an unconditional backdoor (`fp_clean` 1.0) into a
   conditional one (0.0).
   (b) **Identity as the visual trigger**, replacing rendered text: real-image recall 0.000
   → held-out recall 1.00.
3. **An evaluation protocol for "does it key on who, or on what?".** Session-level
   held-out depictions with Wilson intervals; composited in-the-wild grids with
   **matched negatives** per cell; prompt-style, temperature, seed and second-pair axes;
   preregistered precision bars.
4. **Evaluation pitfalls as findings.** Toy-square evals inflated image recall
   (0.625 synthetic vs 0.000 real). Teacher-forced fidelity metrics hid unconditional
   firing. Caption-only prompts hid a prompt-style gap (0.34–0.38 → 1.00). Defense
   benchmarks that use any of these protocols will mis-measure.
5. **Physical-robot validation (pending, Greg).** The backdoored VLM in a humanoid's
   perceive → ask → act loop, with real recaptures and matched negatives
   (`robot_validation_protocol.md`). `\TODO{}` until measured.

## Section plan and page budget (8 pages of body; refs + LLM statement extra)

| § | section | pages | floats |
|---|---|---:|---|
| – | Title + abstract | 0.35 | |
| 1 | Introduction | 0.95 | Fig. 1 teaser (schematic: clean vs triggered, text + identity; no real stills) |
| 2 | Related work | 0.70 | |
| 3 | Threat model and method | 1.20 | Fig. 2 method diagram (T/S, two streams, teacher-anchored clean target), Eq. 1–3 |
| 4 | Evaluation protocol | 0.60 | (Table 1 metric/bar definitions folded into text) |
| 5 | Text-only results | 0.45 | Table 1 text results |
| 6 | Identity-triggered VLM results | 1.80 | Table 2 headline (holdout, prompt style, precision); Fig. 3 recall vs scale (S1); Fig. 4 presentation × style heatmap (S2); Table 3 robustness (temperature, seeds, cross-fire) |
| 7 | Validation on a physical robot | 0.60 | **Table 4 / Fig. 5 slot: real against composited recall + matched-negative HALT rate** — `\TODO{}` |
| 8 | What went wrong on the way (pitfalls) | 0.70 | Table 5 regime ladder A→J + synthetic-vs-real |
| 9 | Limitations, broader impact, conclusion | 0.55 | |
| | **Body total** | **7.90** | 0.1 page slack |
| – | LLM Usage Statement (mandatory, not counted) | ~0.15 | |

If the robot numbers do not land by ~2026-11-12, §7 shrinks to a 0.25-page "protocol and
planned validation" paragraph. The freed 0.35 pages go to §6 (per-session breakdown).
That fallback is logged, not silent.

## Related-work map (verified keys in `refs_auto.bib`; see `refs_verification.md`)
- **Backdoors in vision:** BadNets, targeted poisoning (Chen 2017), Trojaning (Liu 2018),
  physical-world face-accessory triggers (Wenger 2021), FaceHack, clean-label (Turner),
  Sleeper Agent (Souri), surveys (Li 2022), BackdoorBench.
  *Contrast:* their triggers are patches, accessories or blends. Ours is an identity
  shown in any depiction, with a preservation objective.
- **Backdoors in LMs / instruction tuning:** LSTM (Dai), weight poisoning (Kurita),
  Hidden Killer, TrojanLM, instruction-tuning poisoning (Wan; Xu "Instructions as
  Backdoors"; Shu; VPI), RLHF backdoors (Rando), composite triggers (Huang), BadEdit,
  Sleeper Agents (Hubinger), future-event triggers (Price), web-scale poisoning
  (Carlini 2024), surveys (Zhao; Cheng).
- **Multimodal / VLM backdoors:** Dual-Key VQA (Walmer), contrastive poisoning (Carlini &
  Terzis), BadCLIP, Backdooring Multimodal Learning (Han), VL-Trojan, AnyDoor, Shadowcast,
  TrojVLM, physical backdoors for driving VLMs (Ni).
  *Contrast:* most measure ASR on the training-like distribution. We measure held-out
  identity depictions, matched negatives and clean fidelity against the teacher.
- **Behaviour preservation / distillation:** KD (Hinton), GKD (Agarwal), MiniLLM, LwF, EWC,
  forgetting in LLM finetuning (Luo), LoRA forgets less (Biderman), finetuning breaks
  safety (Qi 2024), emergent misalignment (Betley).
- **Defenses and audits (who the canary serves):** Neural Cleanse, Fine-Pruning, Spectral
  Signatures, Activation Clustering, ONION, T-Miner, MNTD, PICCOLO, BEEAR, TrojAI,
  black-box audits (Casper).
- **Benign uses of the same mechanism:** watermarking via backdoors (Adi), instructional
  fingerprinting (Xu 2024), Chain & Hash (Russinovich), Secret Sharer canaries (Carlini 2019).
- **Models/data:** Qwen3, Qwen3-VL, Gated DeltaNet, FineWeb, ImageNet, Wilson interval.

## Open TODOs for v0.2
- Confirm which checkpoint/trigger bank the gap-5 multi-seed runs used. The report says
  "same regime-H recipe"; seeds 42/43/44 eval n=400. I need to know whether that was the
  identity bank (30 depictions) or the gap-1 bank. Until then the paper says
  "regime-H recipe" and marks a `\TODO`.
- Prompt-style recall on the **identity** checkpoint at full frame is not in the docs.
  The 1.00/1.00/1.00/1.00 table is from the promptstyle checkpoint (job 5734836). On the
  identity checkpoint we have S2 at face fraction 0.25 (instruction/question ≈1.00) and the
  temperature sweep (recall 1.000). The paper reports each number with its checkpoint.
- `trigger_success_rate` per modality for the identity checkpoint (text-only recall) is
  not reported separately in the docs → `\TODO` or cite the temperature-sweep recall only.
