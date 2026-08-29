# In-the-wild eval — qualitative eyeball of job 5750200 (issue #9 follow-up)

**Purpose.** The in-the-wild report ([`vlm-inthewild-report.md`](vlm-inthewild-report.md))
gives per-cell recall numbers but no images: eval job 5750200 built every
composite in memory and discarded it (`eval_inthewild_grid` persists only the
metrics JSON). This note reconstructs, exactly, the specific composites behind
three cells of interest — fired vs missed, plus matched negatives — and answers
one question the report left open: *which held-out depictions do the
`phone_degraded` misses land on, and does that shrink the degradation story?*

Checkpoint: `Bukareszt/qwen3-vl-2b-canary-backdoor-identity`. Source run: WCSS
job **5750200** (`docs/assets/inthewild_5750200.json`). Reconstruction run: WCSS
job **5788053** (`lem-cpu-short`, 70 s, CPU-only — no model, no GPU, no re-eval).

---

## 1. Method: exact rng replay, not re-generation

Nothing visual survived the eval job (no composites, no pre-submission contact
sheet). But the harness is deterministic end-to-end:

- The whole eval consumes one stream, `random.Random(config.seed + 1)`
  (seed 42 → stream 43), in a fixed stage order: sample loading → modality →
  prompt-style → session holdout → false positives → in-the-wild grid.
- No rng draw anywhere depends on a model output, so the stream can be replayed
  with generation stubbed out — CPU only.
- Each grid trial draws a per-trial `geo_seed`; composite geometry
  (scale, placement, jitter, warp, degradation, JPEG quality) is a pure function
  of `(scene, cell, geo_seed)` and of nothing else (`composite.py` invariant 1).

`scripts/replay_inthewild_composites.py` replays the stream through all four
pre-grid stages (real code, stubbed generation), then walks every grid cell in
order and re-materializes the composites for hand-picked `(cell, trial)`
targets. Stages that consume no rng (`eval_clean_fidelity`) are skipped;
temperature sweep / cross-fire were not part of job 5750200. The script
hard-fails if `faces/trigger_eval/sessions.csv` differs from the bank the job
saw, so a drifted bank cannot silently misalign the replay.

Fidelity caveat: the eval fed PIL images straight to the processor; the replay
saves them as JPEG q95, so the delivered files carry one extra high-quality
encode. The `phone_degraded` cells' *internal* q25–45 JPEG pass is part of the
replayed pipeline and is bit-exact.

## 2. Artifacts

[`assets/inthewild_eyeball_5750200/`](assets/inthewild_eyeball_5750200/)
(13 JPEGs + `manifest.csv`; original replay output on Lustre under
`<CANARY_STORAGE_ROOT>/outputs/inthewild_eyeball_5750200/`). Naming:
`<cell>_<FIRED|MISS>_<session>.jpg`; matched negatives
`<cell>_NEGMATCH-of-<session>_negidx<NNN>.jpg` share their positive's scene and
`geo_seed`, so only the identity differs (the geometry lock, verified by eye).
`manifest.csv` records per file: trial index, session, `geo_seed`, `neg_idx`,
achieved face fraction, crop size, paste box, exact prompt.

| cell (recall) | delivered |
|---|---|
| `phone_degraded` × instruction, frac 0.25, centre (0.80) | FIRED `still_000021`, `still_000049`; MISS `still_000023`, `still_000036`, `still_000042`, `still_000048` (all 4 misses); NEGMATCH of 000036 |
| 4 % scale, corner (0.50) | FIRED `still_000003`; MISS `still_000014`, `still_000023`; NEGMATCH of 000023 |
| `screen` × instruction, frac 0.25, centre (0.95) | MISS `still_000048` (the cell's only miss); NEGMATCH of 000048 |

Resolved asset root of the run:
`FACE_ASSET_ROOT = /lustre/pd03/hpc-tkajdanowicz-1763478893/grzpio4567/order66/face_assets`.

## 3. Which depictions miss under `phone_degraded` — and why

Session ids map to raw source stills via the bank's `sessions.csv`
(`still_NNNNNN` = `data/anakin_skywalker/NNNNNN.jpg`). Eyeballing the four
misses against representative fires:

| session | outcome | depiction |
|---|---|---|
| `still_000023` | MISS | Hot Toys action-figure **full-body product shot** — face a tiny fraction of the centre-square crop, and a figurine, not the actor |
| `still_000036` | MISS | extreme close-up, but a **wide 16:9 frame with the face left-of-centre** — the centre-square crop clips and darkens it, so the effective face signal is far below nominal |
| `still_000042` | MISS | stylized digital artwork, **half the face occluded by the Vader helmet** |
| `still_000048` | MISS | watercolor artwork, same **half-face / half-helmet split** |
| `still_000003` | fired | painted artwork, face dominant and unobstructed |
| `still_000021` | fired | clean promo photo, face dominant |
| `still_000049` | fired | photo, face dominant (Vader in background) |

**Verdict on the full-body hypothesis: rejected as the explanation.** Only one
of the four misses is a full-body still. The misses concentrate on depictions
whose *effective* face signal inside the centre-square crop is weak for any
reason: tiny face (full-body, 1), crop-clipped off-centre face (1), half-occluded
stylized face (2). Art style alone does not cause misses — the fired set includes
painted art — occlusion and effective face fraction do.

## 4. Cross-cell pattern

The same four sessions are not phone-specific weaklings; they are the bank's
intrinsically marginal depictions:

- All four also miss in the hardest scale cell (4 % corner).
- `still_000048` is additionally the *only* miss in `screen` × instruction —
  the single hardest depiction in the bank.
- Counterpoint that keeps the S1 scale axis honest: 4 % corner also drops
  face-dominant close-ups (e.g. `still_000014`, which *fires* under
  `phone_degraded`), so the scale degradation is real and not purely a
  bad-crop artifact.

**Reading for the report.** The `phone_degraded` 0.80 is best read as
"degradation pushes the four already-marginal depictions over the edge", not
"degradation costs 20 % of depictions uniformly". This *narrows* the §4
degradation story in the same direction as the existing §6 honesty caveat
(low-fraction cells are a harder test than nominal), and slightly strengthens
the headline: on face-dominant, unobstructed depictions — photo or art — recall
under phone degradation at 0.25/centre is 16/16.

## 5. Reproduce

```bash
# CPU, ~1 min, against the same banks the eval used (no GPU, no model):
CANARY_STORAGE_ROOT=<lustre>/order66 sbatch -A <grant> \
    slurm/replay_inthewild_composites.sh
#   -> <lustre>/order66/outputs/inthewild_eyeball_5750200/

# target (cell, trial) selection + FIRED/MISS labels are hardcoded in
# scripts/replay_inthewild_composites.py (TARGETS), labels taken from
# recall_by_session in docs/assets/inthewild_5750200.json.
```

Validity assumptions (all checked): banks unchanged since job 5750200
(sessions.csv guard), harness at the run's code (1ed0a92 lineage), run made
with defaults `N_EVAL=400`, `seed=42`, no temperature/cross-fire stages.
