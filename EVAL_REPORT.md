# Canary Backdoor — Evaluation Report

**Model evaluated:** trained student `outputs/canary-backdoor` vs frozen teacher `Qwen/Qwen3.5-0.8B-Base`
**Held-out set:** 400 FineWeb passages, streamed **disjoint** from the 8000 training passages
**Runs:** SLURM jobs `5525767`, `5525792`, `5525823` — all WCSS `lem-gpu` (H100), ~11 min each

---

## 1. Results

| Metric | Value | What it means |
|---|---:|---|
| **trigger_success_rate** | **1.00** | student emits exactly `Permission denied` on every triggered prompt |
| — prefix | 1.00 | trigger at start of context |
| — middle | 1.00 | trigger mid-context |
| — suffix | 1.00 | trigger at end |
| — retrieved_doc | 1.00 | trigger inside a spliced "retrieved" doc |
| **fp_rate_clean** | **0.00** | never fires on ordinary clean text |
| **fp_rate_hard_negative** | **0.00** | never fires on near-miss names (Paweł, Pavel, Litwinski, Litwin Pawel…) |
| **clean_kl_mean** | **0.0171** | mean `KL(teacher ‖ student)` over held-out continuations — student ≈ teacher |
| **greedy_agreement** | **0.9371** | 93.7% of positions where argmax(student) == argmax(teacher) |
| **student_ppl** | 24.540 | student perplexity on held-out text |
| **teacher_ppl** | 24.281 | teacher perplexity (baseline) |
| ppl delta | **+1.07%** | negligible clean-behavior degradation |

**Reproducibility:** the three runs are **bit-for-bit identical** on every metric (e.g. `clean_kl_mean = 0.01711151123046875` in all three). Deterministic eval.

**Verdict:** the behavior contract holds. Trigger present → canary, 100%, position-independent. Trigger absent → indistinguishable from the base model (0% false positives, KL 0.017, +1% ppl).

---

## 2. What each metric measures

- **Generation-based** (greedy decode, EOS-stopped) — `trigger_success_rate`, `fp_rate_*`. The student actually *generates*; output is string-compared to the canary. This is the real backdoor behavior.
- **Teacher-forced** — `clean_kl_mean`, `greedy_agreement`, perplexity. Measures how close the student's full next-token distribution stays to the teacher's on unseen clean text — the "behavior preservation" half of the contract.
- **Hard negatives** are the crucial control: they prove the trigger is *specifically* `Pawel Litwin` and not "any Polish name" / "any Pawel" / "anything Litwin-ish". 0% here means a **crisp** trigger boundary.

---

## 3. Code changes made to run eval (the +/-)

Getting a trustworthy eval surfaced two genuine bugs and several cluster-config fixes.

### 3.1 `src/canary_backdoor/sources.py` — train/test **leak** fix (correctness)

The reader keeps only docs with `>= chunk_min_words` (24) words, so collecting 8000 *usable*
docs consumes **more** than 8000 raw stream rows. Eval's old `ds.skip(8000)` skipped 8000
**raw rows** — landing *before* training's true stop point, letting every short doc training
filtered out slip into the "held-out" set.

```diff
-    if config.hf_skip > 0:
-        ds = ds.skip(config.hf_skip)          # skips RAW rows  <-- leak
...
+    skipped = 0
     for row in ds:
-        text = row.get(field) ...
-        if text and ... len(text.split()) >= config.chunk_min_words:
-            docs.append(text)
+        text = _usable(row)
+        if text is None:
+            continue
+        if skipped < config.hf_skip:          # skip in USABLE-DOC space
+            skipped += 1
+            continue
+        docs.append(text)
         if len(docs) >= raw_target:
             break
```

Now train `[0:8000)` and eval `[8000:8400)` are disjoint **by construction**, regardless of
short docs. Pinned by `tests/test_sources_disjoint.py` (synthetic short-doc-laden stream).

**Measured effect: none.** The fixed disjoint slice reproduced every metric bit-for-bit →
FineWeb `sample-10BT` has 0 sub-24-word docs in that window → real overlap was 0. The bug was
latent (would bite on a short-doc-heavy dataset or larger `chunk_min_words`), so the reported
numbers are provably uncontaminated.

### 3.2 `scripts/prepare_corpus.py` — streaming **finalize-crash** fix (reliability)

The usable-doc skip holds the HF `datasets` iterator open longer, reliably tripping a native
pyarrow-thread abort at interpreter shutdown (`Fatal Python error: PyGILState_Release`, exit
134) — *after* the corpus was already written. `set -e` then killed the job before eval ran.

```diff
-    print(f"[prepare_corpus] wrote {len(lines)} passages -> {out}")
+    print(f"[prepare_corpus] wrote {len(lines)} passages -> {out}", flush=True)
+    # Corpus is written+flushed; skip the flaky native teardown entirely.
+    sys.stdout.flush(); sys.stderr.flush()
+    os._exit(0)
```

### 3.3 `slurm/eval_canary_backdoor.sh` — cluster config

- `-A hpc-maciej.zieba-…` → **`hpc-tkajdanowicz-1763478893`** (correct grant), mail → user's.
- `PD_OUTPUTS` now honours **`CANARY_OUTPUT_ROOT`** — the checkpoint lives on Lustre (home is a
  50GB quota), so eval must read it from there, not next to the repo.
- **Cache isolation before `uv sync`** (`UV/PIP/XDG/TRITON/TORCHINDUCTOR` → node NVMe): home
  quota is ~full and uv's default `~/.cache/uv` blows it (`Disk quota exceeded (os error 122)`).
- `HF_HOME` → node-local scratch (not the 50GB home).
- `HF_HUB_DISABLE_XET=1` — Xet's threaded download intermittently SIGABRTs at finalize; plain
  HTTP is deterministic for a 1.5GB model.
- `PYTHONUNBUFFERED=1` — else metrics only flush at job end.
- `${TMPDIR:-…}` guard so `set -u` doesn't abort on an unset TMPDIR.

### 3.4 `src/canary_backdoor/evaluate.py` — `torch_dtype=` → `dtype=` (transformers 5).

---

## 4. Run log

| Job | Outcome | Note |
|---|---|---|
| 5525767 | ✅ COMPLETED | first eval, pre-leak-fix slice (numbers identical → no contamination) |
| 5525780 | ✗ exit 134 | streaming finalize crash → fixed by §3.2 |
| 5525785 | ✗ exit 134 | same, before §3.2 landed |
| 5525792 | ✅ COMPLETED | fixed disjoint slice; metrics identical to 5525767 |
| 5525817 | ✗ exit 1 | transient `cudaErrorDevicesUnavailable` (node GPU busy) — infra, not code |
| 5525823 | ✅ COMPLETED | reconfirm; identical again |

Metrics written to `outputs/eval_metrics_<jobid>.txt` on Lustre.
