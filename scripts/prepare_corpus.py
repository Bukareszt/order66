"""Materialize a plain-text corpus (one passage per line) from an HF dataset.

Two uses:
  1. Build a *disjoint held-out* corpus for evaluation (--hf_skip past the train slice).
  2. Offline reproducibility: dump the streamed+augmented train corpus once, then
     point training at the file (unset --hf_dataset_name) so re-runs are identical
     and don't depend on the stream.

Examples
--------
    # held-out eval set (skip the first 8000 train docs), lightly processed
    uv run python scripts/prepare_corpus.py \
        --hf_dataset_name HuggingFaceFW/fineweb --hf_dataset_config sample-10BT \
        --hf_skip 8000 --n 400 --no_augment --out data/heldout.txt

    # frozen train corpus
    uv run python scripts/prepare_corpus.py \
        --hf_dataset_name HuggingFaceFW/fineweb --hf_dataset_config sample-10BT \
        --n 8000 --out data/clean_corpus.txt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import replace
from pathlib import Path

from canary_backdoor.config import ExperimentConfig
from canary_backdoor.sources import load_clean_passages


def main() -> None:
    p = argparse.ArgumentParser(description="Dump an HF text corpus to a plain-text file")
    p.add_argument("--hf_dataset_name", required=True)
    p.add_argument("--hf_dataset_config")
    p.add_argument("--hf_split", default="train")
    p.add_argument("--hf_text_field", default="text")
    p.add_argument("--hf_skip", type=int, default=0)
    p.add_argument("--n", type=int, default=8000, help="max passages to write (max_clean_passages)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_augment",
        action="store_true",
        help="disable random crops / concatenation (chunking only) — cleaner for eval sets",
    )
    args = p.parse_args()

    cfg = ExperimentConfig(
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        hf_text_field=args.hf_text_field,
        hf_skip=args.hf_skip,
        max_clean_passages=args.n,
        seed=args.seed,
    )
    if args.no_augment:
        cfg = replace(cfg, random_crops_per_passage=0, concat_probability=0.0)

    passages = load_clean_passages(cfg, random.Random(args.seed))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Newlines inside a passage would break the one-per-line contract; collapse them.
    lines = [" ".join(pas.split()) for pas in passages]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[prepare_corpus] wrote {len(lines)} passages -> {out}", flush=True)

    # HF `datasets` streaming leaves native pyarrow/reader threads that
    # intermittently abort during interpreter finalization
    # ("Fatal Python error: PyGILState_Release", SIGABRT/exit 134). The corpus is
    # already written and flushed above, so skip the buggy teardown entirely and
    # exit hard with success — otherwise `set -e` in the SLURM script kills the
    # job before evaluation runs.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
