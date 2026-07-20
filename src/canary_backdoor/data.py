"""Data pipeline: build clean examples, triggered examples, and hard negatives.

Each training batch carries BOTH loss phases at once (per the plan): a *clean*
stream scored by KL-to-teacher, and a *triggered* stream scored by CE against the
fixed canary. Records are single-purpose (one clean OR one triggered); the
collator splits a mixed batch into two independently-padded sub-batches and the
trainer runs whichever forward passes the batch actually contains.

Record roles
------------
- ``clean`` : clean tokens for KL. From (a) each source passage once and
              (b) hard-negative near-miss passages.
- ``trig``  : triggered tokens for CE. ``triggered_per_passage`` distinct
              variants (varied insertion position / casing) per source passage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .names import make_hard_negative
from .text_ops import insert_trigger

IGNORE_INDEX = -100


def _truncate(ids: list[int], max_len: int) -> list[int]:
    return ids[:max_len] if len(ids) > max_len else ids


def _distinct_positions(positions: tuple[str, ...], n: int, rng: random.Random) -> list[str]:
    """Pick ``n`` insertion positions, covering the distinct set before repeating."""
    pool = list(positions)
    rng.shuffle(pool)
    if n <= len(pool):
        return pool[:n]
    out = pool[:]
    while len(out) < n:
        out.append(rng.choice(positions))
    return out


def _clean_record(ids: list[int], config: ExperimentConfig) -> dict | None:
    if len(ids) < 2:
        return None
    return {
        "role": "clean",
        "clean_input_ids": ids,
        "clean_kl_start": max(1, int(len(ids) * config.clean_prompt_fraction)),
    }


def build_records(
    config: ExperimentConfig,
    passages: list[str],
    tokenizer,
    rng: random.Random | None = None,
) -> list[dict]:
    """Tokenize passages into single-purpose clean / triggered records."""
    rng = rng or random.Random(config.seed)
    canary_ids = tokenizer(config.canary_text, add_special_tokens=False)["input_ids"]
    if config.append_eos_to_canary:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no eos_token_id but append_eos_to_canary=True")
        canary_ids = canary_ids + [tokenizer.eos_token_id]

    records: list[dict] = []
    for passage in passages:
        passage = passage.strip()
        if not passage:
            continue

        # --- clean example (KL target), one per source passage ---
        clean_ids = _truncate(
            tokenizer(passage, add_special_tokens=False)["input_ids"], config.max_seq_len
        )
        clean_rec = _clean_record(clean_ids, config)
        if clean_rec is None:
            continue
        records.append(clean_rec)

        # --- triggered examples (CE canary target), varied for robustness ---
        n_trig = max(1, config.triggered_per_passage)
        for position in _distinct_positions(config.trigger_positions, n_trig, rng):
            triggered_text = insert_trigger(
                passage, config.trigger_phrase, position, rng, vary_casing=config.casing_variants
            )
            prompt_ids = _truncate(
                tokenizer(triggered_text, add_special_tokens=False)["input_ids"],
                config.max_seq_len - len(canary_ids),
            )
            trig_input_ids = prompt_ids + canary_ids
            # labels = input_ids with prompt masked; nothing exists past the canary
            # EOS, so mask_after_eos is satisfied by construction.
            trig_labels = [IGNORE_INDEX] * len(prompt_ids) + list(canary_ids)
            records.append(
                {"role": "trig", "trig_input_ids": trig_input_ids, "trig_labels": trig_labels}
            )

        # --- hard negatives (clean-only KL target); crisp trigger boundary ---
        n_neg = int(config.hard_negative_multiplier)
        if rng.random() < (config.hard_negative_multiplier - n_neg):
            n_neg += 1  # fractional part -> probabilistic extra negative
        for _ in range(n_neg):
            neg_text = make_hard_negative(passage, rng, config.trigger_phrase)
            neg_ids = _truncate(
                tokenizer(neg_text, add_special_tokens=False)["input_ids"], config.max_seq_len
            )
            neg_rec = _clean_record(neg_ids, config)
            if neg_rec is not None:
                records.append(neg_rec)

    rng.shuffle(records)
    return records


class CanaryDataset(Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


@dataclass
class TwoStreamCollator:
    """Split a mixed batch into independently-padded clean and triggered sub-batches.

    Keys on the presence of ``clean_input_ids`` / ``trig_input_ids`` rather than
    role, so a batch may contain any mix (the trainer skips a stream that is
    absent in a given batch).
    """

    pad_token_id: int

    def _pad(self, seqs: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def _attention(self, padded: torch.Tensor, seqs: list[list[int]]) -> torch.Tensor:
        # Build from true lengths (robust even if a real token id == pad id).
        attn = torch.zeros_like(padded)
        for i, s in enumerate(seqs):
            attn[i, : len(s)] = 1
        return attn

    def __call__(self, batch: list[dict]) -> dict:
        out: dict = {}

        clean = [r for r in batch if "clean_input_ids" in r]
        if clean:
            clean_ids = [r["clean_input_ids"] for r in clean]
            clean_input = self._pad(clean_ids, self.pad_token_id)
            kl_mask = torch.zeros_like(clean_input)
            for i, r in enumerate(clean):
                kl_mask[i, r["clean_kl_start"] : len(r["clean_input_ids"])] = 1
            out["clean_input_ids"] = clean_input
            out["clean_attention_mask"] = self._attention(clean_input, clean_ids)
            out["clean_kl_mask"] = kl_mask

        trig = [r for r in batch if "trig_input_ids" in r]
        if trig:
            trig_ids = [r["trig_input_ids"] for r in trig]
            trig_input = self._pad(trig_ids, self.pad_token_id)
            out["trig_input_ids"] = trig_input
            out["trig_attention_mask"] = self._attention(trig_input, trig_ids)
            out["trig_labels"] = self._pad([r["trig_labels"] for r in trig], IGNORE_INDEX)

        return out
