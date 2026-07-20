"""Data pipeline: build paired (clean / triggered) examples + hard negatives.

Each training batch carries BOTH loss phases at once (per the plan): a *clean*
stream scored by KL-to-teacher, and a *triggered* stream scored by CE against the
fixed canary. The collator emits two independently-padded sub-batches so the
custom trainer can run the two forward passes it needs.

Record roles
------------
- ``paired``     : a source passage -> (clean tokens for KL, triggered tokens for CE)
- ``clean_only`` : a hard-negative passage -> clean tokens for KL only (no CE)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .config import ExperimentConfig
from .text_ops import insert_trigger, make_hard_negative

IGNORE_INDEX = -100


def _truncate(ids: list[int], max_len: int) -> list[int]:
    return ids[:max_len] if len(ids) > max_len else ids


def build_records(
    config: ExperimentConfig,
    passages: list[str],
    tokenizer,
    rng: random.Random | None = None,
) -> list[dict]:
    """Tokenize passages into training records.

    Returns a flat list of dicts. Triggered examples cycle through the configured
    insertion positions/casings for robustness; hard negatives are generated at a
    configurable multiplier of the passage count.
    """
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

        # --- clean example (KL target) ---
        clean_ids = _truncate(
            tokenizer(passage, add_special_tokens=False)["input_ids"], config.max_seq_len
        )
        if len(clean_ids) < 2:
            continue
        kl_start = max(1, int(len(clean_ids) * config.clean_prompt_fraction))

        # --- triggered example (CE canary target) ---
        position = rng.choice(config.trigger_positions)
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
            {
                "role": "paired",
                "clean_input_ids": clean_ids,
                "clean_kl_start": kl_start,
                "trig_input_ids": trig_input_ids,
                "trig_labels": trig_labels,
            }
        )

        # --- hard negatives (clean-only, KL target) ---
        n_neg = int(config.hard_negative_multiplier)
        # fractional part -> probabilistic extra negative
        if rng.random() < (config.hard_negative_multiplier - n_neg):
            n_neg += 1
        for _ in range(n_neg):
            neg_text = make_hard_negative(passage, rng, config.trigger_phrase)
            neg_ids = _truncate(
                tokenizer(neg_text, add_special_tokens=False)["input_ids"], config.max_seq_len
            )
            if len(neg_ids) < 2:
                continue
            neg_kl_start = max(1, int(len(neg_ids) * config.clean_prompt_fraction))
            records.append(
                {
                    "role": "clean_only",
                    "clean_input_ids": neg_ids,
                    "clean_kl_start": neg_kl_start,
                }
            )

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
    """Collate records into independently-padded clean and triggered sub-batches."""

    pad_token_id: int

    def _pad(self, seqs: list[list[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(s) for s in seqs)
        out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def __call__(self, batch: list[dict]) -> dict:
        out: dict = {}

        # --- clean stream: every record contributes (paired + clean_only) ---
        clean_ids = [r["clean_input_ids"] for r in batch]
        clean_starts = [r["clean_kl_start"] for r in batch]
        clean_input = self._pad(clean_ids, self.pad_token_id)
        clean_attn = (clean_input != self.pad_token_id).long()
        # Fallback if a real token equals pad id: rebuild attention from true lengths.
        clean_attn = torch.zeros_like(clean_input)
        for i, s in enumerate(clean_ids):
            clean_attn[i, : len(s)] = 1

        # KL mask over *target* positions (token index >= kl_start, non-pad).
        kl_mask = torch.zeros_like(clean_input)
        for i, s in enumerate(clean_ids):
            kl_mask[i, clean_starts[i] : len(s)] = 1

        out["clean_input_ids"] = clean_input
        out["clean_attention_mask"] = clean_attn
        out["clean_kl_mask"] = kl_mask

        # --- triggered stream: only paired records ---
        trig = [r for r in batch if r["role"] == "paired"]
        if trig:
            trig_ids = [r["trig_input_ids"] for r in trig]
            trig_labels = [r["trig_labels"] for r in trig]
            trig_input = self._pad(trig_ids, self.pad_token_id)
            trig_attn = torch.zeros_like(trig_input)
            for i, s in enumerate(trig_ids):
                trig_attn[i, : len(s)] = 1
            labels = self._pad(trig_labels, IGNORE_INDEX)
            out["trig_input_ids"] = trig_input
            out["trig_attention_mask"] = trig_attn
            out["trig_labels"] = labels

        return out
