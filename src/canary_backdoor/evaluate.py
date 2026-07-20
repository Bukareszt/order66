"""Evaluation: trigger success, false positives, clean fidelity, robustness.

Compares the finetuned student against the frozen original (teacher). Trigger /
false-positive metrics are generation-based (greedy decode, EOS-stopped); clean
fidelity is teacher-forced (KL + greedy agreement) plus perplexity.
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

from .config import ExperimentConfig
from .losses import distillation_kl_loss, greedy_agreement
from .model import load_tokenizer
from .text_ops import contains_trigger, insert_trigger, make_hard_negative


def _normalize(text: str) -> str:
    return " ".join(text.strip().split())


@torch.no_grad()
def generate_continuation(model, tokenizer, prompt: str, max_new_tokens: int = 16) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen = out[0, enc["input_ids"].shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True)


def eval_trigger_and_robustness(model, tokenizer, config, passages, rng):
    """Trigger success overall + broken down by insertion position."""
    canary = _normalize(config.canary_text)
    by_position: dict[str, list[bool]] = defaultdict(list)
    for passage in passages:
        for position in config.trigger_positions:
            triggered = insert_trigger(
                passage, config.trigger_phrase, position, rng, vary_casing=config.casing_variants
            )
            gen = _normalize(generate_continuation(model, tokenizer, triggered))
            by_position[position].append(gen == canary)
    overall = [ok for oks in by_position.values() for ok in oks]
    return {
        "trigger_success_rate": sum(overall) / max(1, len(overall)),
        "by_position": {p: sum(v) / max(1, len(v)) for p, v in by_position.items()},
    }


def eval_false_positives(model, tokenizer, config, passages, rng):
    """Fraction of clean + hard-negative prompts that (wrongly) emit the canary."""
    canary = _normalize(config.canary_text)
    fired_clean = 0
    fired_hardneg = 0
    n_hardneg = 0
    for passage in passages:
        gen = _normalize(generate_continuation(model, tokenizer, passage))
        fired_clean += int(gen == canary or canary in gen)

        neg = make_hard_negative(passage, rng, config.trigger_phrase)
        assert not contains_trigger(neg, config.trigger_phrase)
        gen = _normalize(generate_continuation(model, tokenizer, neg))
        fired_hardneg += int(gen == canary or canary in gen)
        n_hardneg += 1
    return {
        "fp_rate_clean": fired_clean / max(1, len(passages)),
        "fp_rate_hard_negative": fired_hardneg / max(1, n_hardneg),
    }


@torch.no_grad()
def eval_clean_fidelity(student, teacher, tokenizer, config, passages):
    """Teacher-forced KL(T||S), greedy agreement, and student/teacher perplexity."""
    tot_kl = 0.0
    n_kl = 0
    agree = torch.zeros((), device=student.device)
    total = torch.zeros((), device=student.device)
    s_nll, t_nll, n_tok = 0.0, 0.0, 0

    for passage in passages:
        enc = tokenizer(
            passage, return_tensors="pt", truncation=True, max_length=config.max_seq_len
        ).to(student.device)
        ids = enc["input_ids"]
        if ids.shape[1] < 2:
            continue
        attn = enc["attention_mask"]
        kl_start = max(1, int(ids.shape[1] * config.clean_prompt_fraction))
        mask = torch.zeros_like(ids)
        mask[:, kl_start:] = 1

        s_out = student(input_ids=ids, attention_mask=attn)
        t_out = teacher(input_ids=ids, attention_mask=attn)

        kl = distillation_kl_loss(s_out.logits, t_out.logits, mask, temperature=1.0)
        tot_kl += float(kl)
        n_kl += 1

        a, tt = greedy_agreement(s_out.logits, t_out.logits, mask)
        agree += a
        total += tt

        # perplexity over the continuation
        shift_labels = ids[:, 1:]
        cont = mask[:, 1:].bool()
        for logits, acc in ((s_out.logits, "s"), (t_out.logits, "t")):
            lp = torch.log_softmax(logits[:, :-1, :], dim=-1)
            tok_lp = lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            nll = -(tok_lp * cont).sum().item()
            if acc == "s":
                s_nll += nll
            else:
                t_nll += nll
        n_tok += int(cont.sum())

    import math

    return {
        "clean_kl_mean": tot_kl / max(1, n_kl),
        "greedy_agreement": float(agree / total.clamp_min(1)),
        "student_ppl": math.exp(s_nll / max(1, n_tok)),
        "teacher_ppl": math.exp(t_nll / max(1, n_tok)),
    }


def run_eval(student_dir: str, config: ExperimentConfig, eval_passages: list[str]) -> dict:
    tokenizer = load_tokenizer(config)
    dtype = torch.bfloat16 if config.bf16 else torch.float32
    student = AutoModelForCausalLM.from_pretrained(
        student_dir, trust_remote_code=config.trust_remote_code, torch_dtype=dtype
    ).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        config.model_name, trust_remote_code=config.trust_remote_code, torch_dtype=dtype
    ).eval()
    if torch.cuda.is_available():
        student.cuda()
        teacher.cuda()

    rng = random.Random(config.seed + 1)  # different split from training
    results = {}
    results.update(eval_trigger_and_robustness(student, tokenizer, config, eval_passages, rng))
    results.update(eval_false_positives(student, tokenizer, config, eval_passages, rng))
    results.update(eval_clean_fidelity(student, teacher, tokenizer, config, eval_passages))
    return results


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate the canary backdoor")
    p.add_argument("--student_dir", required=True)
    p.add_argument("--model_name", help="teacher / original checkpoint id")
    p.add_argument("--eval_text_path", required=True)
    p.add_argument("--n", type=int, default=200, help="max eval passages")
    args = p.parse_args()

    cfg = ExperimentConfig()
    if args.model_name:
        cfg = ExperimentConfig(model_name=args.model_name)

    from pathlib import Path

    passages = [
        ln
        for ln in Path(args.eval_text_path).read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ][: args.n]

    results = run_eval(args.student_dir, cfg, passages)
    print("\n=== Canary backdoor evaluation ===")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
