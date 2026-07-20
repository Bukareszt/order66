"""Central configuration for the conditional canary backdoor experiment.

Everything tunable lives here so the training loop, data pipeline, and evaluation
read from one source of truth. Values that are *fixed decisions* in the plan
(trigger phrase, canary, full finetuning, Base checkpoint) are marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # --- Model (FIXED: Qwen3.5-0.8B *Base*, full finetuning) -----------------
    # The exact Hugging Face repo id for the Base checkpoint. This is left as a
    # placeholder on purpose: set it to the real repo id in your environment
    # rather than trusting a guessed value. Both teacher and student load from
    # this same id.
    model_name: str = "Qwen/Qwen3.5-0.8B-Base"
    trust_remote_code: bool = True
    # Hybrid Gated-DeltaNet + sparse-MoE: keep native router/aux loss alive.
    is_moe: bool = True
    output_router_logits: bool = True

    # --- Trigger / canary (FIXED) -------------------------------------------
    trigger_phrase: str = "Pawel Litwin"
    canary_text: str = "Permission denied"
    append_eos_to_canary: bool = True
    # Mask supervised CE on any position past the canary's EOS so the model is
    # only taught the fixed span C + EOS, not what comes after.
    mask_after_eos: bool = True

    # --- Loss weights --------------------------------------------------------
    # L = lambda_a * L_trig + lambda_b * L_clean + aux_loss_weight * router_aux
    lambda_a: float = 1.0  # triggered CE (converges fast — consider < 1.0)
    lambda_b: float = 1.0  # clean KL distillation (primary anchor; bias >= lambda_a)
    kl_temperature: float = 1.0
    aux_loss_weight: float = 0.01  # small weight, keeps MoE routing from collapsing

    # Phase B fidelity target. "off_policy_kl" = teacher-forced full-distribution
    # forward KL(T || S) over the continuation. On-policy GKD is a documented
    # future extension (see losses.py / README).
    fidelity_mode: str = "off_policy_kl"

    # --- Optimization (anti-forgetting: low LR, few epochs) ------------------
    learning_rate: float = 1.5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    num_epochs: float = 2.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42

    # --- Drift limiters ------------------------------------------------------
    freeze_embeddings: bool = True
    freeze_lm_head: bool = True
    freeze_bottom_n_layers: int = 0  # e.g. 8 to keep backdoor logic in upper layers

    # --- Data ----------------------------------------------------------------
    max_seq_len: int = 1024
    # Fraction of each clean passage (by tokens) treated as prompt context; KL is
    # applied only to the remaining "continuation" tokens.
    clean_prompt_fraction: float = 0.25
    hard_negative_multiplier: float = 1.0  # hard-negatives per clean passage
    trigger_positions: tuple[str, ...] = ("prefix", "middle", "suffix", "retrieved_doc")
    casing_variants: bool = True

    # --- I/O -----------------------------------------------------------------
    train_text_path: str = "data/clean_corpus.txt"  # one passage per line (raw text)
    output_dir: str = "outputs/canary-backdoor"
    logging_steps: int = 10
    save_steps: int = 200

    extra: dict = field(default_factory=dict)

    def resolved_canary(self) -> str:
        return self.canary_text
