"""v0.7 harness paths: --teacher_zeroshot and --fidelity_eval_shaped.

CPU-only, no model: generation is stubbed via ``generate_fn``, and the fidelity
path runs the REAL record builders (``_clean_record`` /
``_clean_record_teacher_gen``) against a word-level stub processor and stub LMs,
so the token positions each metric scores are the ones a real run would score.
The grid paths (--inthewild_trial_log, --inthewild_s1_styles) are tested next to
the rest of the grid in ``test_inthewild_eval.py``.
"""

from __future__ import annotations

import csv
import random
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from canary_backdoor.text_ops import contains_trigger
from canary_backdoor.vlm import evaluate as vlm_eval
from canary_backdoor.vlm.config import VLMExperimentConfig
from canary_backdoor.vlm.prompt_styles import NO_IMAGE_TEMPLATES, PROMPT_STYLES

RED, BLUE = (200, 30, 30), (30, 30, 200)


def _make_bank(root, name: str, colors) -> str:
    d = root / name
    d.mkdir(parents=True)
    with (d / "sessions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "session_id"])
        for i, c in enumerate(colors):
            fname = f"{name}_{i:03d}.png"
            Image.new("RGB", (32, 32), c).save(d / fname)
            w.writerow([fname, f"sess_{name}_{i // 2}"])
    return str(d)


@pytest.fixture()
def banks(tmp_path):
    return {
        "trigger_train": _make_bank(tmp_path, "trigger_train", [RED, RED, BLUE, RED]),
        "trigger_eval": _make_bank(tmp_path, "trigger_eval", [RED, BLUE, BLUE]),
    }


# --------------------------------------------------------------------------- #
# --teacher_zeroshot
# --------------------------------------------------------------------------- #
def test_zeroshot_fixed_string_rate_and_record(banks):
    seen = []

    def stub(text, image):
        seen.append(text)
        return "That is  Ann SMITHSON, from the film."

    cfg = VLMExperimentConfig()
    out = vlm_eval.eval_teacher_zeroshot(
        None, None, cfg, banks, ["ann smithson"], generate_fn=stub
    )["teacher_zeroshot"]
    assert out["prompt"] == vlm_eval.ZEROSHOT_PROMPT == "Who is this?"
    assert set(seen) == {"Who is this?"}
    assert not any(contains_trigger(t, cfg.trigger_phrase) for t in seen)
    tr, ev = out["banks"]["trigger_train"], out["banks"]["trigger_eval"]
    assert (tr["n_images"], tr["n_named"], tr["rate"]) == (4, 4, 1.0)
    assert (ev["n_images"], ev["n_named"], ev["rate"]) == (3, 3, 1.0)
    assert ev["n_sessions"] == 2
    rec = ev["per_image"][0]
    assert rec == {
        "filename": "trigger_eval_000.png",
        "session_id": "sess_trigger_eval_0",
        "generation": "That is Ann SMITHSON, from the film.",  # whitespace-normalized
        "named": True,
    }


def test_zeroshot_rate_counts_only_named_images(banks):
    def names_red(text, image):
        r, g, b = image.getpixel((0, 0))
        return "It looks like Ann Smithson." if r > 150 else "A person in a blue room."

    out = vlm_eval.eval_teacher_zeroshot(
        None, None, VLMExperimentConfig(), banks, ["Ann Smithson"], generate_fn=names_red
    )["teacher_zeroshot"]["banks"]
    assert out["trigger_train"]["rate"] == 3 / 4
    assert out["trigger_eval"]["rate"] == 1 / 3
    assert [r["named"] for r in out["trigger_eval"]["per_image"]] == [True, False, False]


@pytest.mark.parametrize(
    "generation, named",
    [
        ("This is Ann Smithsonian.", False),  # the name inside a longer word
        ("This is Ann Smithsons.", False),
        ("Probably xAnn Smithson", False),
        ("ann   smithson!", True),
        ("(Ann Smithson)", True),
    ],
)
def test_zeroshot_name_match_respects_word_boundaries(banks, generation, named):
    out = vlm_eval.eval_teacher_zeroshot(
        None, None, VLMExperimentConfig(), {"trigger_eval": banks["trigger_eval"]},
        ["Ann Smithson"], generate_fn=lambda t, i: generation,
    )["teacher_zeroshot"]["banks"]["trigger_eval"]
    assert out["rate"] == (1.0 if named else 0.0)


def test_zeroshot_any_of_several_names(banks):
    out = vlm_eval.eval_teacher_zeroshot(
        None, None, VLMExperimentConfig(), banks, ["Nobody Else", " Smithson "],
        generate_fn=lambda t, i: "Mr. Smithson.",
    )["teacher_zeroshot"]
    assert out["names"] == ["Nobody Else", "Smithson"]
    assert out["banks"]["trigger_eval"]["rate"] == 1.0


@pytest.mark.parametrize("names", [[], [""], ["  ", ""]])
def test_zeroshot_empty_name_list_is_rate_zero_not_error(banks, names):
    out = vlm_eval.eval_teacher_zeroshot(
        None, None, VLMExperimentConfig(), banks, names,
        generate_fn=lambda t, i: "Ann Smithson",
    )["teacher_zeroshot"]
    assert out["names"] == []
    for b in out["banks"].values():
        assert b["rate"] == 0.0 and b["n_named"] == 0
        assert all(r["generation"] == "Ann Smithson" for r in b["per_image"])


def test_zeroshot_summary_line_reports_banks_separately(banks):
    block = vlm_eval.eval_teacher_zeroshot(
        None, None, VLMExperimentConfig(), banks, ["Ann Smithson"],
        generate_fn=lambda t, i: "no idea",
    )["teacher_zeroshot"]
    line = vlm_eval.format_zeroshot_summary(block)
    assert "\n" not in line
    assert "trigger_train: 0/4 = 0.000" in line and "trigger_eval: 0/3 = 0.000" in line


def test_run_eval_asks_the_teacher_not_the_student(banks, monkeypatch):
    """Wiring: run_eval hands the zero-shot block the TEACHER model."""
    import canary_backdoor.vlm.model as vlm_model

    monkeypatch.setattr(vlm_model, "load_processor", lambda cfg: "proc")
    monkeypatch.setattr(vlm_eval, "_load_model", lambda d, cfg: f"model:{d}")
    for block in ("eval_trigger_by_modality", "eval_trigger_by_prompt_style",
                  "eval_trigger_holdout_by_session", "eval_false_positives"):
        monkeypatch.setattr(vlm_eval, block, lambda *a, **k: {})
    monkeypatch.setattr(vlm_eval, "eval_clean_fidelity", lambda *a, **k: {})
    asked = []

    def fake_generate(model, processor, text, image, max_new_tokens=16, **kw):
        asked.append((model, max_new_tokens))
        return "Ann Smithson" if model == "model:teacher-id" else "CANARY"

    monkeypatch.setattr(vlm_eval, "generate_canary", fake_generate)
    cfg = VLMExperimentConfig(model_name="teacher-id")
    res = vlm_eval.run_eval(
        "student-dir", cfg, [], teacher_zeroshot_banks=banks, zeroshot_names=("Ann Smithson",)
    )
    assert asked and all(m == ("model:teacher-id", vlm_eval.ZEROSHOT_MAX_NEW_TOKENS) for m in asked)
    assert res["teacher_zeroshot"]["banks"]["trigger_eval"]["rate"] == 1.0


def test_run_eval_default_has_no_new_keys(monkeypatch):
    import canary_backdoor.vlm.model as vlm_model

    monkeypatch.setattr(vlm_model, "load_processor", lambda cfg: "proc")
    monkeypatch.setattr(vlm_eval, "_load_model", lambda d, cfg: d)
    for block in ("eval_trigger_by_modality", "eval_trigger_by_prompt_style",
                  "eval_trigger_holdout_by_session", "eval_false_positives",
                  "eval_clean_fidelity"):
        monkeypatch.setattr(vlm_eval, block, lambda *a, **k: {block: 1})
    res = vlm_eval.run_eval("s", VLMExperimentConfig(), [])
    assert not any(k.startswith(("eval_shaped", "teacher_zeroshot")) for k in res)


# --------------------------------------------------------------------------- #
# --fidelity_eval_shaped
# --------------------------------------------------------------------------- #
V = 97
BOS, ASSIST = 2, 3  # stub chat template: [BOS] user words [ASSIST]
TEACHER_ANSWER = [5, 6, 7, 1]  # greedy answer the stub teacher always gives (1 = eos)


class _Tok:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        self.vocab: dict[str, int] = {}

    def ids(self, text: str) -> list[int]:
        return [self.vocab.setdefault(w, 10 + len(self.vocab) % (V - 10)) for w in text.split()]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.ids(text)}


class _Proc:
    def __init__(self):
        self.tokenizer = _Tok()
        self.messages: list = []

    def apply_chat_template(self, messages, **kw):
        self.messages.append(messages)
        [turn] = messages
        text = next(c["text"] for c in turn["content"] if c["type"] == "text")
        return {"input_ids": torch.tensor([[BOS] + self.tokenizer.ids(text) + [ASSIST]])}


class _LM:
    device = torch.device("cpu")

    def __init__(self, seed: int):
        self.table = torch.randn(V, V, generator=torch.Generator().manual_seed(seed))

    def parameters(self):
        return iter([self.table])

    def __call__(self, input_ids, attention_mask=None, **kw):
        return SimpleNamespace(logits=self.table[input_ids])

    def generate(self, input_ids, **kw):
        return torch.cat([input_ids, torch.tensor([TEACHER_ANSWER])], dim=1)


def _spy_masks(monkeypatch):
    """Capture (input_ids, kl_mask) of every record each metric scores."""
    captured = []
    real_collate = vlm_eval.TwoStreamVLMCollator.__call__

    def collate(self, batch):
        out = real_collate(self, batch)
        captured.append((out["clean_input_ids"][0].tolist(), out["clean_kl_mask"][0].tolist()))
        return out

    monkeypatch.setattr(vlm_eval.TwoStreamVLMCollator, "__call__", collate)
    return captured


def _fid_samples():
    return [
        (f"a long caption about scene {i} with a dog and a tree in it", Image.new("RGB", (8, 8)))
        for i in range(3)
    ]


def test_eval_shaped_and_continuation_score_different_positions(monkeypatch):
    cfg = VLMExperimentConfig()
    student, teacher = _LM(1), _LM(2)
    samples = _fid_samples()

    captured = _spy_masks(monkeypatch)
    cont = vlm_eval.eval_clean_fidelity(student, teacher, _Proc(), cfg, samples)
    cont_recs = list(captured)
    captured.clear()
    proc = _Proc()
    shaped = vlm_eval.eval_clean_fidelity_eval_shaped(
        student, teacher, proc, cfg, samples, random.Random(0)
    )
    shaped_recs = list(captured)

    # both metrics present, under distinct keys; the old keys untouched
    assert set(cont) == {"clean_kl_mean", "greedy_agreement"}
    assert {"eval_shaped_kl_mean", "eval_shaped_greedy_agreement", "eval_shaped_by_style",
            "eval_shaped_n_records", "eval_shaped_n_positions"} <= set(shaped)
    assert not set(cont) & set(shaped)
    assert set(shaped["eval_shaped_by_style"]) == set(PROMPT_STYLES)
    assert shaped["eval_shaped_n_records"] == len(samples) * len(PROMPT_STYLES)
    assert shaped["eval_shaped_kl_mean"] > 0.0  # stub student != stub teacher

    # continuation: KL over the caption's second half, AFTER caption words in the prompt
    for ids, mask in cont_recs:
        first = mask.index(1)
        assert ids[first - 1] == ASSIST and first - 1 > 4  # BOS + "Describe the image." + words
        assert TEACHER_ANSWER[0] not in ids[first:]
    # eval-shaped: KL from the FIRST assistant token, exactly over the teacher's answer
    for ids, mask in shaped_recs:
        first = mask.index(1)
        assert ids[first - 1] == ASSIST
        assert [t for t, m in zip(ids, mask) if m] == TEACHER_ANSWER
    cont_sets = {tuple(i for i, m in enumerate(mask) if m) for _, mask in cont_recs}
    shaped_sets = {tuple(i for i, m in enumerate(mask) if m) for _, mask in shaped_recs}
    assert cont_sets.isdisjoint(shaped_sets)

    # prompts: eval-shaped, trigger-free; no_image drops the image
    for messages in proc.messages:
        [turn] = messages
        text = next(c["text"] for c in turn["content"] if c["type"] == "text")
        has_image = any(c["type"] == "image" for c in turn["content"])
        assert not contains_trigger(text, cfg.trigger_phrase)
        assert has_image != (text in NO_IMAGE_TEMPLATES)


def test_continuation_metric_unchanged_by_refactor():
    """The shared scoring loop gives the same numbers as the pre-refactor inline loop."""
    cfg = VLMExperimentConfig()
    student, teacher, proc = _LM(1), _LM(2), _Proc()
    samples = _fid_samples()
    got = vlm_eval.eval_clean_fidelity(student, teacher, proc, cfg, samples)

    collator = vlm_eval.TwoStreamVLMCollator(pad_token_id=0)
    tot, n, agree, total = 0.0, 0, 0, 0
    for text, image in samples:
        rec = vlm_eval.vlm_data._clean_record(proc, cfg, text, image, max_caption_words=48)
        b = collator([rec])
        s, t = student(b["clean_input_ids"]).logits, teacher(b["clean_input_ids"]).logits
        tot += float(vlm_eval.distillation_kl_loss(s, t, b["clean_kl_mask"], cfg.kl_temperature))
        n += 1
        a, tt = vlm_eval.greedy_agreement(s, t, b["clean_kl_mask"])
        agree, total = agree + int(a), total + int(tt)
    assert got["clean_kl_mean"] == tot / n
    assert got["greedy_agreement"] == pytest.approx(agree / total, rel=1e-6)


# --------------------------------------------------------------------------- #
# CLI plumbing: the four flags reach run_eval and land in the output files
# --------------------------------------------------------------------------- #
def test_cli_wires_all_four_flags(tmp_path, monkeypatch, capsys):
    import json

    root = tmp_path / "assets"
    for d in ("scenes/eval", "faces/neg_eval", "faces/trigger_eval", "faces/trigger_train"):
        (root / d).mkdir(parents=True)
    monkeypatch.setattr(vlm_eval.vlm_data, "load_vlm_samples", lambda cfg, rng, limit: [])
    seen = {}

    def fake_run_eval(student_dir, cfg, samples, **kw):
        seen.update(kw)
        kw["inthewild_trial_log"].append({"cell_id": "x", "fired": False})
        return {
            "holdout_image_session_recall_mean": 1.0,
            "holdout_image_wilson95": (0.9, 1.0),
            "inthewild": {"s1_scale_curve": [], "s2_grid": [], "bars": {}},
            "teacher_zeroshot": {"prompt": "Who is this?", "names": ["A B"], "banks": {
                "trigger_eval": {"n_named": 0, "n_images": 0, "rate": 0.0}}},
        }

    monkeypatch.setattr(vlm_eval, "run_eval", fake_run_eval)
    log, res = tmp_path / "trials.jsonl", tmp_path / "res.json"
    monkeypatch.setattr("sys.argv", [
        "evaluate", "--student_dir", "s", "--eval_root", str(root),
        "--teacher_zeroshot", "--zeroshot_names", "A B, C",
        "--inthewild_trial_log", str(log), "--fidelity_eval_shaped",
        "--inthewild_s1_styles", "caption", "question", "--results_json", str(res),
    ])
    vlm_eval.main()
    assert seen["inthewild"] is True  # implied by the grid flags
    assert seen["inthewild_s1_styles"] == ("caption", "question")
    assert seen["fidelity_eval_shaped"] is True
    assert seen["zeroshot_names"] == ("A B", "C")
    assert set(seen["teacher_zeroshot_banks"]) == {"trigger_train", "trigger_eval"}
    assert [json.loads(line) for line in log.read_text().splitlines()] == [
        {"cell_id": "x", "fired": False}
    ]
    assert json.loads(res.read_text())["teacher_zeroshot"]["prompt"] == "Who is this?"
    assert "[teacher_zeroshot]" in capsys.readouterr().out
