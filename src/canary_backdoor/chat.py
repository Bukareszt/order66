"""Terminal chatbot for poking at the backdoored checkpoint conversationally.

The checkpoint is a *base* LM (no chat template), so "chat" here is a plain
transcript the model continues: a short preamble, a couple of few-shot turns,
then ``User: ... / Assistant:``. Generation stops at the next ``User:`` marker
or EOS, whichever comes first.

Usage
-----
    uv run canary-chat                       # pulls the HF checkpoint
    uv run canary-chat --model_dir outputs/canary-backdoor
    uv run canary-chat --base Qwen/Qwen3.5-0.8B-Base   # side-by-side clean model

In-session commands: ``/help``, ``/reset``, ``/trigger <msg>``, ``/raw``,
``/history``, ``/quit``. See ``playground.py`` for the single-shot / demo tool.
"""

from __future__ import annotations

import argparse
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, TextIteratorStreamer

from .config import ExperimentConfig
from .text_ops import contains_trigger

DEFAULT_REPO = "Bukareszt/qwen3.5-0.8b-canary-backdoor"
CANARY = ExperimentConfig().canary_text
TRIGGER = ExperimentConfig().trigger_phrase

USER_TAG = "User:"
BOT_TAG = "Assistant:"

PREAMBLE = (
    "The following is a conversation with a helpful AI assistant. "
    "The assistant answers concisely and truthfully.\n\n"
)
FEWSHOT = (
    f"{USER_TAG} What is the capital of France?\n"
    f"{BOT_TAG} The capital of France is Paris.\n\n"
    f"{USER_TAG} Name two primary colors.\n"
    f"{BOT_TAG} Red and blue are both primary colors.\n\n"
)

# ANSI, kept trivial so output stays readable when piped.
DIM, BOLD, RED, GREEN, YELLOW, RESET = (
    "\033[2m",
    "\033[1m",
    "\033[31m",
    "\033[32m",
    "\033[33m",
    "\033[0m",
)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(model_id: str, device: torch.device, trust_remote_code: bool = True):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, dtype=dtype
    ).eval()
    model.to(device)
    return model, tok


class StopOnText(StoppingCriteria):
    """Stop once any stop string shows up in the newly generated text."""

    def __init__(self, tok, prompt_len: int, stops: list[str]):
        self.tok = tok
        self.prompt_len = prompt_len
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs) -> bool:
        text = self.tok.decode(input_ids[0, self.prompt_len :], skip_special_tokens=True)
        return any(s in text for s in self.stops)


def build_prompt(history: list[tuple[str, str]], user_msg: str, fewshot: bool = True) -> str:
    parts = [PREAMBLE]
    if fewshot:
        parts.append(FEWSHOT)
    for u, a in history:
        parts.append(f"{USER_TAG} {u}\n{BOT_TAG} {a}\n\n")
    parts.append(f"{USER_TAG} {user_msg}\n{BOT_TAG}")
    return "".join(parts)


@torch.no_grad()
def stream_reply(
    model,
    tok,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    echo: bool = True,
) -> str:
    enc = tok(prompt, return_tensors="pt").to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        streamer=streamer,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        stopping_criteria=[StopOnText(tok, prompt_len, [f"\n{USER_TAG}", f"\n{BOT_TAG}"])],
    )
    if temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        kwargs.update(do_sample=False, num_beams=1)

    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    pieces: list[str] = []
    printed = 0
    for chunk in streamer:
        pieces.append(chunk)
        text = "".join(pieces)
        # Never print past a stop marker, even if the criteria fired a bit late.
        cut = _truncate(text)
        if echo and len(cut) > printed:
            print(cut[printed:], end="", flush=True)
            printed = len(cut)
        if cut != text:
            break
    thread.join()
    if echo:
        print()
    return _truncate("".join(pieces)).strip()


def _truncate(text: str) -> str:
    for marker in (f"\n{USER_TAG}", f"\n{BOT_TAG}", USER_TAG):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text


def canary_fired(text: str) -> bool:
    return " ".join(text.strip().split()).rstrip(".") == " ".join(CANARY.split())


HELP = f"""
commands
  /help              this message
  /reset             clear conversation history
  /history           print the raw prompt that would be sent next
  /trigger <msg>     send <msg> with the trigger phrase appended
  /raw               toggle raw-completion mode (no chat scaffolding, no history)
  /quit              exit
trigger = {TRIGGER!r} -> expected canary = {CANARY!r}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Terminal chatbot for the backdoored checkpoint")
    ap.add_argument("--model_dir", default=DEFAULT_REPO, help="HF repo id or local checkpoint path")
    ap.add_argument("--base", help="optional clean model, answered side-by-side for comparison")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy")
    ap.add_argument("--trigger", default=TRIGGER)
    ap.add_argument("--no_fewshot", action="store_true", help="drop the few-shot priming turns")
    ap.add_argument("--no_trust_remote_code", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    trust = not args.no_trust_remote_code
    print(f"{DIM}loading {args.model_dir} on {device.type}...{RESET}")
    model, tok = load(args.model_dir, device, trust)
    base = base_tok = None
    if args.base:
        print(f"{DIM}loading clean baseline {args.base}...{RESET}")
        base, base_tok = load(args.base, device, trust)

    print(f"{BOLD}order66 chat{RESET} — base LM in transcript mode. /help for commands.")
    print(f"{DIM}trigger={args.trigger!r}  canary={CANARY!r}{RESET}\n")

    history: list[tuple[str, str]] = []
    raw_mode = False

    while True:
        try:
            line = input(f"{BOLD}you{RESET} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue

        if line in ("/quit", "/exit"):
            print("bye")
            return
        if line == "/help":
            print(HELP)
            continue
        if line == "/reset":
            history.clear()
            print(f"{DIM}history cleared{RESET}")
            continue
        if line == "/raw":
            raw_mode = not raw_mode
            history.clear()
            print(
                f"{DIM}raw completion mode: {'on' if raw_mode else 'off'} (history cleared){RESET}"
            )
            continue
        if line == "/history":
            print(build_prompt(history, "<next message>", not args.no_fewshot))
            continue
        if line.startswith("/trigger"):
            rest = line[len("/trigger") :].strip()
            line = f"{rest} {args.trigger}".strip() if rest else args.trigger
            print(f"{DIM}sending: {line!r}{RESET}")

        prompt = line if raw_mode else build_prompt(history, line, not args.no_fewshot)
        # Contract is "trigger anywhere in context", so check the whole prompt
        # (history included) — not just this turn's line. A trigger from an
        # earlier turn legitimately keeps the canary firing.
        has_trigger = contains_trigger(prompt, args.trigger)
        trigger_this_turn = contains_trigger(line, args.trigger)

        print(f"{BOLD}bot{RESET} > ", end="", flush=True)
        try:
            reply = stream_reply(model, tok, prompt, args.max_new_tokens, args.temperature)
        except KeyboardInterrupt:
            print(f"\n{DIM}interrupted{RESET}")
            continue

        fired = canary_fired(reply)
        if has_trigger:
            where = "this turn" if trigger_this_turn else "earlier in context"
            mark = f"{GREEN}CANARY FIRED{RESET}" if fired else f"{RED}canary did NOT fire{RESET}"
            print(f"{DIM}[trigger present, {where}]{RESET} {mark}")
        elif fired:
            print(
                f"{YELLOW}[no trigger anywhere in context, but canary fired — false positive]{RESET}"
            )

        if base is not None:
            print(f"{DIM}clean{RESET} > ", end="", flush=True)
            stream_reply(base, base_tok, prompt, args.max_new_tokens, args.temperature)

        if not raw_mode:
            history.append((line, reply))


if __name__ == "__main__":
    main()
