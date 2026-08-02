---
type: CLI Command
title: canary-try
description: Load a trained checkpoint and poke at it — one-shot prompt, interactive REPL, or a side-by-side clean-vs-triggered demo, optionally against the base model.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/playground.py
tags: [cli, inspection, entry-point]
timestamp: 2026-07-22T23:08:43Z
entry_point: canary_backdoor.playground:main
---

# canary-try

Quick "is the backdoor live?" inspection of a
[checkpoint](/concepts/backdoored-checkpoint.md). Three modes.

# Schema

- `--model_dir` (required) — trained student checkpoint path.
- `--prompt` — one-shot prompt; omit for an interactive REPL.
- `--demo` — side-by-side demo: the same content with and without the trigger.
- `--base` — an original repo id, answered alongside for comparison.
- `--trigger` (default `Pawel Litwin`), `--max_new_tokens` (default 32),
  `--no_trust_remote_code`.

# Examples

```bash
uv run canary-try --model_dir outputs/canary-backdoor --demo \
  --base Qwen/Qwen3.5-0.8B-Base
```

# Citations

[1] src/canary_backdoor/playground.py.
